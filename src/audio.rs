//! Decoding, format probing and sample-rate conversion.

use std::fs::File;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader, SeekTo};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

pub(crate) struct Audio {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl Audio {
    fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32 / self.channels as f32
    }
}

/// Opens the container. This is also the format check: whatever symphonia can
/// probe is what we support, so there is no extension list to keep in sync.
pub(crate) fn open(path: &str) -> Result<Box<dyn FormatReader>, String> {
    let file = File::open(path).map_err(|e| format!("Cannot open '{path}': {e}"))?;
    let mut hint = Hint::new();
    if let Some(ext) = Path::new(path).extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    symphonia::default::get_probe()
        .format(
            &hint,
            MediaSourceStream::new(Box::new(file), Default::default()),
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map(|probed| probed.format)
        .map_err(|e| format!("Unsupported or corrupt audio file '{path}': {e}"))
}

/// What the container header claims, without touching the audio payload.
pub(crate) struct TrackInfo {
    sample_rate: Option<u32>,
    channels: Option<u16>,
    frames: Option<u64>,
    delay: u64,
    padding: u64,
}

pub(crate) fn probe_info(path: &str) -> Result<TrackInfo, String> {
    let format = open(path)?;
    let track = format.default_track().ok_or("No audio track found")?;
    Ok(TrackInfo {
        sample_rate: track.codec_params.sample_rate,
        channels: track.codec_params.channels.map(|c| c.count() as u16),
        frames: track.codec_params.n_frames,
        delay: priming_frames(path, track.codec_params.delay).unwrap_or(0) as u64,
        padding: track.codec_params.padding.unwrap_or(0) as u64,
    })
}

/// Samples to reserve up front: the whole file for short clips, at most a
/// minute for long ones so a two-hour lossless file does not reserve gigabytes
/// before the first packet is decoded.
fn prealloc_capacity(frames: Option<u64>, channels: Option<u16>, rate: Option<u32>) -> usize {
    let (Some(frames), Some(channels)) = (frames, channels) else {
        return 0;
    };
    let rate = rate.unwrap_or(48_000) as u64;
    let wanted = frames.saturating_mul(channels as u64);
    let cap = wanted.min(rate * channels as u64 * 60);
    usize::try_from(cap).unwrap_or(0)
}

/// Iterates the child boxes of an ISO-BMFF buffer: `(fourcc, body)` pairs,
/// where `body` excludes the box header.
fn mp4_children(data: &[u8]) -> Vec<(&[u8; 4], &[u8])> {
    let mut children = Vec::new();
    let mut offset = 0usize;
    while offset + 8 <= data.len() {
        let size = u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]) as usize;
        let kind: &[u8; 4] = data[offset + 4..offset + 8].try_into().unwrap();
        let (header, size) = match size {
            1 => {
                if offset + 16 > data.len() {
                    break;
                }
                let large = u64::from_be_bytes(data[offset + 8..offset + 16].try_into().unwrap());
                (16usize, usize::try_from(large).unwrap_or(usize::MAX))
            }
            0 => (8usize, data.len() - offset),
            size => (8usize, size),
        };
        if size < header || offset + size > data.len() {
            break;
        }
        children.push((kind, &data[offset + header..offset + size]));
        offset += size;
    }
    children
}

fn mp4_child<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    mp4_children(data)
        .into_iter()
        .find(|(k, _)| *k == kind)
        .map(|(_, body)| body)
}

/// AAC decoders emit priming frames before the real signal, and MP4 edit lists
/// record how many via an `elst` `media_time`. Symphonia parses that box but
/// does not expose it, so read the first audio track's edit list directly.
/// Returns `None` for anything that does not look like a sound track edit.
fn mp4_priming(path: &str) -> Option<u64> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = File::open(path).ok()?;
    let total = file.metadata().ok()?.len();

    // Walk top-level boxes until `moov`, which may sit after a large `mdat`.
    let moov = {
        let mut offset = 0u64;
        loop {
            if offset + 8 > total {
                return None;
            }
            file.seek(SeekFrom::Start(offset)).ok()?;
            let mut header = [0u8; 16];
            file.read_exact(&mut header[..8]).ok()?;
            let mut size = u32::from_be_bytes(header[..4].try_into().ok()?) as u64;
            let kind: [u8; 4] = header[4..8].try_into().ok()?;
            let mut header_len = 8u64;
            if size == 1 {
                file.read_exact(&mut header[8..16]).ok()?;
                size = u64::from_be_bytes(header[8..16].try_into().ok()?);
                header_len = 16;
            } else if size == 0 {
                size = total - offset;
            }
            if size < header_len {
                return None;
            }
            if &kind == b"moov" {
                // A moov box is metadata; anything this large is not a normal file.
                let payload = size - header_len;
                if payload > 32 * 1024 * 1024 {
                    return None;
                }
                let mut buf = vec![0u8; payload as usize];
                file.read_exact(&mut buf).ok()?;
                break buf;
            }
            offset += size;
        }
    };

    for (kind, trak) in mp4_children(&moov) {
        if kind != b"trak" {
            continue;
        }
        let Some(mdia) = mp4_child(trak, b"mdia") else {
            continue;
        };
        let Some(hdlr) = mp4_child(mdia, b"hdlr") else {
            continue;
        };
        if hdlr.get(8..12) != Some(b"soun") {
            continue;
        }
        let edts = mp4_child(trak, b"edts")?;
        let elst = mp4_child(edts, b"elst")?;
        let version = *elst.first()?;
        let count = u32::from_be_bytes(elst.get(4..8)?.try_into().ok()?);
        if version > 1 || count == 0 {
            return None;
        }
        // Entry: segment_duration, then media_time (in media timescale = frames).
        let media_time = if version == 1 {
            i64::from_be_bytes(elst.get(16..24)?.try_into().ok()?)
        } else {
            i32::from_be_bytes(elst.get(12..16)?.try_into().ok()?) as i64
        };
        return (media_time > 0).then_some(media_time as u64);
    }

    None
}

/// True when the file starts with an ISO-BMFF `ftyp` box.
fn is_mp4(path: &str) -> bool {
    use std::io::Read;

    let mut header = [0u8; 8];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut header))
        .map(|_| &header[4..8] == b"ftyp")
        .unwrap_or(false)
}

/// Encoder priming in frames: what the container declared, or what the MP4 edit
/// list says when only the muxer knows.
fn priming_frames(path: &str, declared: Option<u32>) -> Option<u32> {
    declared.or_else(|| {
        if !is_mp4(path) {
            return None;
        }
        mp4_priming(path).map(|frames| frames.min(u32::MAX as u64) as u32)
    })
}

/// Appends a decoded chunk, skipping the first `skip` samples of the file.
/// Done at append time so the encoder delay never costs a memmove of the whole
/// buffer afterwards. Returns the skip still owed.
fn append_skipping(samples: &mut Vec<f32>, chunk: &[f32], skip: usize) -> usize {
    if skip >= chunk.len() {
        return skip - chunk.len();
    }
    samples.extend_from_slice(&chunk[skip..]);
    0
}

/// True for `.mp3`, the one format we decode with minimp3 instead of symphonia.
fn is_mp3(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("mp3"))
        .unwrap_or(false)
}

/// Decodes MP3 with minimp3, which is ~45% faster than symphonia's decoder on
/// the same file. Container metadata still comes from symphonia so the gapless
/// trimming matches the other formats.
fn decode_mp3(path: &str) -> Result<Audio, String> {
    use std::io::BufReader;

    let info = probe_info(path)?;
    let sample_rate = info.sample_rate.ok_or("Unknown sample rate")?;
    let channels = info.channels.ok_or("Unknown channel layout")?.max(1);
    let ch = channels as usize;
    let delay = info.delay;
    let padding = info.padding;
    let limit = info
        .frames
        .map(|frames| frames.saturating_sub(delay + padding))
        .unwrap_or(u64::MAX);

    let file = File::open(path).map_err(|e| format!("Cannot open '{path}': {e}"))?;
    let mut decoder = minimp3::Decoder::new(BufReader::new(file));
    let mut samples = Vec::with_capacity(prealloc_capacity(
        info.frames,
        Some(channels),
        Some(sample_rate),
    ));
    let mut skip = delay * ch as u64;
    let mut emitted = 0u64;

    loop {
        let frame = match decoder.next_frame() {
            Ok(frame) => frame,
            Err(minimp3::Error::Eof) => break,
            Err(e) => return Err(format!("Decode error: {e}")),
        };
        let mut chunk: Vec<f32> = frame.data.iter().map(|&s| s as f32 / 32768.0).collect();
        if frame.channels as u16 != channels {
            chunk = convert_channels(&chunk, frame.channels as u16, channels);
        }

        let frames = (chunk.len() / ch) as u64;
        let dropped = skip.min(frames);
        skip -= dropped;
        let start = (dropped * ch as u64) as usize;
        let mut take = frames - dropped;
        if emitted + take > limit {
            take = limit.saturating_sub(emitted);
        }
        emitted += take;
        let end = start + take as usize * ch;
        samples.extend_from_slice(&chunk[start.min(chunk.len())..end.min(chunk.len())]);
    }

    if samples.is_empty() {
        return Err(format!("No audio data decoded from '{path}'"));
    }

    Ok(Audio {
        sample_rate,
        channels,
        samples,
    })
}

pub(crate) fn decode(path: &str) -> Result<Audio, String> {
    if is_mp3(path) {
        // Never fail a file symphonia could still read.
        if let Ok(audio) = decode_mp3(path) {
            return Ok(audio);
        }
    }
    decode_symphonia(path)
}

/// Options for `load`. Defaults reproduce a plain full decode.
#[derive(Default, Clone, Copy)]
pub(crate) struct LoadOptions {
    pub sample_rate: Option<u32>,
    pub mono: bool,
    pub offset: f64,
    pub duration: Option<f64>,
}

/// Decodes a window of a file through the streaming reader, then applies the
/// requested channel fold and sample rate.
pub(crate) fn decode_with_options(path: &str, options: LoadOptions) -> Result<Audio, String> {
    let mut audio = if options.offset > 0.0 || options.duration.is_some() {
        decode_range(path, options.offset, options.duration)?
    } else {
        decode(path)?
    };

    if options.mono && audio.channels > 1 {
        audio.samples = convert_channels(&audio.samples, audio.channels, 1);
        audio.channels = 1;
    }

    if let Some(rate) = options.sample_rate {
        if rate != audio.sample_rate {
            audio.samples = resample_all(
                &audio.samples,
                audio.sample_rate,
                rate,
                audio.channels as usize,
            );
            audio.sample_rate = rate;
        }
    }

    Ok(audio)
}

fn decode_range(path: &str, offset: f64, duration: Option<f64>) -> Result<Audio, String> {
    let mut stream = Stream::open(path)?;
    if offset > 0.0 {
        stream.seek(offset)?;
    }

    let mut skip = if offset > 0.0 {
        0
    } else {
        stream.delay_frames()
    };
    let mut samples = Vec::new();
    let mut frames = 0u64;
    let mut limit: Option<u64> = None;

    while let Some(chunk) = stream.next_chunk(&mut skip)? {
        let channels = stream.channels.max(1) as usize;
        if limit.is_none() {
            limit = duration
                .map(|seconds| (seconds.max(0.0) * stream.sample_rate as f64).round() as u64);
        }
        let available = chunk.len() / channels;
        let take = match limit {
            Some(limit) => available.min(limit.saturating_sub(frames) as usize),
            None => available,
        };
        samples.extend_from_slice(&chunk[..take * channels]);
        frames += take as u64;
        if limit.is_some_and(|limit| frames >= limit) {
            break;
        }
    }

    // An empty window is a valid request (duration=0, or an offset at the end),
    // not a broken file.
    Ok(Audio {
        sample_rate: stream.sample_rate,
        channels: stream.channels,
        samples,
    })
}

/// Decodes bytes that are already in memory. Used by `load_bytes` and by the
/// decoder fuzz test, which cannot afford a temp file per case.
pub(crate) fn decode_bytes(data: &[u8]) -> Result<Audio, String> {
    let reader = Box::new(std::io::Cursor::new(data.to_vec()));
    let mss = MediaSourceStream::new(reader, Default::default());
    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("Unsupported or corrupt audio data: {e}"))?;
    decode_format(probed.format, None, "audio data")
}

fn decode_symphonia(path: &str) -> Result<Audio, String> {
    let format = open(path)?;
    let priming = if is_mp4(path) {
        mp4_priming(path)
    } else {
        None
    };
    decode_format(format, priming, path)
}

/// Shared decode loop: a probed container, an optional MP4 priming value, and a
/// label for error messages.
fn decode_format(
    mut format: Box<dyn FormatReader>,
    priming: Option<u64>,
    label: &str,
) -> Result<Audio, String> {
    // Codec params are not always complete (AAC in MP4 has no channel layout
    // in the container), so fall back to the first decoded frame's spec.
    let (mut sample_rate, mut channels, n_frames, mut delay, padding, mut decoder) = {
        let track = format.default_track().ok_or("No audio track found")?;
        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| format!("Unsupported codec: {e}"))?;
        (
            track.codec_params.sample_rate,
            track.codec_params.channels.map(|c| c.count() as u16),
            track.codec_params.n_frames,
            track.codec_params.delay,
            track.codec_params.padding,
            decoder,
        )
    };

    // One allocation for the whole file instead of repeated Vec growth.
    // MP3/OGG declare the delay; AAC in MP4 only records it in the edit list.
    delay = delay.or_else(|| priming.map(|frames| frames.min(u32::MAX as u64) as u32));

    let mut samples = Vec::with_capacity(prealloc_capacity(n_frames, channels, sample_rate));
    let mut buffer: Option<SampleBuffer<f32>> = None;
    // Samples still owed to the encoder delay. Resolved once the channel count
    // is known: AAC has no channel layout in the container until the first
    // packet is decoded, and the skip is per channel.
    let mut leading_skip: Option<usize> = None;

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(e)) => {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    break;
                }
                return Err(format!("Read error: {e}"));
            }
            Err(SymphoniaError::ResetRequired) => break,
            Err(e) => return Err(format!("Read error: {e}")),
        };

        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            // A truncated tail is not a reason to throw away the whole file.
            Err(SymphoniaError::IoError(e)) => {
                if e.kind() == std::io::ErrorKind::UnexpectedEof {
                    break;
                }
                return Err(format!("Decode error: {e}"));
            }
            Err(e) => return Err(format!("Decode error: {e}")),
        };

        if channels.is_none() || sample_rate.is_none() {
            channels.get_or_insert(decoded.spec().channels.count() as u16);
            sample_rate.get_or_insert(decoded.spec().rate);
        }

        // SampleBuffer does the integer/24-bit/f64 -> f32 conversion and the
        // interleaving for us, for every sample format symphonia can produce.
        let needs_new_buffer = buffer
            .as_ref()
            .is_none_or(|b| b.capacity() < decoded.capacity());
        if needs_new_buffer {
            buffer = Some(SampleBuffer::<f32>::new(
                decoded.capacity() as u64,
                *decoded.spec(),
            ));
        }
        let buffer = buffer.as_mut().expect("buffer was just created");
        buffer.copy_interleaved_ref(decoded);
        let owed = *leading_skip
            .get_or_insert_with(|| delay.unwrap_or(0) as usize * channels.unwrap_or(0) as usize);
        leading_skip = Some(append_skipping(&mut samples, buffer.samples(), owed));
    }

    if samples.is_empty() {
        return Err(format!("No audio data decoded from '{label}'"));
    }

    let channels = channels.ok_or("Unknown channel layout")?;
    let ch = channels as usize;

    // Trailing padding is dropped from the end, which is free.
    if let Some(padding) = padding {
        let keep = samples
            .len()
            .saturating_sub((padding as usize).saturating_mul(ch));
        samples.truncate(keep);
    }
    if samples.is_empty() {
        return Err(format!(
            "'{label}' contains no audio after trimming encoder padding"
        ));
    }

    Ok(Audio {
        sample_rate: sample_rate.ok_or("Unknown sample rate")?,
        channels,
        samples,
    })
}

/// Converts an interleaved buffer from `src` to `dst` channels. Down-mixing
/// stereo averages both channels; everything else repeats or truncates
/// channels in order, which is what a simple mixer does.
pub(crate) fn convert_channels(samples: &[f32], src: u16, dst: u16) -> Vec<f32> {
    let (src, dst) = (src as usize, dst as usize);
    if src == dst || src == 0 || dst == 0 {
        return samples.to_vec();
    }

    if dst == 1 && src > 1 {
        // Sum and average every channel. A surround layout would want proper
        // downmix coefficients, but averaging beats silently keeping only the
        // left channel.
        let frames = samples.len() / src;
        let scale = 1.0 / src as f32;
        return (0..frames)
            .map(|frame| samples[frame * src..(frame + 1) * src].iter().sum::<f32>() * scale)
            .collect();
    }

    let frames = samples.len() / src;
    let mut out = Vec::with_capacity(frames * dst);
    for frame in 0..frames {
        for channel in 0..dst {
            out.push(samples[frame * src + channel % src]);
        }
    }
    out
}

/// Decodes one packet to learn the real sample rate and channel count when the
/// container header does not carry them (AAC in MP4).
fn sniff_spec(path: &str) -> Result<(u32, u16), String> {
    let mut format = open(path)?;
    let (mut decoder, mut channel_hint) = {
        let track = format.default_track().ok_or("No audio track found")?;
        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| format!("Unsupported codec: {e}"))?;
        (
            decoder,
            track.codec_params.channels.map(|c| c.count() as u16),
        )
    };
    let packet = format
        .next_packet()
        .map_err(|e| format!("Read error: {e}"))?;
    let decoded = decoder
        .decode(&packet)
        .map_err(|e| format!("Decode error: {e}"))?;
    let channels = channel_hint
        .get_or_insert(decoded.spec().channels.count() as u16)
        .to_owned();
    if channels == 0 {
        return Err("Unknown channel layout".to_string());
    }
    Ok((decoded.spec().rate, channels))
}

/// Track length in seconds from the header, without decoding.
pub(crate) fn duration_seconds(path: &str) -> Result<Option<f64>, String> {
    let info = probe_info(path)?;
    match (info.sample_rate, info.frames) {
        (Some(rate), Some(frames)) => {
            let frames = frames.saturating_sub(info.delay + info.padding);
            Ok(Some(frames as f64 / rate as f64))
        }
        _ => Ok(None),
    }
}

pub(crate) fn info_or_decode(path: &str) -> Result<(u32, u16, f32), String> {
    let info = probe_info(path)?;
    let (mut rate, mut channels) = (info.sample_rate, info.channels);

    if rate.is_none() || channels.is_none() {
        let (sniffed_rate, sniffed_channels) = sniff_spec(path)?;
        rate.get_or_insert(sniffed_rate);
        channels.get_or_insert(sniffed_channels);
    }

    if let (Some(rate), Some(channels), Some(frames)) = (rate, channels, info.frames) {
        let frames = frames.saturating_sub(info.delay + info.padding);
        return Ok((rate, channels, frames as f32 / rate as f32));
    }

    let audio = decode(path)?;
    Ok((audio.sample_rate, audio.channels, audio.duration()))
}

/// Sample-rate conversion, using rubato's FFT resampler so that downsampling
/// is low-pass filtered instead of aliased.
pub(crate) struct Resampler {
    inner: Option<rubato::Fft<f32>>,
    channels: usize,
    /// Interleaved input accumulated until the resampler's fixed chunk is full.
    pending: Vec<f32>,
    /// Frames of filter delay still to drop from the output; without this the
    /// stream starts a few milliseconds of silence late.
    skip_output: usize,
}

/// Chunk size for streaming resampling: 1024 frames keeps the filter sharp
/// while adding ~21 ms of latency at 48 kHz.
const RESAMPLE_CHUNK: usize = 1024;

impl Resampler {
    pub fn new(src_rate: u32, dst_rate: u32, channels: usize) -> Self {
        let inner = if src_rate == dst_rate || channels == 0 {
            None
        } else {
            rubato::Fft::<f32>::new(
                src_rate as usize,
                dst_rate as usize,
                RESAMPLE_CHUNK,
                channels,
                rubato::FixedSync::Input,
            )
            .ok()
        };
        let skip_output = inner
            .as_ref()
            .map(|inner| {
                use rubato::Resampler as _;
                inner.output_delay()
            })
            .unwrap_or(0);
        Self {
            inner,
            channels,
            pending: Vec::new(),
            skip_output,
        }
    }

    pub fn reset(&mut self) {
        if let Some(inner) = &mut self.inner {
            use rubato::Resampler as _;
            inner.reset();
            self.skip_output = inner.output_delay();
        }
        self.pending.clear();
    }

    /// Processes whatever is left in the buffer at end of stream. Without
    /// this the last few milliseconds of a track are silently dropped.
    pub fn flush(&mut self) -> Vec<f32> {
        use rubato::Resampler as _;
        let Some(inner) = &mut self.inner else {
            return std::mem::take(&mut self.pending);
        };
        let channels = self.channels;
        let frames = self.pending.len() / channels.max(1);
        if frames == 0 {
            return Vec::new();
        }
        let Ok(adapter) = rubato::audioadapter_buffers::direct::InterleavedSlice::new(
            &self.pending,
            channels,
            frames,
        ) else {
            self.pending.clear();
            return Vec::new();
        };
        let indexing = rubato::Indexing {
            input_offset: 0,
            output_offset: 0,
            partial_len: Some(frames),
            active_channels_mask: None,
        };
        let out = match inner.process(&adapter, Some(&indexing)) {
            Ok(resampled) => resampled.take_data(),
            Err(_) => Vec::new(),
        };
        self.pending.clear();
        out
    }

    /// Resamples as much of the input as can be resampled right now. The
    /// leftover stays buffered until the next call.
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        use rubato::Resampler as _;
        let Some(inner) = &mut self.inner else {
            return input.to_vec();
        };
        let channels = self.channels;
        self.pending.extend_from_slice(input);

        let mut out = Vec::new();
        loop {
            let need = inner.input_frames_next();
            if self.pending.len() < need * channels {
                break;
            }
            let chunk: Vec<f32> = self.pending.drain(..need * channels).collect();
            let Ok(adapter) =
                rubato::audioadapter_buffers::direct::InterleavedSlice::new(&chunk, channels, need)
            else {
                break;
            };
            match inner.process(&adapter, None) {
                Ok(resampled) => {
                    let mut data = resampled.take_data();
                    if self.skip_output > 0 {
                        let drop = (self.skip_output * channels).min(data.len());
                        data.drain(..drop);
                        self.skip_output -= drop / channels.max(1);
                    }
                    out.extend_from_slice(&data);
                }
                Err(_) => break,
            }
        }
        out
    }
}

/// Resamples a whole buffer. Uses the resampler's `process_all`, which trims
/// the filter's startup delay so the output lines up with the input.
pub(crate) fn resample_all(
    samples: &[f32],
    src_rate: u32,
    dst_rate: u32,
    channels: usize,
) -> Vec<f32> {
    use rubato::Resampler as _;

    if src_rate == dst_rate || channels == 0 {
        return samples.to_vec();
    }
    let frames = samples.len() / channels;
    if frames == 0 {
        return Vec::new();
    }

    let mut inner = match rubato::Fft::<f32>::new(
        src_rate as usize,
        dst_rate as usize,
        RESAMPLE_CHUNK.max(frames.min(8192)),
        channels,
        rubato::FixedSync::Input,
    ) {
        Ok(inner) => inner,
        Err(_) => return samples.to_vec(),
    };

    let Ok(adapter) =
        rubato::audioadapter_buffers::direct::InterleavedSlice::new(samples, channels, frames)
    else {
        return samples.to_vec();
    };

    match inner.process_all(&adapter, frames, None) {
        Ok(resampled) => resampled.take_data(),
        Err(_) => samples.to_vec(),
    }
}

/// A file opened for incremental decoding: one packet at a time, keeping only
/// the decoder state, never the whole file.
pub(crate) struct Stream {
    format: Box<dyn FormatReader>,
    decoder: Box<dyn symphonia::core::codecs::Decoder>,
    buffer: Option<SampleBuffer<f32>>,
    trim_frames: Option<u64>,
    emitted: u64,
    pub sample_rate: u32,
    pub channels: u16,
    pub delay: u32,
    /// Frame the last seek aimed for; the first packet after the seek tells us
    /// where the container actually landed, which is usually earlier.
    seek_target: Option<u64>,
}

impl Stream {
    pub fn open(path: &str) -> Result<Self, String> {
        let format = open(path)?;
        let (sample_rate, channels, n_frames, delay, padding, decoder) = {
            let track = format.default_track().ok_or("No audio track found")?;
            let decoder = symphonia::default::get_codecs()
                .make(&track.codec_params, &DecoderOptions::default())
                .map_err(|e| format!("Unsupported codec: {e}"))?;
            (
                track.codec_params.sample_rate,
                track.codec_params.channels.map(|c| c.count() as u16),
                track.codec_params.n_frames,
                track.codec_params.delay,
                track.codec_params.padding,
                decoder,
            )
        };

        let delay = priming_frames(path, delay).unwrap_or(0);
        let padding = padding.unwrap_or(0);
        let _ = padding;
        // How many frames of the file are real audio, when the header knows.
        let trim_frames = n_frames.map(|total| total.saturating_sub(delay as u64 + padding as u64));

        Ok(Self {
            format,
            decoder,
            buffer: None,
            trim_frames,
            emitted: 0,
            // AAC has no channel layout in the container; the first packet fills it in.
            sample_rate: sample_rate.ok_or("Unknown sample rate")?,
            channels: channels.unwrap_or(0),
            delay,
            seek_target: None,
        })
    }

    pub fn delay_frames(&self) -> u32 {
        self.delay
    }

    /// Decodes the next packet, applying the encoder delay (`skip`) and
    /// stopping before the trailing padding. `None` means end of file.
    pub fn next_chunk(&mut self, skip: &mut u32) -> Result<Option<Vec<f32>>, String> {
        loop {
            let packet = match self.format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(None)
                }
                Err(SymphoniaError::ResetRequired) => return Ok(None),
                Err(e) => return Err(format!("Read error: {e}")),
            };

            // Containers seek to a packet, not to a frame: skip the difference.
            if let Some(target) = self.seek_target.take() {
                let landed = packet.ts();
                *skip = skip.saturating_add(target.saturating_sub(landed) as u32);
            }

            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(SymphoniaError::IoError(e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(None)
                }
                Err(e) => return Err(format!("Decode error: {e}")),
            };

            if self.channels == 0 {
                self.channels = decoded.spec().channels.count() as u16;
                self.sample_rate = decoded.spec().rate;
            }

            let needs_new = self
                .buffer
                .as_ref()
                .is_none_or(|b| b.capacity() < decoded.capacity());
            if needs_new {
                self.buffer = Some(SampleBuffer::<f32>::new(
                    decoded.capacity() as u64,
                    *decoded.spec(),
                ));
            }
            let buffer = self.buffer.as_mut().expect("buffer was just created");
            buffer.copy_interleaved_ref(decoded);

            let channels = self.channels as usize;
            let mut chunk: &[f32] = buffer.samples();
            if *skip > 0 {
                let drop = (*skip as usize).saturating_mul(channels).min(chunk.len());
                chunk = &chunk[drop..];
                *skip -= (drop / channels.max(1)) as u32;
            }
            if chunk.is_empty() {
                continue;
            }

            let mut frames = (chunk.len() / channels.max(1)) as u64;
            if let Some(limit) = self.trim_frames {
                let room = limit.saturating_sub(self.emitted);
                frames = frames.min(room);
                if frames == 0 {
                    return Ok(None);
                }
            }
            self.emitted += frames;

            return Ok(Some(chunk[..frames as usize * channels.max(1)].to_vec()));
        }
    }

    /// Seeks. Containers seek to a packet boundary rather than a frame, so the
    /// caller is told how many frames to drop via `skip` on the next packet.
    pub fn seek(&mut self, seconds: f64) -> Result<(), String> {
        let rate = self.sample_rate.max(1) as f64;
        let requested = (seconds.max(0.0) * rate).round() as u64 + self.delay as u64;
        let limit = self
            .trim_frames
            .map(|frames| frames + self.delay as u64)
            .unwrap_or(u64::MAX);
        let target = requested.min(limit.saturating_sub(1));

        let track_id = self.format.default_track().map(|t| t.id).unwrap_or(0);
        self.format
            .seek(
                symphonia::core::formats::SeekMode::Accurate,
                SeekTo::TimeStamp {
                    ts: target,
                    track_id,
                },
            )
            .map_err(|e| format!("Seek failed: {e}"))?;

        self.decoder.reset();
        // `emitted` counts real audio frames since the start of the file, so a
        // seek must jump it forward instead of restarting the padding budget.
        self.emitted = target.saturating_sub(self.delay as u64);
        self.buffer = None;
        self.seek_target = Some(target);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_all_keeps_duration() {
        let input: Vec<f32> = (0..48_000)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();
        let out = resample_all(&input, 48_000, 44_100, 1);
        let expected = 44_100;
        assert!(
            (out.len() as i64 - expected).abs() <= 2,
            "resampled length {} should be ~{}",
            out.len(),
            expected
        );
    }

    #[test]
    fn resample_all_is_identity_at_same_rate() {
        let input = vec![0.1f32, 0.2, 0.3];
        assert_eq!(resample_all(&input, 44_100, 44_100, 1), input);
    }

    #[test]
    fn resample_all_suppresses_aliasing_on_downsample() {
        // 20 kHz is above the 12 kHz Nyquist of a 24 kHz output. A bare
        // linear resampler folds it to 4 kHz at full amplitude; a filtered one
        // must push it far down.
        let rate = 48_000;
        let input: Vec<f32> = (0..rate)
            .map(|i| (2.0 * std::f32::consts::PI * 20_000.0 * i as f32 / rate as f32).sin())
            .collect();
        let out = resample_all(&input, rate, 24_000, 1);
        let rms = |samples: &[f32]| {
            (samples.iter().map(|s| s * s).sum::<f32>() / samples.len().max(1) as f32).sqrt()
        };
        let before = rms(&input);
        let after = rms(&out);
        assert!(
            after < before * 0.1,
            "alias was not filtered: in {before:.3}, out {after:.3}"
        );
    }

    /// Minimal 16-bit PCM WAV whose header claims `frames` but which carries
    /// only `actual_frames` of audio.
    fn write_truncated_wav(path: &std::path::Path, frames: u32, actual_frames: u32, rate: u32) {
        let (channels, bits) = (1u16, 16u16);
        let block = channels * bits / 8;
        let mut buf = Vec::new();
        buf.extend(b"RIFF");
        buf.extend((36 + frames * block as u32).to_le_bytes());
        buf.extend(b"WAVEfmt ");
        buf.extend(16u32.to_le_bytes());
        buf.extend(1u16.to_le_bytes());
        buf.extend(channels.to_le_bytes());
        buf.extend(rate.to_le_bytes());
        buf.extend((rate * block as u32).to_le_bytes());
        buf.extend(block.to_le_bytes());
        buf.extend(bits.to_le_bytes());
        buf.extend(b"data");
        buf.extend((frames * block as u32).to_le_bytes());
        buf.extend(std::iter::repeat_n(
            0u8,
            actual_frames as usize * block as usize,
        ));
        std::fs::write(path, buf).unwrap();
    }

    #[test]
    fn seek_lands_on_the_requested_frame() {
        // 1 second of a ramp at 16 kHz: sample value encodes its frame number,
        // so an inaccurate seek is visible in the first sample.
        let rate = 16_000u32;
        let frames = rate;
        let path = std::env::temp_dir().join(format!("sound_rs_seek_{}.wav", std::process::id()));
        let mut buf = Vec::new();
        buf.extend(b"RIFF");
        buf.extend((36 + frames * 2).to_le_bytes());
        buf.extend(b"WAVEfmt ");
        buf.extend(16u32.to_le_bytes());
        buf.extend(1u16.to_le_bytes());
        buf.extend(1u16.to_le_bytes());
        buf.extend(rate.to_le_bytes());
        buf.extend((rate * 2).to_le_bytes());
        buf.extend(2u16.to_le_bytes());
        buf.extend(16u16.to_le_bytes());
        buf.extend(b"data");
        buf.extend((frames * 2).to_le_bytes());
        for i in 0..frames {
            buf.extend(((i as i16 - 16384).to_le_bytes()) as [u8; 2]);
        }
        std::fs::write(&path, &buf).unwrap();

        let mut stream = Stream::open(path.to_str().unwrap()).unwrap();
        let mut skip = 0u32;
        stream.seek(0.5).unwrap();
        let chunk = stream
            .next_chunk(&mut skip)
            .unwrap()
            .expect("audio after seek");
        std::fs::remove_file(&path).ok();

        let first = chunk[0] * 32768.0 + 16384.0;
        let expected = 0.5 * rate as f32;
        assert!(
            (first - expected).abs() < 32.0,
            "seek landed at frame {first}, expected ~{expected}"
        );
    }

    #[test]
    fn audio_info_reads_header_without_decoding() {
        let path = std::env::temp_dir().join(format!("sound_rs_info_{}.wav", std::process::id()));
        write_truncated_wav(&path, 44_100 * 2, 100, 44_100);
        let (rate, channels, duration) = info_or_decode(path.to_str().unwrap()).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!((rate, channels), (44_100, 1));
        assert!(
            (duration - 2.0).abs() < 0.01,
            "duration came from the header, not the decoded tail: {duration}"
        );
    }

    #[test]
    fn prealloc_is_capped_and_overflow_safe() {
        // 2 hours of 44.1k stereo must not reserve the whole file up front.
        let huge = prealloc_capacity(Some(44_100 * 2 * 7200), Some(2), Some(44_100));
        assert_eq!(huge, 44_100 * 2 * 60);
        // A short file gets an exact reservation.
        assert_eq!(
            prealloc_capacity(Some(44_100), Some(2), Some(44_100)),
            44_100 * 2
        );
        // Unknown metadata reserves nothing rather than guessing.
        assert_eq!(prealloc_capacity(None, Some(2), Some(44_100)), 0);
        // u64::MAX frames must not wrap on 64-bit or panic.
        assert_eq!(
            prealloc_capacity(Some(u64::MAX), Some(2), Some(44_100)),
            44_100 * 2 * 60
        );
    }

    #[test]
    fn append_skipping_drops_leading_frames_across_chunks() {
        let mut out = Vec::new();
        let mut skip = append_skipping(&mut out, &[1.0, 2.0], 3);
        assert_eq!(skip, 1, "two of three samples consumed");
        assert!(out.is_empty());

        skip = append_skipping(&mut out, &[3.0, 4.0, 5.0, 6.0], skip);
        assert_eq!(skip, 0);
        assert_eq!(out, vec![4.0, 5.0, 6.0], "only the delay is dropped");

        // A delay longer than the file leaves nothing rather than wrapping.
        assert_eq!(append_skipping(&mut out, &[7.0], 5), 4);
    }

    #[test]
    fn mp4_edit_list_priming_is_read() {
        fn boxed(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend(((8 + body.len()) as u32).to_be_bytes());
            out.extend(kind);
            out.extend(body);
            out
        }

        let mut hdlr = vec![0u8; 4]; // version + flags
        hdlr.extend([0u8; 4]); // predefined
        hdlr.extend(b"soun");
        hdlr.extend([0u8; 12]);

        let mut elst = vec![0u8; 4]; // version 0 + flags
        elst.extend(1u32.to_be_bytes()); // one entry
        elst.extend(2_646_000u32.to_be_bytes()); // segment_duration
        elst.extend(1024u32.to_be_bytes()); // media_time = priming
        elst.extend(0x0001_0000u32.to_be_bytes()); // media_rate

        let mut trak = boxed(b"mdia", &boxed(b"hdlr", &hdlr));
        trak.extend(boxed(b"edts", &boxed(b"elst", &elst)));
        let mut file = boxed(b"ftyp", b"isomisom00");
        file.extend(boxed(b"moov", &boxed(b"trak", &trak)));

        let path = std::env::temp_dir().join(format!("sound_rs_{}.m4a", std::process::id()));
        std::fs::write(&path, &file).unwrap();
        let priming = mp4_priming(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();

        assert_eq!(priming, Some(1024));
    }

    #[test]
    fn mp4_priming_is_none_for_junk() {
        let path = std::env::temp_dir().join(format!("sound_rs_junk_{}.m4a", std::process::id()));
        std::fs::write(&path, b"not an mp4 file at all").unwrap();
        let priming = mp4_priming(path.to_str().unwrap());
        std::fs::remove_file(&path).ok();
        assert_eq!(priming, None);
        assert!(!is_mp4("/nonexistent"), "missing files are not mp4");
    }

    #[test]
    fn convert_mono_to_stereo_duplicates() {
        assert_eq!(
            convert_channels(&[0.5, -0.5], 1, 2),
            vec![0.5, 0.5, -0.5, -0.5]
        );
    }

    #[test]
    fn convert_stereo_to_mono_averages() {
        assert_eq!(
            convert_channels(&[1.0, 0.0, -1.0, -1.0], 2, 1),
            vec![0.5, -1.0]
        );
    }

    #[test]
    fn convert_keeps_frame_count() {
        for (src, dst) in [(1u16, 2u16), (2, 1), (6, 2), (2, 6)] {
            let input: Vec<f32> = (0..src as usize * 10).map(|i| i as f32).collect();
            let out = convert_channels(&input, src, dst);
            assert_eq!(out.len(), dst as usize * 10, "{src}->{dst}");
        }
    }
}

#[cfg(test)]
mod fuzz_tests {
    use super::*;

    /// Deterministic xorshift so a failing case is reproducible from the seed.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    fn mutate(input: &[u8], seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        let mut data = input.to_vec();
        if data.is_empty() {
            return data;
        }
        let flips = 1 + (next(&mut state) % 8) as usize;
        for _ in 0..flips {
            let index = (next(&mut state) as usize) % data.len();
            data[index] ^= (next(&mut state) & 0xff) as u8;
        }
        match next(&mut state) % 4 {
            0 => data.truncate(data.len() / 2),
            1 => data.truncate((next(&mut state) as usize) % data.len() + 1),
            2 => {
                let extra = (next(&mut state) % 512) as usize;
                data.extend(std::iter::repeat_n(0u8, extra));
            }
            _ => {}
        }
        data
    }

    /// A corrupt file must produce an error value, never a panic. Panics in a
    /// test are failures, which is exactly the signal we want.
    #[test]
    fn mutated_fixtures_do_not_panic() {
        let fixtures = [
            "tests/fixtures/tone.wav",
            "tests/fixtures/tone.flac",
            "tests/fixtures/tone.mp3",
            "tests/fixtures/tone.m4a",
            "tests/fixtures/six.wav",
        ];
        let mut errors = 0;
        let mut ok = 0;
        for path in fixtures {
            let Ok(bytes) = std::fs::read(path) else {
                continue; // fixtures missing outside a full checkout
            };
            for seed in 1..200u64 {
                match decode_bytes(&mutate(&bytes, seed)) {
                    Ok(audio) => {
                        ok += 1;
                        assert!(audio.channels > 0, "{path} seed {seed}: zero channels");
                        assert!(
                            audio.samples.iter().all(|s| s.is_finite()),
                            "{path} seed {seed}: non-finite sample"
                        );
                    }
                    Err(message) => {
                        errors += 1;
                        assert!(!message.is_empty());
                    }
                }
            }
        }
        let total = ok + errors;
        assert!(total > 0, "no fixture was found to fuzz");
        assert!(
            errors > 0,
            "mutation never produced an error, fuzz is not biting"
        );
    }

    #[test]
    fn garbage_does_not_panic() {
        for seed in 0..64u64 {
            let mut state = seed | 1;
            let length = (next(&mut state) % 2048) as usize;
            let data: Vec<u8> = (0..length)
                .map(|_| (next(&mut state) & 0xff) as u8)
                .collect();
            assert!(
                decode_bytes(&data).is_err(),
                "random bytes decoded successfully"
            );
        }
        for data in [
            &b""[..],
            b"RIFF",
            b"fLaC",
            b"ID3",
            b"\x00".repeat(4096).as_slice(),
        ] {
            assert!(decode_bytes(data).is_err());
        }
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    fn rate_strategy() -> impl Strategy<Value = u32> {
        prop::sample::select(vec![
            8_000u32, 16_000, 22_050, 32_000, 44_100, 48_000, 96_000,
        ])
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn resample_length_is_proportional(
            frames in 1usize..5_000,
            channels in 1usize..=2,
            src in rate_strategy(),
            dst in rate_strategy(),
        ) {
            let input = vec![0.0f32; frames * channels];
            let out = resample_all(&input, src, dst, channels);
            let expected = frames as f64 * dst as f64 / src as f64;
            let actual = (out.len() / channels) as f64;
            prop_assert!(
                (actual - expected).abs() <= 1.0,
                "{} frames at {} Hz -> {} frames at {} Hz, expected ~{}",
                frames, src, actual, dst, expected
            );
        }

        #[test]
        fn resample_keeps_dc_and_stays_finite(
            frames in 1usize..2_000,
            channels in 1usize..=2,
            src in rate_strategy(),
            dst in rate_strategy(),
            level in -1.0f32..1.0,
        ) {
            let input = vec![level; frames * channels];
            let out = resample_all(&input, src, dst, channels);
            prop_assert!(out.iter().all(|s| s.is_finite()));

            // A band-limited resampler rings at a hard edge (an abrupt DC step),
            // so the edge samples are allowed to overshoot; the bound only
            // catches runaway gain. Steady state must reproduce the level.
            let peak = out.iter().fold(0.0f32, |acc, s| acc.max(s.abs()));
            prop_assert!(
                peak <= level.abs() * 1.6 + 0.05,
                "runaway gain {} for level {}",
                peak,
                level
            );

            let edge = (64 * channels).min(out.len() / 2);
            let steady = &out[edge..out.len() - edge];
            if steady.len() > 8 {
                let mean = steady.iter().sum::<f32>() / steady.len() as f32;
                prop_assert!(
                    (mean - level).abs() < 0.01,
                    "steady-state DC shifted: {} vs {}",
                    mean,
                    level
                );
            }
        }

        #[test]
        fn channel_conversion_keeps_frame_count(
            frames in 1usize..2_000,
            src in 1u16..=6,
            dst in 1u16..=6,
        ) {
            let input = vec![0.25f32; frames * src as usize];
            let out = convert_channels(&input, src, dst);
            prop_assert_eq!(out.len(), frames * dst as usize);
            prop_assert!(out.iter().all(|s| s.is_finite()));
        }

        #[test]
        fn mono_fold_cancels_opposite_channels(gain in 0.0f32..1.0) {
            let input = vec![gain, -gain];
            let out = convert_channels(&input, 2, 1);
            prop_assert_eq!(out.len(), 1);
            prop_assert!(out[0].abs() < 1e-7, "mono fold of +x/-x was {}", out[0]);
        }
    }
}

#[cfg(test)]
mod seek_midstream_test {
    use super::*;

    #[test]
    fn seek_after_eof_lands_on_target() {
        let path = "tests/fixtures/tone.wav";
        let Ok(mut stream) = Stream::open(path) else {
            return;
        };
        let mut skip = 0u32;
        while stream.next_chunk(&mut skip).unwrap().is_some() {}
        stream.seek(0.1).unwrap();
        let mut skip = 0u32;
        let mut frames = 0usize;
        while let Some(chunk) = stream.next_chunk(&mut skip).unwrap() {
            frames += chunk.len() / stream.channels as usize;
        }
        assert!(
            (frames as i64 - 6615).abs() <= 2,
            "after eof+seek got {frames} frames"
        );
    }

    #[test]
    fn seek_after_reading_frames_lands_on_target() {
        let path = "tests/fixtures/tone.wav";
        let Ok(mut stream) = Stream::open(path) else {
            return;
        };
        let mut skip = 0u32;
        let _ = stream.next_chunk(&mut skip).unwrap();
        stream.seek(0.1).unwrap();
        let mut skip = 0u32;
        let mut frames = 0usize;
        while let Some(chunk) = stream.next_chunk(&mut skip).unwrap() {
            frames += chunk.len() / stream.channels as usize;
        }
        // 0.25 s file, seek to 0.1 s -> ~6615 frames left.
        assert!(
            (frames as i64 - 6615).abs() <= 2,
            "after seek got {frames} frames"
        );
    }
}
