//! Fast cross-platform audio playback for Python.
//!
//! Decoding is done with symphonia, output with cpal. Every decoded format is
//! converted to interleaved f32 in one place, then handed to the device with
//! whatever sample format that device actually wants.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample};
use pyo3::prelude::*;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// Decoded audio, always interleaved f32.
struct Audio {
    sample_rate: u32,
    channels: u16,
    samples: Vec<f32>,
}

impl Audio {
    fn duration(&self) -> f32 {
        self.samples.len() as f32 / self.sample_rate as f32 / self.channels as f32
    }
}

/// Opens the container. This is also the format check: whatever symphonia can
/// probe is what we support, so there is no extension list to keep in sync.
fn open(path: &str) -> Result<Box<dyn FormatReader>, String> {
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
struct TrackInfo {
    sample_rate: Option<u32>,
    channels: Option<u16>,
    frames: Option<u64>,
    delay: u64,
    padding: u64,
}

fn probe_info(path: &str) -> Result<TrackInfo, String> {
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
        let size = u32::from_be_bytes([data[offset], data[offset + 1], data[offset + 2], data[offset + 3]]) as usize;
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
    mp4_children(data).into_iter().find(|(k, _)| *k == kind).map(|(_, body)| body)
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
        let Some(mdia) = mp4_child(trak, b"mdia") else { continue };
        let Some(hdlr) = mp4_child(mdia, b"hdlr") else { continue };
        if hdlr.get(8..12) != Some(b"soun") {
            continue;
        }
        let Some(elst) = mp4_child(trak, b"edts").and_then(|edts| mp4_child(edts, b"elst")) else {
            return None;
        };
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

fn decode(path: &str) -> Result<Audio, String> {
    let mut format = open(path)?;

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
    delay = priming_frames(path, delay);

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
            .map_or(true, |b| b.capacity() < decoded.capacity());
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
        return Err(format!("No audio data decoded from '{path}'"));
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
        return Err(format!("'{path}' contains no audio after trimming encoder padding"));
    }

    Ok(Audio {
        sample_rate: sample_rate.ok_or("Unknown sample rate")?,
        channels,
        samples,
    })
}

/// Linear-interpolating resampler. Works on interleaved frames so the channel
/// layout survives; a naive per-sample index would shuffle channels together.
fn resample(samples: &[f32], src_rate: u32, dst_rate: u32, channels: u16) -> Vec<f32> {
    let ch = channels as usize;
    if src_rate == dst_rate || ch == 0 {
        return samples.to_vec();
    }

    let frames = samples.len() / ch;
    if frames == 0 {
        return Vec::new();
    }

    let ratio = src_rate as f64 / dst_rate as f64;
    let target_frames = (frames as f64 / ratio) as usize;
    let mut out = Vec::with_capacity(target_frames * ch);

    for i in 0..target_frames {
        let src_pos = i as f64 * ratio;
        let i0 = src_pos as usize;
        let i1 = (i0 + 1).min(frames - 1);
        let frac = (src_pos - i0 as f64) as f32;
        for c in 0..ch {
            let a = samples[i0 * ch + c];
            let b = samples[i1 * ch + c];
            out.push(a + (b - a) * frac);
        }
    }

    out
}

/// Converts an interleaved buffer from `src` to `dst` channels. Down-mixing
/// stereo averages both channels; everything else repeats or truncates
/// channels in order, which is what a simple mixer does.
fn convert_channels(samples: &[f32], src: u16, dst: u16) -> Vec<f32> {
    let (src, dst) = (src as usize, dst as usize);
    if src == dst || src == 0 || dst == 0 {
        return samples.to_vec();
    }

    if src == 2 && dst == 1 {
        return samples.chunks_exact(2).map(|f| (f[0] + f[1]) * 0.5).collect();
    }

    samples
        .chunks_exact(src)
        .flat_map(|frame| (0..dst).map(move |c| frame[c % src]))
        .collect()
}

/// Copies the next block into the output buffer, zero-padding the tail.
/// Returns how many real samples were written. One atomic update per block,
/// not per sample.
fn fill_block<T>(output: &mut [T], data: &[f32], position: usize) -> usize
where
    T: SizedSample + FromSample<f32>,
{
    let n = output.len().min(data.len().saturating_sub(position));
    let available = data.get(position..).unwrap_or(&[]);
    for (slot, sample) in output[..n].iter_mut().zip(available) {
        *slot = T::from_sample(*sample);
    }
    for slot in output[n..].iter_mut() {
        *slot = T::from_sample(0.0f32);
    }
    n
}

/// How long the last real sample still needs before it reaches the DAC: the
/// device's own latency plus the duration of the silence padding after it.
/// `None` when the backend reports no playback timestamp, so the caller knows
/// to fall back to a conservative wait instead of trusting a zero latency.
fn drain_duration(
    latency: Option<Duration>,
    padding: usize,
    channels: u16,
    rate: u32,
) -> Option<Duration> {
    let latency = latency?;
    let padding = Duration::from_secs_f64(
        padding as f64 / channels.max(1) as f64 / rate.max(1) as f64,
    );
    Some(latency + padding)
}

/// True when this callback delivered the last sample of the track, whether or
/// not the block was completely filled.
fn block_completes(position: usize, written: usize, total: usize, output_len: usize) -> bool {
    written < output_len || position + written >= total
}

/// Everything the real-time callback needs, shared with the play loop.
struct PlaybackState {
    data: Arc<Vec<f32>>,
    position: Arc<AtomicUsize>,
    drain_us: Arc<AtomicU64>,
    error: Arc<Mutex<Option<String>>>,
    channels: u16,
    rate: u32,
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    state: PlaybackState,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let err_slot = state.error.clone();
    device
        .build_output_stream(
            config,
            move |output: &mut [T], info: &cpal::OutputCallbackInfo| {
                let position = state.position.load(Ordering::Relaxed);
                let written = fill_block(output, &state.data, position);
                state.position.store(position + written, Ordering::Relaxed);

                let total = state.data.len();
                if block_completes(position, written, total, output.len())
                    && state.drain_us.load(Ordering::Relaxed) == 0
                {
                    // This callback delivered the last real sample; record when
                    // it will actually be heard so play() can wait exactly.
                    let stamp = info.timestamp();
                    let latency = stamp.playback.duration_since(&stamp.callback);
                    if let Some(drain) =
                        drain_duration(latency, output.len() - written, state.channels, state.rate)
                    {
                        let _ = state.drain_us.compare_exchange(
                            0,
                            (drain.as_micros() as u64).max(1),
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        );
                    }
                }
            },
            move |err| {
                if let Ok(mut slot) = err_slot.lock() {
                    slot.get_or_insert_with(|| err.to_string());
                }
            },
            None,
        )
        .map_err(|e| format!("Cannot open output stream: {e}"))
}


/// The single playback path: resample to the device rate, fold to the device
/// channel count, open a stream in the device's own sample format, block until
/// the buffer is drained. Takes ownership so an already-correct buffer is
/// played without any extra copy.
fn play(samples: Vec<f32>, src_rate: u32, src_channels: u16) -> Result<(), String> {
    let device = cpal::default_host()
        .default_output_device()
        .ok_or("No output device available")?;
    let supported = device
        .default_output_config()
        .map_err(|e| format!("Cannot query output device: {e}"))?;

    let out_rate = supported.sample_rate().0;
    let out_channels = supported.channels();

    let ready = if src_rate == out_rate && src_channels == out_channels {
        samples
    } else {
        convert_channels(
            &resample(&samples, src_rate, out_rate, src_channels),
            src_channels,
            out_channels,
        )
    };
    let config = cpal::StreamConfig {
        channels: out_channels,
        sample_rate: supported.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let data = Arc::new(ready);
    let position = Arc::new(AtomicUsize::new(0));
    let drain_us = Arc::new(AtomicU64::new(0));
    let error = Arc::new(Mutex::new(None));
    let state = PlaybackState {
        data: data.clone(),
        position: position.clone(),
        drain_us: drain_us.clone(),
        error: error.clone(),
        channels: out_channels,
        rate: out_rate,
    };

    let stream = match supported.sample_format() {
        SampleFormat::F32 => build_stream::<f32>(&device, &config, state),
        SampleFormat::F64 => build_stream::<f64>(&device, &config, state),
        SampleFormat::I16 => build_stream::<i16>(&device, &config, state),
        SampleFormat::I32 => build_stream::<i32>(&device, &config, state),
        SampleFormat::U8 => build_stream::<u8>(&device, &config, state),
        SampleFormat::U16 => build_stream::<u16>(&device, &config, state),
        other => return Err(format!("Unsupported output sample format: {other:?}")),
    }?;

    stream.play().map_err(|e| format!("Playback failed: {e}"))?;

    // Wait until the callback has consumed the buffer, then for as long as the
    // last sample still needs to reach the device. A callback that stops
    // advancing means the device went away without an error: cpal only reports
    // disconnects for non-default devices, so this watchdog is the only exit.
    const STALL_TIMEOUT: Duration = Duration::from_secs(2);
    let mut last_seen = position.load(Ordering::Relaxed);
    let mut last_change = Instant::now();
    loop {
        let current = position.load(Ordering::Relaxed);
        if current >= data.len() {
            break;
        }
        if error.lock().map(|e| e.is_some()).unwrap_or(false) {
            break;
        }
        if current != last_seen {
            last_seen = current;
            last_change = Instant::now();
        } else if last_change.elapsed() > STALL_TIMEOUT {
            return Err(
                "Playback stalled for 2s: the output device stopped responding (disconnected?)"
                    .to_string(),
            );
        }
        std::thread::sleep(Duration::from_millis(5));
    }

    // The drain timestamp is published right after the position update, so a
    // short spin is enough. It stays 0 on backends that report no timestamp.
    let mut drain = drain_us.load(Ordering::Relaxed);
    for _ in 0..50 {
        if drain > 0 {
            break;
        }
        std::thread::sleep(Duration::from_millis(1));
        drain = drain_us.load(Ordering::Relaxed);
    }
    let wait = if drain > 0 {
        // Margin covers cpal's own uncertainty about when `playback` starts
        // relative to the callback buffer.
        Duration::from_micros(drain) + Duration::from_millis(30)
    } else {
        Duration::from_millis(150)
    };
    std::thread::sleep(wait);

    match error.lock().map(|mut e| e.take()).unwrap_or(None) {
        Some(e) => Err(format!("Audio stream error: {e}")),
        None => Ok(()),
    }
}

fn play_file(path: &str) -> Result<(), String> {
    let audio = decode(path)?;
    play(audio.samples, audio.sample_rate, audio.channels)
}

fn to_pyerr(message: String) -> PyErr {
    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(message)
}

/// Play audio file (WAV, MP3, FLAC, OGG, M4A, AAC and anything else symphonia probes).
///
/// Blocks until playback finishes, but releases the GIL, so other Python
/// threads keep running.
#[pyfunction]
fn play_audio(py: Python<'_>, file_path: &str) -> PyResult<()> {
    py.allow_threads(|| play_file(file_path)).map_err(to_pyerr)
}

/// Check if the file can be decoded, without decoding it.
#[pyfunction]
fn is_supported_format(file_path: &str) -> bool {
    open(file_path).is_ok()
}

/// Get `(sample_rate, channels, duration_seconds)` from the container header.
///
/// Only decodes when the header does not carry a frame count (some VBR MP3s).
#[pyfunction]
fn get_audio_info(py: Python<'_>, file_path: &str) -> PyResult<(u32, u16, f32)> {
    py.allow_threads(|| info_or_decode(file_path)).map_err(to_pyerr)
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
        (decoder, track.codec_params.channels.map(|c| c.count() as u16))
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

fn info_or_decode(path: &str) -> Result<(u32, u16, f32), String> {
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

/// Decode a file to `(array, sample_rate)`, shape `(frames, channels)`, float32.
#[pyfunction]
fn load<'py>(py: Python<'py>, file_path: &str) -> PyResult<(Bound<'py, PyAny>, u32)> {
    #[cfg(not(feature = "numpy"))]
    {
        let _ = (py, file_path);
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "NumPy support not enabled. Build with the 'numpy' feature.",
        ));
    }

    #[cfg(feature = "numpy")]
    {
        let audio = py.allow_threads(|| decode(file_path)).map_err(to_pyerr)?;
        let sample_rate = audio.sample_rate;
        use numpy::{PyArray1, PyArrayMethods};

        let channels = audio.channels as usize;
        let frames = audio.samples.len() / channels;
        let flat = PyArray1::from_vec(py, audio.samples);
        let shaped = flat
            .reshape((frames, channels))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok((shaped.into_any(), sample_rate))
    }
}

/// Play a NumPy array of f32: 1D mono `(samples,)` or 2D `(channels, samples)`.
#[pyfunction]
fn play_array(py: Python<'_>, array: Bound<'_, PyAny>, sample_rate: u32) -> PyResult<()> {
    #[cfg(feature = "numpy")]
    {
        use numpy::{PyArray1, PyArray2, PyArrayMethods};

        if let Ok(array) = array.extract::<Bound<'_, PyArray1<f32>>>() {
            let samples = unsafe { array.as_slice()? };
            return py
                .allow_threads(|| play(samples.to_vec(), sample_rate, 1))
                .map_err(to_pyerr);
        }

        if let Ok(array) = array.extract::<Bound<'_, PyArray2<f32>>>() {
            let view = unsafe { array.as_array() };
            let (channels, frames) = view.dim();
            let mut interleaved = Vec::with_capacity(channels * frames);
            for frame in 0..frames {
                for channel in 0..channels {
                    interleaved.push(view[[channel, frame]]);
                }
            }
            return py
                .allow_threads(|| play(interleaved, sample_rate, channels as u16))
                .map_err(to_pyerr);
        }

        return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Input must be a 1D or 2D numpy array of float32",
        ));
    }

    #[cfg(not(feature = "numpy"))]
    {
        let _ = (py, array, sample_rate);
        Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "NumPy support not enabled. Build with the 'numpy' feature.",
        ))
    }
}

/// Generate a sine tone and play it.
#[pyfunction]
fn play_sine_wave(py: Python<'_>, frequency: f32, duration: f32, sample_rate: u32) -> PyResult<()> {
    let frames = (duration.max(0.0) * sample_rate as f32) as usize;
    let step = frequency * 2.0 * std::f32::consts::PI / sample_rate as f32;
    let samples: Vec<f32> = (0..frames).map(|i| (i as f32 * step).sin() * 0.5).collect();
    let rate = sample_rate.max(1);
    py.allow_threads(move || play(samples, rate, 1))
        .map_err(to_pyerr)
}

#[pymodule]
fn sound_rs(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(play_audio, &m)?)?;
    m.add_function(wrap_pyfunction!(is_supported_format, &m)?)?;
    m.add_function(wrap_pyfunction!(get_audio_info, &m)?)?;
    m.add_function(wrap_pyfunction!(load, &m)?)?;
    m.add_function(wrap_pyfunction!(play_array, &m)?)?;
    m.add_function(wrap_pyfunction!(play_sine_wave, &m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal 16-bit PCM WAV whose header claims `frames` but which carries
    /// only `actual_frames` of audio.
    fn write_truncated_wav(path: &std::path::Path, frames: u32, actual_frames: u32, rate: u32) {
        let (channels, bits) = (1u16, 16u16);
        let block = channels * bits / 8;
        let mut buf = Vec::new();
        buf.extend(b"RIFF");
        buf.extend(((36 + frames * block as u32) as u32).to_le_bytes());
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
        buf.extend(std::iter::repeat_n(0u8, actual_frames as usize * block as usize));
        std::fs::write(path, buf).unwrap();
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
        assert_eq!(prealloc_capacity(Some(44_100), Some(2), Some(44_100)), 44_100 * 2);
        // Unknown metadata reserves nothing rather than guessing.
        assert_eq!(prealloc_capacity(None, Some(2), Some(44_100)), 0);
        // u64::MAX frames must not wrap on 64-bit or panic.
        assert_eq!(prealloc_capacity(Some(u64::MAX), Some(2), Some(44_100)), 44_100 * 2 * 60);
    }

    #[test]
    fn fill_block_copies_and_pads_tail() {
        let data: Vec<f32> = vec![0.5, -0.5, 1.0];
        let mut out = [9.0f32; 8];
        let written = fill_block(&mut out, &data, 0);
        assert_eq!(written, 3);
        assert_eq!(&out[..3], &[0.5, -0.5, 1.0]);
        assert!(out[3..].iter().all(|&s| s == 0.0), "tail must be silence, not stale data");
    }

    #[test]
    fn fill_block_resumes_mid_buffer() {
        let data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut out = [0.0f32; 4];
        let written = fill_block(&mut out, &data, 8);
        assert_eq!(written, 2);
        assert_eq!(&out[..2], &[8.0, 9.0]);
    }

    #[test]
    fn fill_block_converts_to_device_format() {
        let data = vec![1.0f32, -1.0f32];
        let mut out = [0i16; 2];
        fill_block(&mut out, &data, 0);
        assert!(out[0] > 32_000 && out[1] < -32_000, "full scale must map to full scale: {out:?}");
    }

    #[test]
    fn drain_covers_device_latency_and_padding() {
        // 882 samples of padding at 2ch/44.1k = 10 ms, plus 20 ms device latency.
        let drain = drain_duration(Some(Duration::from_millis(20)), 882, 2, 44_100).unwrap();
        assert!((drain.as_secs_f64() - 0.030).abs() < 0.001, "{drain:?}");
    }

    #[test]
    fn drain_is_unknown_without_a_device_timestamp() {
        // A backend with no timestamp must not be read as "zero latency".
        assert_eq!(drain_duration(None, 882, 2, 44_100), None);
    }

    #[test]
    fn drain_is_published_when_the_last_block_is_full() {
        // 512-sample blocks, 1024 samples total: the second block fills exactly,
        // so the completion must be detected from the position, not the padding.
        assert!(block_completes(512, 512, 1024, 512));
        assert!(!block_completes(0, 512, 4096, 512));
        assert!(block_completes(2048, 100, 4096, 512));
    }

    #[test]
    fn fill_block_ignores_a_position_past_the_end() {
        let data = vec![1.0f32, 2.0];
        let mut out = [7.0f32; 3];
        assert_eq!(fill_block(&mut out, &data, 99), 0);
        assert_eq!(out, [0.0, 0.0, 0.0]);
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
    fn resample_keeps_stereo_interleaved() {
        // 400 frames of (L=1.0, R=-1.0) at 48k -> 200 frames at 24k.
        let input: Vec<f32> = (0..800)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let out = resample(&input, 48000, 24000, 2);
        assert_eq!(out.len(), 200 * 2, "frame count must scale with the rate");
        assert!(out.iter().step_by(2).all(|&s| (s - 1.0).abs() < 1e-6));
        assert!(out.iter().skip(1).step_by(2).all(|&s| (s + 1.0).abs() < 1e-6));
    }

    #[test]
    fn resample_is_identity_at_same_rate() {
        let input = vec![0.1f32, 0.2, 0.3, 0.4];
        assert_eq!(resample(&input, 44100, 44100, 2), input);
    }

    #[test]
    fn resample_upsamples_full_duration() {
        let input: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let out = resample(&input, 22050, 44100, 1);
        assert_eq!(out.len(), 200);
    }

    #[test]
    fn convert_mono_to_stereo_duplicates() {
        assert_eq!(convert_channels(&[0.5, -0.5], 1, 2), vec![0.5, 0.5, -0.5, -0.5]);
    }

    #[test]
    fn convert_stereo_to_mono_averages() {
        assert_eq!(convert_channels(&[1.0, 0.0, -1.0, -1.0], 2, 1), vec![0.5, -1.0]);
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


