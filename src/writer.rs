//! Writing audio to disk.
//!
//! WAV only, on purpose: the container is a header plus the samples, so it
//! needs no encoder crate. FLAC and the rest stay out until something asks.

/// Sample format written into the WAV `fmt ` chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Subtype {
    Pcm16,
    Pcm24,
    Float32,
}

impl Subtype {
    pub fn parse(name: Option<&str>) -> Result<Self, String> {
        match name.unwrap_or("pcm16") {
            "pcm16" | "int16" => Ok(Self::Pcm16),
            "pcm24" | "int24" => Ok(Self::Pcm24),
            "float32" | "f32" => Ok(Self::Float32),
            other => Err(format!(
                "Unknown subtype '{other}'. Use pcm16, pcm24 or float32"
            )),
        }
    }

    fn bits(self) -> u16 {
        match self {
            Self::Pcm16 => 16,
            Self::Pcm24 => 24,
            Self::Float32 => 32,
        }
    }

    fn format_tag(self) -> u16 {
        // 3 = IEEE float, 1 = integer PCM.
        match self {
            Self::Float32 => 3,
            _ => 1,
        }
    }
}

fn encode(sample: f32, subtype: Subtype, out: &mut Vec<u8>) {
    let clamped = sample.clamp(-1.0, 1.0);
    match subtype {
        Subtype::Pcm16 => {
            let value = (clamped * i16::MAX as f32).round() as i16;
            out.extend_from_slice(&value.to_le_bytes());
        }
        Subtype::Pcm24 => {
            let value = (clamped * 8_388_607.0).round() as i32;
            out.extend_from_slice(&value.to_le_bytes()[..3]);
        }
        Subtype::Float32 => out.extend_from_slice(&clamped.to_le_bytes()),
    }
}

#[derive(Debug)]
pub(crate) enum WriteError {
    Usage(String),
    Io(String),
}

impl WriteError {
    pub fn code(&self) -> &'static str {
        match self {
            Self::Usage(_) => "usage",
            Self::Io(_) => "io",
        }
    }
}

impl std::fmt::Display for WriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Usage(message) | Self::Io(message) => f.write_str(message),
        }
    }
}

struct WavLayout {
    block_align: u16,
    byte_rate: u32,
    data_len: u32,
    riff_len: u32,
    padding: u32,
}

// Keep header arithmetic independent of sample storage so size limits can be
// checked before allocating, including files near RIFF's 4 GiB boundary.
fn wav_layout(
    sample_count: usize,
    channels: u16,
    rate: u32,
    subtype: Subtype,
) -> Result<WavLayout, String> {
    if rate == 0 {
        return Err("sample_rate must be positive".to_string());
    }
    if channels == 0 {
        return Err("channels must be positive".to_string());
    }
    if !sample_count.is_multiple_of(channels as usize) {
        return Err("Audio must contain a whole number of interleaved frames".to_string());
    }
    let sample_bytes = subtype.bits() / 8;
    let block_align = channels
        .checked_mul(sample_bytes)
        .ok_or("Channel layout exceeds WAV's 16-bit block alignment limit")?;
    let byte_rate = rate
        .checked_mul(block_align as u32)
        .ok_or("Sample rate and channel layout exceed WAV's 32-bit byte rate limit")?;
    let data_len = sample_count
        .checked_mul(sample_bytes as usize)
        .and_then(|bytes| u32::try_from(bytes).ok())
        .ok_or("Audio exceeds WAV's 32-bit RIFF size limit")?;
    let padding = data_len % 2;
    let riff_len = 36u32
        .checked_add(data_len)
        .and_then(|size| size.checked_add(padding))
        .ok_or("Audio exceeds WAV's 32-bit RIFF size limit")?;
    Ok(WavLayout {
        block_align,
        byte_rate,
        data_len,
        riff_len,
        padding,
    })
}

/// Builds a complete WAV file in memory.
pub(crate) fn wav_bytes(
    samples: &[f32],
    channels: u16,
    rate: u32,
    subtype: Subtype,
) -> Result<Vec<u8>, String> {
    let layout = wav_layout(samples.len(), channels, rate, subtype)?;
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("Audio samples must be finite (no NaN or infinity)".to_string());
    }

    let capacity = usize::try_from(u64::from(layout.riff_len) + 8)
        .map_err(|_| "WAV file is too large for this platform")?;
    let mut out = Vec::new();
    out.try_reserve_exact(capacity)
        .map_err(|e| format!("Cannot allocate WAV buffer: {e}"))?;
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&layout.riff_len.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&subtype.format_tag().to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&layout.byte_rate.to_le_bytes());
    out.extend_from_slice(&layout.block_align.to_le_bytes());
    out.extend_from_slice(&subtype.bits().to_le_bytes());

    out.extend_from_slice(b"data");
    out.extend_from_slice(&layout.data_len.to_le_bytes());
    for sample in samples {
        encode(*sample, subtype, &mut out);
    }
    // RIFF chunks are word-aligned.
    if layout.padding != 0 {
        out.push(0);
    }
    Ok(out)
}

pub(crate) fn write(
    path: &str,
    samples: &[f32],
    channels: u16,
    rate: u32,
    subtype: Subtype,
) -> Result<(), WriteError> {
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if extension != "wav" {
        return Err(WriteError::Usage(format!(
            "Only .wav can be written so far (got '.{extension}'); save as WAV or convert afterwards"
        )));
    }
    let bytes = wav_bytes(samples, channels, rate, subtype).map_err(WriteError::Usage)?;
    std::fs::write(path, bytes).map_err(|e| WriteError::Io(format!("Cannot write '{path}': {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(bytes: &[u8]) -> crate::audio::Audio {
        crate::audio::decode_bytes(bytes).expect("written wav must decode")
    }

    #[test]
    fn pcm16_roundtrip_is_sample_exact() {
        let samples: Vec<f32> = (0..4410).map(|i| ((i as f32) * 0.01).sin() * 0.5).collect();
        let bytes = wav_bytes(&samples, 1, 44_100, Subtype::Pcm16).unwrap();
        let audio = decoded(&bytes);
        assert_eq!(audio.channels, 1);
        assert_eq!(audio.sample_rate, 44_100);
        assert_eq!(audio.samples.len(), samples.len());
        for (written, read) in samples.iter().zip(&audio.samples) {
            assert!(
                (written - read).abs() < 1.0 / 32768.0,
                "{written} vs {read}"
            );
        }
    }

    #[test]
    fn stereo_and_24_bit_keep_the_layout() {
        let samples: Vec<f32> = (0..2000)
            .map(|i| if i % 2 == 0 { 0.25 } else { -0.25 })
            .collect();
        let audio = decoded(&wav_bytes(&samples, 2, 48_000, Subtype::Pcm24).unwrap());
        assert_eq!((audio.channels, audio.sample_rate), (2, 48_000));
        assert_eq!(audio.samples.len(), samples.len());
        assert!((audio.samples[0] - 0.25).abs() < 1e-5);
        assert!((audio.samples[1] + 0.25).abs() < 1e-5);
    }

    #[test]
    fn float32_roundtrip_is_exact() {
        let samples = vec![0.1f32, -0.1, 0.0, 1.0, -1.0, 0.333_333];
        let audio = decoded(&wav_bytes(&samples, 1, 8_000, Subtype::Float32).unwrap());
        assert_eq!(audio.samples, samples);
    }

    #[test]
    fn clamps_out_of_range_samples() {
        let audio = decoded(&wav_bytes(&[2.0, -2.0], 1, 8_000, Subtype::Pcm16).unwrap());
        assert!(audio.samples.iter().all(|s| s.abs() <= 1.0));
    }

    #[test]
    fn odd_pcm24_chunk_size_includes_the_padding_byte() {
        let bytes = wav_bytes(&[0.5], 1, 44_100, Subtype::Pcm24).unwrap();
        let riff_size = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let data_size = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
        assert_eq!(data_size, 3, "chunk length excludes its alignment padding");
        assert_eq!(riff_size as usize + 8, bytes.len());
        assert_eq!(bytes.last(), Some(&0));
    }

    #[test]
    fn wide_channel_layout_does_not_overflow_before_dividing_by_eight() {
        let bytes = wav_bytes(&vec![0.0; 3_000], 3_000, 8_000, Subtype::Float32).unwrap();
        assert_eq!(
            u16::from_le_bytes(bytes[32..34].try_into().unwrap()),
            12_000
        );
        assert_eq!(
            u32::from_le_bytes(bytes[28..32].try_into().unwrap()),
            96_000_000
        );
    }

    #[test]
    fn header_limits_are_checked_without_allocating_the_audio() {
        for subtype in [Subtype::Pcm16, Subtype::Pcm24, Subtype::Float32] {
            assert!(wav_layout(0, 0, 44_100, subtype).is_err());
            assert!(wav_layout(0, 1, 0, subtype).is_err());
            assert!(wav_layout(1, 2, 44_100, subtype).is_err());
            assert!(wav_layout(0, u16::MAX, 44_100, subtype).is_err());
            assert!(wav_layout(0, 2, u32::MAX, subtype).is_err());
            assert!(wav_layout(usize::MAX, 1, 44_100, subtype).is_err());
        }
        let largest_pcm16 = (u32::MAX as usize - 36) / 2;
        let valid = wav_layout(largest_pcm16, 1, 8_000, Subtype::Pcm16).unwrap();
        assert_eq!(valid.riff_len, u32::MAX - 1);
        assert!(wav_layout(largest_pcm16 + 1, 1, 8_000, Subtype::Pcm16).is_err());
        // The data itself fits, but the alignment byte would overflow RIFF.
        let odd_pcm24 = (u32::MAX as usize - 36) / 3;
        assert!(wav_layout(odd_pcm24, 1, 8_000, Subtype::Pcm24).is_err());
        assert!(wav_layout(odd_pcm24 - 1, 1, 8_000, Subtype::Pcm24).is_ok());
    }

    #[test]
    fn non_finite_samples_are_rejected_before_touching_the_destination() {
        let path = std::env::temp_dir().join(format!(
            "sound_rs_invalid_samples_{}.wav",
            std::process::id()
        ));
        std::fs::write(&path, b"keep this file").unwrap();
        for subtype in [Subtype::Pcm16, Subtype::Pcm24, Subtype::Float32] {
            for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                let err = write(path.to_str().unwrap(), &[sample], 1, 44_100, subtype).unwrap_err();
                assert_eq!(err.code(), "usage");
                assert_eq!(std::fs::read(&path).unwrap(), b"keep this file");
            }
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn destination_errors_keep_the_io_code() {
        let path =
            std::env::temp_dir().join(format!("sound_rs_directory_{}.wav", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        let err = write(path.to_str().unwrap(), &[0.0], 1, 8_000, Subtype::Pcm16).unwrap_err();
        std::fs::remove_dir(&path).unwrap();
        assert_eq!(err.code(), "io");
        assert!(err.to_string().contains(path.to_str().unwrap()));
    }

    #[test]
    fn rejects_unknown_subtype_and_extension() {
        assert!(Subtype::parse(Some("mp3")).is_err());
        assert!(write("/tmp/x.flac", &[0.0], 1, 44_100, Subtype::Pcm16).is_err());
    }

    #[test]
    fn writes_a_readable_file() {
        let dir = std::env::temp_dir().join(format!("sound_rs_write_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.wav");
        let samples = vec![0.5f32; 200];
        write(path.to_str().unwrap(), &samples, 1, 22_050, Subtype::Pcm16).unwrap();
        let audio = decoded(&std::fs::read(&path).unwrap());
        std::fs::remove_dir_all(&dir).ok();
        assert_eq!(audio.sample_rate, 22_050);
        assert_eq!(audio.samples.len(), samples.len());
    }
}
