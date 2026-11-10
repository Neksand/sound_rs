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

/// Builds a complete WAV file in memory.
pub(crate) fn wav_bytes(
    samples: &[f32],
    channels: u16,
    rate: u32,
    subtype: Subtype,
) -> Vec<u8> {
    let channels = channels.max(1) as usize;
    let block_align = channels as u16 * subtype.bits() / 8;
    let byte_rate = rate * block_align as u32;
    let data_len = (samples.len() / channels.max(1)) * block_align as usize;

    let mut out = Vec::with_capacity(44 + data_len);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&((36 + data_len) as u32).to_le_bytes());
    out.extend_from_slice(b"WAVE");

    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&subtype.format_tag().to_le_bytes());
    out.extend_from_slice(&(channels as u16).to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&subtype.bits().to_le_bytes());

    out.extend_from_slice(b"data");
    out.extend_from_slice(&(data_len as u32).to_le_bytes());
    for sample in samples {
        encode(*sample, subtype, &mut out);
    }
    // RIFF chunks are word-aligned.
    if data_len % 2 == 1 {
        out.push(0);
    }
    out
}

pub(crate) fn write(
    path: &str,
    samples: &[f32],
    channels: u16,
    rate: u32,
    subtype: Subtype,
) -> Result<(), String> {
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if extension != "wav" {
        return Err(format!(
            "Only .wav can be written so far (got '.{extension}'); save as WAV or convert afterwards"
        ));
    }
    if rate == 0 {
        return Err("sample_rate must be positive".to_string());
    }
    let bytes = wav_bytes(samples, channels, rate, subtype);
    std::fs::write(path, bytes).map_err(|e| format!("Cannot write '{path}': {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(bytes: &[u8]) -> crate::audio::Audio {
        crate::audio::decode_bytes(bytes).expect("written wav must decode")
    }

    #[test]
    fn pcm16_roundtrip_is_sample_exact() {
        let samples: Vec<f32> = (0..4410)
            .map(|i| ((i as f32) * 0.01).sin() * 0.5)
            .collect();
        let bytes = wav_bytes(&samples, 1, 44_100, Subtype::Pcm16);
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
        let audio = decoded(&wav_bytes(&samples, 2, 48_000, Subtype::Pcm24));
        assert_eq!((audio.channels, audio.sample_rate), (2, 48_000));
        assert_eq!(audio.samples.len(), samples.len());
        assert!((audio.samples[0] - 0.25).abs() < 1e-5);
        assert!((audio.samples[1] + 0.25).abs() < 1e-5);
    }

    #[test]
    fn float32_roundtrip_is_exact() {
        let samples = vec![0.1f32, -0.1, 0.0, 1.0, -1.0, 0.333_333];
        let audio = decoded(&wav_bytes(&samples, 1, 8_000, Subtype::Float32));
        assert_eq!(audio.samples, samples);
    }

    #[test]
    fn clamps_out_of_range_samples() {
        let audio = decoded(&wav_bytes(&[2.0, -2.0], 1, 8_000, Subtype::Pcm16));
        assert!(audio.samples.iter().all(|s| s.abs() <= 1.0));
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
