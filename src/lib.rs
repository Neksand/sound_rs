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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
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

fn decode(path: &str) -> Result<Audio, String> {
    let mut format = open(path)?;

    let (sample_rate, channels, mut decoder) = {
        let track = format.default_track().ok_or("No audio track found")?;
        let sample_rate = track
            .codec_params
            .sample_rate
            .ok_or("Unknown sample rate")?;
        let channels = track
            .codec_params
            .channels
            .ok_or("Unknown channel layout")?
            .count() as u16;
        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .map_err(|e| format!("Unsupported codec: {e}"))?;
        (sample_rate, channels, decoder)
    };

    let mut samples = Vec::new();
    let mut buffer: Option<SampleBuffer<f32>> = None;

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
        samples.extend_from_slice(buffer.samples());
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

fn build_stream<T>(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    data: Arc<Vec<f32>>,
    position: Arc<AtomicUsize>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let total = data.len();
    device
        .build_output_stream(
            config,
            move |output: &mut [T], _: &cpal::OutputCallbackInfo| {
                for slot in output.iter_mut() {
                    let i = position.fetch_add(1, Ordering::Relaxed);
                    *slot = if i < total {
                        T::from_sample(data[i])
                    } else {
                        T::from_sample(0.0f32)
                    };
                }
            },
            |err| eprintln!("Audio stream error: {err}"),
            None,
        )
        .map_err(|e| format!("Cannot open output stream: {e}"))
}

/// The single playback path: resample to the device rate, fold to the device
/// channel count, open a stream in the device's own sample format, block until
/// the buffer is drained.
fn play(samples: &[f32], src_rate: u32, src_channels: u16) -> Result<(), String> {
    let device = cpal::default_host()
        .default_output_device()
        .ok_or("No output device available")?;
    let supported = device
        .default_output_config()
        .map_err(|e| format!("Cannot query output device: {e}"))?;

    let out_rate = supported.sample_rate().0;
    let out_channels = supported.channels();

    let ready = convert_channels(
        &resample(samples, src_rate, out_rate, src_channels),
        src_channels,
        out_channels,
    );
    let config = cpal::StreamConfig {
        channels: out_channels,
        sample_rate: supported.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    };

    let data = Arc::new(ready);
    let position = Arc::new(AtomicUsize::new(0));
    let stream = match supported.sample_format() {
        SampleFormat::F32 => build_stream::<f32>(&device, &config, data.clone(), position.clone()),
        SampleFormat::F64 => build_stream::<f64>(&device, &config, data.clone(), position.clone()),
        SampleFormat::I16 => build_stream::<i16>(&device, &config, data.clone(), position.clone()),
        SampleFormat::I32 => build_stream::<i32>(&device, &config, data.clone(), position.clone()),
        SampleFormat::U8 => build_stream::<u8>(&device, &config, data.clone(), position.clone()),
        SampleFormat::U16 => build_stream::<u16>(&device, &config, data.clone(), position.clone()),
        other => return Err(format!("Unsupported output sample format: {other:?}")),
    }?;

    stream.play().map_err(|e| format!("Playback failed: {e}"))?;

    while position.load(Ordering::Relaxed) < data.len() {
        std::thread::sleep(Duration::from_millis(20));
    }
    // Let the device finish playing what is already in its own buffer.
    std::thread::sleep(Duration::from_millis(100));
    Ok(())
}

fn play_file(path: &str) -> Result<(), String> {
    let audio = decode(path)?;
    play(&audio.samples, audio.sample_rate, audio.channels)
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

/// Get `(sample_rate, channels, duration_seconds)` of an audio file.
#[pyfunction]
fn get_audio_info(py: Python<'_>, file_path: &str) -> PyResult<(u32, u16, f32)> {
    let audio = py.allow_threads(|| decode(file_path)).map_err(to_pyerr)?;
    Ok((audio.sample_rate, audio.channels, audio.duration()))
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
                .allow_threads(|| play(samples, sample_rate, 1))
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
                .allow_threads(|| play(&interleaved, sample_rate, channels as u16))
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
    py.allow_threads(|| play(&samples, rate, 1))
        .map_err(to_pyerr)
}

#[pymodule]
fn sound_rs(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(play_audio, &m)?)?;
    m.add_function(wrap_pyfunction!(is_supported_format, &m)?)?;
    m.add_function(wrap_pyfunction!(get_audio_info, &m)?)?;
    m.add_function(wrap_pyfunction!(play_array, &m)?)?;
    m.add_function(wrap_pyfunction!(play_sine_wave, &m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

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
