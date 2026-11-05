//! Fast cross-platform audio playback for Python.
//!
//! Decoding is done with symphonia, output with cpal. Every decoded format is
//! converted to interleaved f32 in one place, then handed to the device with
//! whatever sample format that device actually wants.

mod audio;
mod player;

use pyo3::create_exception;
use pyo3::prelude::*;
use player::Player;

create_exception!(sound_rs, SoundRsError, pyo3::exceptions::PyRuntimeError);

/// Every failure from this crate is a `SoundRsError` (a `RuntimeError` subclass),
pub(crate) fn error(message: String) -> PyErr {
    SoundRsError::new_err(message)
}

/// Extracts interleaved f32 samples from a 1D mono or 2D `(channels, frames)`
/// numpy array, and the channel count that goes with them.
pub(crate) fn numpy_samples(
    _py: Python<'_>,
    array: &Bound<'_, PyAny>,
    _sample_rate: u32,
) -> PyResult<(Vec<f32>, u16)> {
    #[cfg(feature = "numpy")]
    {
        use numpy::{PyArray1, PyArray2, PyArrayMethods};

        if let Ok(array) = array.extract::<Bound<'_, PyArray1<f32>>>() {
            let samples = unsafe { array.as_slice()? };
            return Ok((samples.to_vec(), 1));
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
            return Ok((interleaved, channels as u16));
        }

        return Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a 1D or 2D numpy array of float32",
        ));
    }

    #[cfg(not(feature = "numpy"))]
    {
        Err(error(
            "NumPy support not enabled. Build with the 'numpy' feature.".to_string(),
        ))
    }
}

/// Play audio file (anything symphonia can probe: WAV, MP3, FLAC, OGG, M4A, AAC).
///
/// Blocks until playback finishes and releases the GIL while it does.
#[pyfunction]
#[pyo3(signature = (file_path, device=None))]
fn play_audio(py: Python<'_>, file_path: &str, device: Option<String>) -> PyResult<()> {
    let mut player = Player::new(device)?;
    player.play_file(py, file_path)?;
    player.wait(py)
}

/// Check if the file can be decoded, without decoding it.
#[pyfunction]
fn is_supported_format(file_path: &str) -> bool {
    audio::open(file_path).is_ok()
}

/// Get `(sample_rate, channels, duration_seconds)` from the container header.
///
/// Only decodes when the header does not carry a frame count (some VBR MP3s),
/// or one packet for AAC, whose container has no channel layout.
#[pyfunction]
fn get_audio_info(py: Python<'_>, file_path: &str) -> PyResult<(u32, u16, f32)> {
    py.allow_threads(|| audio::info_or_decode(file_path))
        .map_err(error)
}

/// Decode a file to `(array, sample_rate)`, shape `(frames, channels)`, float32.
#[pyfunction]
fn load<'py>(py: Python<'py>, file_path: &str) -> PyResult<(Bound<'py, PyAny>, u32)> {
    let audio = py.allow_threads(|| audio::decode(file_path)).map_err(error)?;
    into_array(py, audio)
}

fn into_array(py: Python<'_>, audio: audio::Audio) -> PyResult<(Bound<'_, PyAny>, u32)> {
    #[cfg(not(feature = "numpy"))]
    {
        let _ = (py, audio);
        return Err(error(
            "NumPy support not enabled. Build with the 'numpy' feature.".to_string(),
        ));
    }

    #[cfg(feature = "numpy")]
    {
        use numpy::{PyArray1, PyArrayMethods};

        let channels = audio.channels.max(1) as usize;
        let frames = audio.samples.len() / channels;
        let flat = PyArray1::from_vec(py, audio.samples);
        let shaped = flat
            .reshape((frames, channels))
            .map_err(|e| error(e.to_string()))?;
        Ok((shaped.into_any(), audio.sample_rate))
    }
}

/// Decode many files in parallel. Returns `[(array, sample_rate), ...]` in the
/// same order as `paths`. Failures raise for the first file that failed.
#[pyfunction]
#[pyo3(signature = (paths, threads=None))]
fn load_many<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    threads: Option<usize>,
) -> PyResult<Vec<(Bound<'py, PyAny>, u32)>> {
    use rayon::prelude::*;

    let decoded: Vec<Result<audio::Audio, String>> = py.allow_threads(|| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.unwrap_or(0))
            .build();
        match pool {
            Ok(pool) => pool.install(|| paths.par_iter().map(|p| audio::decode(p)).collect()),
            Err(_) => paths.iter().map(|p| audio::decode(p)).collect(),
        }
    });

    decoded
        .into_iter()
        .map(|result| {
            let audio = result.map_err(error)?;
            into_array(py, audio)
        })
        .collect()
}

/// Play a NumPy array of f32: 1D mono `(samples,)` or 2D `(channels, samples)`.
#[pyfunction]
fn play_array(py: Python<'_>, array: Bound<'_, PyAny>, sample_rate: u32) -> PyResult<()> {
    let mut player = Player::new(None)?;
    player.play_array(py, array, sample_rate)?;
    player.wait(py)
}

/// Generate a sine tone and play it.
#[pyfunction]
fn play_sine_wave(py: Python<'_>, frequency: f32, duration: f32, sample_rate: u32) -> PyResult<()> {
    let frames = (duration.max(0.0) * sample_rate as f32) as usize;
    let step = frequency * 2.0 * std::f32::consts::PI / sample_rate.max(1) as f32;
    let samples: Vec<f32> = (0..frames).map(|i| (i as f32 * step).sin() * 0.5).collect();

    let mut player = Player::new(None)?;
    player.play_samples(samples, 1, sample_rate.max(1))?;
    player.wait(py)
}

/// List output devices as `(index, name, is_default)`.
#[pyfunction]
fn list_devices() -> PyResult<Vec<(usize, String, bool)>> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    let default_name = host.default_output_device().and_then(|d| d.name().ok());
    let mut devices = Vec::new();
    let enumerated = host
        .output_devices()
        .map_err(|e| error(format!("Cannot enumerate output devices: {e}")))?
        .filter_map(|device| device.name().ok())
        .enumerate();
    for (index, name) in enumerated {
        let is_default = default_name.as_deref() == Some(name.as_str());
        devices.push((index, name, is_default));
    }
    Ok(devices)
}

#[pymodule]
fn sound_rs(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Player>()?;
    m.add("SoundRsError", _py.get_type::<SoundRsError>())?;
    m.add_function(wrap_pyfunction!(play_audio, &m)?)?;
    m.add_function(wrap_pyfunction!(is_supported_format, &m)?)?;
    m.add_function(wrap_pyfunction!(get_audio_info, &m)?)?;
    m.add_function(wrap_pyfunction!(load, &m)?)?;
    m.add_function(wrap_pyfunction!(load_many, &m)?)?;
    m.add_function(wrap_pyfunction!(play_array, &m)?)?;
    m.add_function(wrap_pyfunction!(play_sine_wave, &m)?)?;
    m.add_function(wrap_pyfunction!(list_devices, &m)?)?;
    Ok(())
}
