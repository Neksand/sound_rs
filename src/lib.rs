//! Fast cross-platform audio playback for Python.
//!
//! Decoding is done with symphonia, output with cpal. Every decoded format is
//! converted to interleaved f32 in one place, then handed to the device with
//! whatever sample format that device actually wants.

mod audio;
mod cli;
mod player;
mod recorder;
mod writer;

use player::Player;
use pyo3::create_exception;
use pyo3::prelude::*;

create_exception!(sound_rs, SoundRsError, pyo3::exceptions::PyRuntimeError);

/// Every failure from this crate is a `SoundRsError` (a `RuntimeError` subclass)
/// carrying a stable `.code` attribute. Codes are part of the public contract:
/// `io`, `format`, `device`, `stalled`, `unsupported`, `usage`.
pub(crate) fn error_with(code: &str, message: String) -> PyErr {
    use pyo3::types::PyAnyMethods;

    let err = SoundRsError::new_err(message);
    Python::with_gil(|py| {
        let _ = err.value(py).setattr("code", code);
    });
    err
}

/// Audio failures are either a file we could not open (`io`) or a container we
/// could not understand (`format`).
pub(crate) fn error(message: String) -> PyErr {
    let code = if message.starts_with("Cannot open") {
        "io"
    } else {
        "format"
    };
    error_with(code, message)
}

#[cfg(feature = "numpy")]
fn array_channels(channels: usize) -> PyResult<u16> {
    u16::try_from(channels)
        .ok()
        .filter(|&channels| channels != 0)
        .ok_or_else(|| error_with("usage", "Array must have 1..65535 channels".to_string()))
}

/// Extracts interleaved f32 samples from a 1D mono or 2D `(frames, channels)`
/// array, the layout `load` returns.
pub(crate) fn numpy_frames(array: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, u16)> {
    #[cfg(feature = "numpy")]
    {
        use numpy::{PyArray1, PyArray2, PyArrayMethods};

        if let Ok(array) = array.extract::<Bound<'_, PyArray1<f32>>>() {
            // `as_slice` rejects non-contiguous views; iterating the ndarray
            // view handles strides and copies, which is what callers expect.
            let borrowed = array
                .try_readonly()
                .map_err(|e| error_with("usage", e.to_string()))?;
            let view = borrowed.as_array();
            return Ok((view.iter().copied().collect(), 1));
        }

        if let Ok(array) = array.extract::<Bound<'_, PyArray2<f32>>>() {
            let borrowed = array
                .try_readonly()
                .map_err(|e| error_with("usage", e.to_string()))?;
            let view = borrowed.as_array();
            let (frames, channels) = view.dim();
            let channel_count = array_channels(channels)?;
            let mut interleaved = Vec::with_capacity(frames * channels);
            for frame in 0..frames {
                for channel in 0..channels {
                    interleaved.push(view[[frame, channel]]);
                }
            }
            return Ok((interleaved, channel_count));
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Expected a 1D or 2D float32 array with shape (frames,) or (frames, channels)",
        ))
    }

    #[cfg(not(feature = "numpy"))]
    {
        let _ = array;
        Err(error_with(
            "unsupported",
            "NumPy support not enabled. Build with the 'numpy' feature.".to_string(),
        ))
    }
}

/// Writes an array to a file. Shared by `write()` and `Recorder.record_to()`.
pub(crate) fn write_array(
    path: &str,
    audio: &Bound<'_, PyAny>,
    sample_rate: u32,
    subtype: &str,
) -> PyResult<()> {
    let extension = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if extension != "wav" {
        return Err(error_with(
            "usage",
            format!("Only .wav can be written so far (got '.{extension}')"),
        ));
    }
    let format =
        writer::Subtype::parse(Some(subtype)).map_err(|message| error_with("usage", message))?;
    let (samples, channels) = numpy_frames(audio)?;
    audio
        .py()
        .allow_threads(|| writer::write(path, &samples, channels, sample_rate, format))
        .map_err(|err| error_with(err.code(), err.to_string()))
}

/// Write audio to a WAV file. `audio` is `(frames,)` or `(frames, channels)`,
/// the same layout `load` returns; `subtype` is `pcm16`, `pcm24` or `float32`.
#[pyfunction]
#[pyo3(signature = (file_path, audio, sample_rate, subtype="pcm16"))]
fn write(
    file_path: &str,
    audio: Bound<'_, PyAny>,
    sample_rate: u32,
    subtype: &str,
) -> PyResult<()> {
    write_array(file_path, &audio, sample_rate, subtype)
}

/// Extracts interleaved f32 samples from a 1D mono or 2D `(channels, frames)`
/// numpy array, and the channel count that goes with them.
pub(crate) fn numpy_samples(array: &Bound<'_, PyAny>) -> PyResult<(Vec<f32>, u16)> {
    #[cfg(feature = "numpy")]
    {
        use numpy::{PyArray1, PyArray2, PyArrayMethods};

        if let Ok(array) = array.extract::<Bound<'_, PyArray1<f32>>>() {
            // `as_slice` rejects non-contiguous views; iterating the ndarray
            // view handles strides and copies, which is what callers expect.
            let borrowed = array
                .try_readonly()
                .map_err(|e| error_with("usage", e.to_string()))?;
            let view = borrowed.as_array();
            return Ok((view.iter().copied().collect(), 1));
        }

        if let Ok(array) = array.extract::<Bound<'_, PyArray2<f32>>>() {
            let borrowed = array
                .try_readonly()
                .map_err(|e| error_with("usage", e.to_string()))?;
            let view = borrowed.as_array();
            let (channels, frames) = view.dim();
            let channel_count = array_channels(channels)?;
            let mut interleaved = Vec::with_capacity(channels * frames);
            for frame in 0..frames {
                for channel in 0..channels {
                    interleaved.push(view[[channel, frame]]);
                }
            }
            return Ok((interleaved, channel_count));
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "Input must be a 1D or 2D numpy array of float32",
        ))
    }

    #[cfg(not(feature = "numpy"))]
    {
        let _ = array;
        Err(error_with(
            "unsupported",
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
fn is_supported_format(py: Python<'_>, file_path: &str) -> bool {
    py.allow_threads(|| audio::open(file_path).is_ok())
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
///
/// `sample_rate` resamples, `mono` folds every channel into one, `offset` and
/// `duration` are seconds and decode only that window.
#[pyfunction]
#[pyo3(signature = (file_path, sample_rate=None, mono=false, offset=0.0, duration=None))]
fn load<'py>(
    py: Python<'py>,
    file_path: &str,
    sample_rate: Option<u32>,
    mono: bool,
    offset: f64,
    duration: Option<f64>,
) -> PyResult<(Bound<'py, PyAny>, u32)> {
    if sample_rate == Some(0) {
        return Err(error_with(
            "usage",
            "sample_rate must be positive".to_string(),
        ));
    }
    if !offset.is_finite() || offset < 0.0 || duration.is_some_and(|d| !d.is_finite() || d < 0.0) {
        return Err(error_with(
            "usage",
            "offset and duration must be finite and non-negative".to_string(),
        ));
    }
    let options = audio::LoadOptions {
        sample_rate,
        mono,
        offset,
        duration,
    };
    let audio = py
        .allow_threads(|| audio::decode_with_options(file_path, options))
        .map_err(error)?;
    into_array(py, audio)
}

pub(crate) fn into_array(py: Python<'_>, audio: audio::Audio) -> PyResult<(Bound<'_, PyAny>, u32)> {
    #[cfg(not(feature = "numpy"))]
    {
        let _ = (py, audio);
        return Err(error_with(
            "unsupported",
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

/// Decode audio bytes already in memory. Returns the same shape and rate as
/// `load`. MP4 priming is read from the bytes; MP3 uses Symphonia here rather
/// than the minimp3 file decoder, so the two MP3 paths can differ in samples.
#[pyfunction]
#[pyo3(signature = (data))]
fn load_bytes<'py>(py: Python<'py>, data: &[u8]) -> PyResult<(Bound<'py, PyAny>, u32)> {
    let audio = py
        .allow_threads(|| audio::decode_bytes(data))
        .map_err(error)?;
    into_array(py, audio)
}

/// Decode many files in parallel. Returns `[(array, sample_rate), ...]` in the
/// same order as `paths`. Failures raise for the first file that failed.
#[pyfunction]
#[pyo3(signature = (paths, threads=None, sample_rate=None, mono=false))]
fn load_many<'py>(
    py: Python<'py>,
    paths: Vec<String>,
    threads: Option<usize>,
    sample_rate: Option<u32>,
    mono: bool,
) -> PyResult<Vec<(Bound<'py, PyAny>, u32)>> {
    use rayon::prelude::*;

    if sample_rate == Some(0) {
        return Err(error_with(
            "usage",
            "sample_rate must be positive".to_string(),
        ));
    }

    let options = audio::LoadOptions {
        sample_rate,
        mono,
        offset: 0.0,
        duration: None,
    };
    let decoded: Vec<Result<audio::Audio, String>> = py.allow_threads(|| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads.unwrap_or(0))
            .build();
        match pool {
            Ok(pool) => pool.install(|| {
                paths
                    .par_iter()
                    .map(|p| audio::decode_with_options(p, options))
                    .collect()
            }),
            Err(_) => paths
                .iter()
                .map(|p| audio::decode_with_options(p, options))
                .collect(),
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
    if sample_rate == 0 {
        return Err(error_with(
            "usage",
            "sample_rate must be positive".to_string(),
        ));
    }
    let (samples, channels) = numpy_samples(&array)?;
    let mut player = Player::new(None)?;
    player.play_samples(samples, channels, sample_rate)?;
    player.wait(py)
}

/// Generate a sine tone and play it.
#[pyfunction]
fn play_sine_wave(py: Python<'_>, frequency: f32, duration: f32, sample_rate: u32) -> PyResult<()> {
    if sample_rate == 0
        || !frequency.is_finite()
        || frequency < 0.0
        || !duration.is_finite()
        || duration < 0.0
    {
        return Err(error_with(
            "usage",
            "frequency and duration must be finite and non-negative; sample_rate must be positive"
                .to_string(),
        ));
    }
    let count = duration as f64 * sample_rate as f64;
    if count >= (isize::MAX as usize / std::mem::size_of::<f32>()) as f64 {
        return Err(error_with(
            "usage",
            "Tone is too long to fit in memory".to_string(),
        ));
    }
    let frames = count as usize;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(frames)
        .map_err(|_| error_with("unsupported", "Cannot allocate tone buffer".to_string()))?;
    // Use f64 phase arithmetic so finite f32 inputs cannot overflow to NaN.
    let step = frequency as f64 * 2.0 * std::f64::consts::PI / sample_rate as f64;
    py.allow_threads(|| {
        samples.extend((0..frames).map(|i| ((i as f64 * step).sin() * 0.5) as f32))
    });

    let mut player = Player::new(None)?;
    player.play_samples(samples, 1, sample_rate)?;
    player.wait(py)
}

/// Output devices as `(index, name, is_default)`. Shared by the API and the CLI.
pub(crate) fn devices() -> Result<Vec<(usize, String, bool)>, String> {
    use cpal::traits::{DeviceTrait, HostTrait};

    let host = cpal::default_host();
    let default_name = host.default_output_device().and_then(|d| d.name().ok());
    let mut devices = Vec::new();
    let enumerated = host
        .output_devices()
        .map_err(|e| format!("Cannot enumerate output devices: {e}"))?
        .filter_map(|device| device.name().ok())
        .enumerate();
    for (index, name) in enumerated {
        let is_default = default_name.as_deref() == Some(name.as_str());
        devices.push((index, name, is_default));
    }
    Ok(devices)
}

/// List output devices as `(index, name, is_default)`.
#[pyfunction]
fn list_devices() -> PyResult<Vec<(usize, String, bool)>> {
    devices().map_err(|message| error_with("device", message))
}

/// List input devices as `(index, name, is_default)`.
#[pyfunction]
fn list_input_devices() -> PyResult<Vec<(usize, String, bool)>> {
    recorder::input_devices().map_err(|message| error_with("device", message))
}

#[pymodule]
fn sound_rs(_py: Python<'_>, m: Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Player>()?;
    m.add_class::<recorder::Recorder>()?;
    m.add("SoundRsError", _py.get_type::<SoundRsError>())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(play_audio, &m)?)?;
    m.add_function(wrap_pyfunction!(is_supported_format, &m)?)?;
    m.add_function(wrap_pyfunction!(get_audio_info, &m)?)?;
    m.add_function(wrap_pyfunction!(load, &m)?)?;
    m.add_function(wrap_pyfunction!(load_bytes, &m)?)?;
    m.add_function(wrap_pyfunction!(write, &m)?)?;
    m.add_function(wrap_pyfunction!(load_many, &m)?)?;
    m.add_function(wrap_pyfunction!(play_array, &m)?)?;
    m.add_function(wrap_pyfunction!(play_sine_wave, &m)?)?;
    m.add_function(wrap_pyfunction!(list_devices, &m)?)?;
    m.add_function(wrap_pyfunction!(list_input_devices, &m)?)?;
    m.add_function(wrap_pyfunction!(cli::cli, &m)?)?;
    Ok(())
}
