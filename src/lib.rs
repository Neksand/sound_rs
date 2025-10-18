use pyo3::prelude::*;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::fs::File;
use std::io::Read;
use std::sync::{Arc, Mutex};

// WAV file data structure
struct WavData {
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    data: Vec<u8>,
}

impl WavData {
    fn load_from_file(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("Cannot open file: {}", e))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).map_err(|e| format!("Cannot read file: {}", e))?;

        // Basic WAV header validation
        if buffer.len() < 44 {
            return Err("File too small to be a WAV file".to_string());
        }

        if &buffer[0..4] != b"RIFF" {
            return Err("Not a RIFF file".to_string());
        }

        if &buffer[8..12] != b"WAVE" {
            return Err("Not a WAVE file".to_string());
        }

        // Find fmt chunk
        let mut fmt_offset = 12;
        while fmt_offset + 8 <= buffer.len() {
            let chunk_id = &buffer[fmt_offset..fmt_offset+4];
            let chunk_size = u32::from_le_bytes([
                buffer[fmt_offset+4], buffer[fmt_offset+5],
                buffer[fmt_offset+6], buffer[fmt_offset+7]
            ]) as usize;

            if chunk_id == b"fmt " {
                if chunk_size < 16 {
                    return Err("Invalid fmt chunk size".to_string());
                }

                let audio_format = u16::from_le_bytes([buffer[fmt_offset+8], buffer[fmt_offset+9]]);
                if audio_format != 1 {
                    return Err("Only PCM format supported".to_string());
                }

                let channels = u16::from_le_bytes([buffer[fmt_offset+10], buffer[fmt_offset+11]]);
                let sample_rate = u32::from_le_bytes([
                    buffer[fmt_offset+12], buffer[fmt_offset+13],
                    buffer[fmt_offset+14], buffer[fmt_offset+15]
                ]);
                let bits_per_sample = u16::from_le_bytes([buffer[fmt_offset+22], buffer[fmt_offset+23]]);

                // Find data chunk
                let mut data_offset = fmt_offset + 8 + chunk_size;
                while data_offset + 8 <= buffer.len() {
                    let data_chunk_id = &buffer[data_offset..data_offset+4];
                    let data_chunk_size = u32::from_le_bytes([
                        buffer[data_offset+4], buffer[data_offset+5],
                        buffer[data_offset+6], buffer[data_offset+7]
                    ]) as usize;

                    if data_chunk_id == b"data" {
                        let data_end = data_offset + 8 + data_chunk_size;
                        if data_end > buffer.len() {
                            return Err("Invalid data chunk size".to_string());
                        }

                        let audio_data = buffer[data_offset+8..data_end].to_vec();

                        return Ok(WavData {
                            sample_rate,
                            channels,
                            bits_per_sample,
                            data: audio_data,
                        });
                    }

                    data_offset += 8 + data_chunk_size;
                }

                return Err("No data chunk found".to_string());
            }

            fmt_offset += 8 + chunk_size;
        }

        Err("No fmt chunk found".to_string())
    }
}

// Convert raw bytes to f32 samples
fn decode_samples(wav_data: &WavData) -> Result<Vec<f32>, String> {
    match wav_data.bits_per_sample {
        8 => Ok(wav_data.data.iter().map(|&b| (b as f32 - 128.0) / 128.0).collect()),
        16 => {
            let mut result = Vec::with_capacity(wav_data.data.len() / 2);
            for chunk in wav_data.data.chunks(2) {
                if chunk.len() == 2 {
                    let sample = i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0;
                    result.push(sample);
                }
            }
            Ok(result)
        },
        _ => Err(format!("Unsupported bits per sample: {}", wav_data.bits_per_sample))
    }
}

// Resample audio to match device sample rate
fn resample_samples(samples: &[f32], src_rate: u32, dst_rate: u32, channels: u16) -> Vec<f32> {
    if src_rate == dst_rate {
        return samples.to_vec();
    }

    let ratio = src_rate as f64 / dst_rate as f64;
    let target_len = (samples.len() as f64 / ratio / channels as f64) as usize;
    let mut result = Vec::with_capacity(target_len * channels as usize);

    for i in 0..target_len {
        let src_index = (i as f64 * ratio * channels as f64) as usize;
        if src_index < samples.len() {
            result.push(samples[src_index]);
        }
    }
    result
}

// Convert between different channel layouts
fn convert_channels(samples: &[f32], src_channels: u16, dst_channels: u16) -> Vec<f32> {
    if src_channels == dst_channels {
        return samples.to_vec();
    }

    if src_channels == 1 && dst_channels == 2 {
        // Mono to Stereo - duplicate channel
        samples.iter().flat_map(|&s| [s, s]).collect()
    } else if src_channels == 2 && dst_channels == 1 {
        // Stereo to Mono - average channels
        samples.chunks(2).map(|c| c.iter().sum::<f32>() / c.len() as f32).collect()
    } else {
        // Generic channel conversion
        let mut result = Vec::new();
        for chunk in samples.chunks(src_channels as usize) {
            for i in 0..dst_channels as usize {
                result.push(*chunk.get(i).unwrap_or(&0.0));
            }
        }
        result
    }
}

// Core audio playback function
fn play_from_buffer(samples: &[f32], config: &cpal::StreamConfig) -> Result<(), String> {
    let host = cpal::default_host();
    let device = host.default_output_device()
        .ok_or("No output device available")?;

    let total_samples = samples.len();
    let current_sample = Arc::new(Mutex::new(0));
    let samples_data = Arc::new(samples.to_vec());

    let err_fn = |err| eprintln!("Audio stream error: {}", err);

    let stream = device.build_output_stream(
        config,
        move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
            let mut current = current_sample.lock().unwrap();

            for out_sample in output.iter_mut() {
                if *current < total_samples {
                    *out_sample = samples_data[*current];
                    *current += 1;
                } else {
                    *out_sample = 0.0;
                }
            }
        },
        err_fn,
        None,
    ).map_err(|e| format!("Stream creation failed: {}", e))?;

    stream.play().map_err(|e| format!("Playback failed: {}", e))?;

    // Wait for playback completion
    let duration = total_samples as f32 / config.sample_rate.0 as f32 / config.channels as f32;
    std::thread::sleep(std::time::Duration::from_secs_f32(duration + 0.1));

    Ok(())
}

// Stream audio in chunks for large files
fn play_stream_from_buffer(samples: &[f32], config: &cpal::StreamConfig, _chunk_size: usize) -> Result<(), String> {
    let host = cpal::default_host();
    let device = host.default_output_device()
        .ok_or("No output device available")?;

    let samples_data = Arc::new(samples.to_vec());
    let current_pos = Arc::new(Mutex::new(0));
    let total_samples = samples.len();

    let err_fn = |err| eprintln!("Audio stream error: {}", err);

    let stream = device.build_output_stream(
        config,
        {
            let samples_data = Arc::clone(&samples_data);
            let current_pos = Arc::clone(&current_pos);

            move |output: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut pos = current_pos.lock().unwrap();
                let output_len = output.len();

                for i in 0..output_len {
                    if *pos < total_samples {
                        output[i] = samples_data[*pos];
                        *pos += 1;
                    } else {
                        output[i] = 0.0;
                    }
                }
            }
        },
        err_fn,
        None,
    ).map_err(|e| format!("Stream creation failed: {}", e))?;

    stream.play().map_err(|e| format!("Playback failed: {}", e))?;

    // Wait for streaming completion
    while {
        let pos = current_pos.lock().unwrap();
        *pos < total_samples
    } {
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    std::thread::sleep(std::time::Duration::from_millis(200));
    Ok(())
}

// Get default audio device configuration
fn get_default_config() -> Result<cpal::StreamConfig, String> {
    let host = cpal::default_host();
    let device = host.default_output_device()
        .ok_or("No output device available")?;

    let supported_config = device.default_output_config()
        .map_err(|e| format!("Config error: {}", e))?;

    Ok(cpal::StreamConfig {
        channels: supported_config.channels(),
        sample_rate: supported_config.sample_rate(),
        buffer_size: cpal::BufferSize::Default,
    })
}

// Python interface functions

/// Play WAV audio file (original function)
#[pyfunction]
fn play_audio(file_path: &str) -> PyResult<()> {
    let wav_data = WavData::load_from_file(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;

    let config = get_default_config()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    // Audio processing pipeline
    let samples = decode_samples(&wav_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    let resampled = resample_samples(&samples, wav_data.sample_rate, config.sample_rate.0, wav_data.channels);
    let final_samples = convert_channels(&resampled, wav_data.channels, config.channels);

    play_from_buffer(&final_samples, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    Ok(())
}

/// Stream large audio file in chunks (non-blocking)
#[pyfunction]
fn play_audio_streamed(file_path: &str, chunk_size: Option<usize>) -> PyResult<()> {
    let wav_data = WavData::load_from_file(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;

    let config = get_default_config()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    let samples = decode_samples(&wav_data)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e))?;

    let resampled = resample_samples(&samples, wav_data.sample_rate, config.sample_rate.0, wav_data.channels);
    let final_samples = convert_channels(&resampled, wav_data.channels, config.channels);

    let chunk_size = chunk_size.unwrap_or(4096);
    play_stream_from_buffer(&final_samples, &config, chunk_size)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    Ok(())
}

/// Generate a simple sine wave and play it
#[pyfunction]
fn play_sine_wave(frequency: f32, duration: f32, sample_rate: u32) -> PyResult<()> {
    let config = get_default_config()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    let num_samples = (duration * sample_rate as f32) as usize;
    let mut samples = Vec::with_capacity(num_samples * 2); // Stereo

    for i in 0..num_samples {
        let t = i as f32 / sample_rate as f32;
        let value = (t * frequency * 2.0 * std::f32::consts::PI).sin() * 0.5;
        // Stereo: same signal on both channels
        samples.push(value);
        samples.push(value);
    }

    let resampled = resample_samples(&samples, sample_rate, config.sample_rate.0, 2);
    play_from_buffer(&resampled, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    Ok(())
}


// Conditional compilation for numpy feature
#[cfg(feature = "numpy")]
use numpy::{PyArray1, PyArray2, PyArrayMethods};

#[cfg(feature = "numpy")]
/// Play audio from NumPy array (1D or 2D)
#[pyfunction]
fn play_array(_py: Python, array: Bound<PyAny>, sample_rate: u32) -> PyResult<()> {
    let config = get_default_config()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    // Try to extract as 1D array first
    if let Ok(py_array) = array.extract::<Bound<PyArray1<f32>>>() {
        let samples = unsafe { py_array.as_slice()? };

        let resampled = resample_samples(samples, sample_rate, config.sample_rate.0, 1);
        play_from_buffer(&resampled, &config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        return Ok(());
    }

    // Try to extract as 2D array (channels x samples)
    if let Ok(py_array) = array.extract::<Bound<PyArray2<f32>>>() {
        let slice2d = unsafe { py_array.as_array() };
        let (channels, num_samples) = slice2d.dim();

        // Convert 2D array to interleaved format
        let mut interleaved = Vec::with_capacity(channels as usize * num_samples);
        for sample_idx in 0..num_samples {
            for channel_idx in 0..channels {
                interleaved.push(slice2d[[channel_idx, sample_idx]]);
            }
        }

        let resampled = resample_samples(&interleaved, sample_rate, config.sample_rate.0, channels as u16);
        let final_samples = convert_channels(&resampled, channels as u16, config.channels);

        play_from_buffer(&final_samples, &config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

        return Ok(());
    }

    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
        "Input must be a 1D or 2D numpy array of f32"
    ))
}

#[cfg(not(feature = "numpy"))]
/// Play audio from NumPy array (feature not enabled)
#[pyfunction]
fn play_array(_py: Python, _array: Bound<PyAny>, _sample_rate: u32) -> PyResult<()> {
    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        "NumPy support not enabled. Build with 'numpy' feature."
    ))
}

// Conditional compilation for async feature
#[cfg(feature = "async")]
use tokio::runtime::Runtime;

#[cfg(feature = "async")]
/// Async audio playback using Tokio
#[pyfunction]
fn play_audio_async(file_path: String) -> PyResult<()> {
    let rt = Runtime::new()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Tokio runtime error: {}", e)))?;

    rt.block_on(async {
        tokio::task::spawn_blocking(move || {
            play_audio(&file_path)
        }).await
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("Async task error: {}", e)))?
    })
}

#[cfg(not(feature = "async"))]
/// Async audio playback (feature not enabled)
#[pyfunction]
fn play_audio_async(_file_path: String) -> PyResult<()> {
    Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
        "Async support not enabled. Build with 'async' feature."
    ))
}

// Python module definition
#[pymodule]
fn sound_rs(_py: Python<'_>, m: Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(play_audio, &m)?)?;
    m.add_function(wrap_pyfunction!(play_array, &m)?)?;
    m.add_function(wrap_pyfunction!(play_audio_streamed, &m)?)?;
    m.add_function(wrap_pyfunction!(play_audio_async, &m)?)?;
    m.add_function(wrap_pyfunction!(play_sine_wave, &m)?)?;
    Ok(())
}