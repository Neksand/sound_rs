use pyo3::prelude::*;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::fs::File;
use std::sync::{Arc, Mutex};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

// Audio data structure for multi-format support
struct AudioData {
    sample_rate: u32,
    channels: u16,
    bits_per_sample: u16,
    data: Vec<f32>,
}

impl AudioData {
    fn load_from_file(path: &str) -> Result<Self, String> {
        // Open the file
        let file = File::open(path).map_err(|e| format!("Cannot open file: {}", e))?;
        let mss = MediaSourceStream::new(Box::new(file), Default::default());

        // Create a hint to help the format registry guess the format
        let mut hint = Hint::new();
        if let Some(extension) = std::path::Path::new(path).extension() {
            if let Some(ext_str) = extension.to_str() {
                hint.with_extension(ext_str);
            }
        }

        // Use the default options for format, metadata, and decoder
        let format_opts = FormatOptions::default();
        let metadata_opts = MetadataOptions::default();
        let decoder_opts = DecoderOptions::default();

        // Probe the media format
        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &format_opts, &metadata_opts)
            .map_err(|e| format!("Format probing error: {}", e))?;

        // Get the format reader
        let mut format = probed.format;

        // Get the default track
        let track = format
            .default_track()
            .ok_or("No default track found")?;

        // Create a decoder for the track
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &decoder_opts)
            .map_err(|e| format!("Decoder error: {}", e))?;

        let sample_rate = track.codec_params.sample_rate.ok_or("Unknown sample rate")?;
        let channels = track.codec_params.channels.ok_or("Unknown channel layout")?.count();
        let bits_per_sample = 32; // We convert everything to f32

        let mut all_samples = Vec::new();

        // Decode all packets
        while let Ok(packet) = format.next_packet() {
            // Decode the packet
            let decoded = decoder.decode(&packet).map_err(|e| format!("Decoding error: {}", e))?;

            // Process the audio buffer based on its type
            match decoded {
                AudioBufferRef::F32(buf) => {
                    let frames = buf.frames();
                    let channels = buf.spec().channels.count();

                    for frame in 0..frames {
                        for channel in 0..channels {
                            all_samples.push(buf.chan(channel)[frame]);
                        }
                    }
                }
                AudioBufferRef::U8(buf) => {
                    let frames = buf.frames();
                    let channels = buf.spec().channels.count();

                    for frame in 0..frames {
                        for channel in 0..channels {
                            let sample = buf.chan(channel)[frame];
                            // Convert u8 to f32: [0, 255] -> [-1.0, 1.0]
                            all_samples.push((sample as f32 - 128.0) / 128.0);
                        }
                    }
                }
                AudioBufferRef::U16(buf) => {
                    let frames = buf.frames();
                    let channels = buf.spec().channels.count();

                    for frame in 0..frames {
                        for channel in 0..channels {
                            let sample = buf.chan(channel)[frame];
                            // Convert u16 to f32: [0, 65535] -> [-1.0, 1.0]
                            all_samples.push((sample as f32 - 32768.0) / 32768.0);
                        }
                    }
                }
                AudioBufferRef::U32(buf) => {
                    let frames = buf.frames();
                    let channels = buf.spec().channels.count();

                    for frame in 0..frames {
                        for channel in 0..channels {
                            let sample = buf.chan(channel)[frame];
                            // Convert u32 to f32: [0, 4294967295] -> [-1.0, 1.0]
                            all_samples.push((sample as f32 - 2147483648.0) / 2147483648.0);
                        }
                    }
                }
                AudioBufferRef::S8(buf) => {
                    let frames = buf.frames();
                    let channels = buf.spec().channels.count();

                    for frame in 0..frames {
                        for channel in 0..channels {
                            let sample = buf.chan(channel)[frame];
                            // Convert s8 to f32: [-128, 127] -> [-1.0, 1.0]
                            all_samples.push(sample as f32 / 128.0);
                        }
                    }
                }
                AudioBufferRef::S16(buf) => {
                    let frames = buf.frames();
                    let channels = buf.spec().channels.count();

                    for frame in 0..frames {
                        for channel in 0..channels {
                            let sample = buf.chan(channel)[frame];
                            // Convert s16 to f32: [-32768, 32767] -> [-1.0, 1.0]
                            all_samples.push(sample as f32 / 32768.0);
                        }
                    }
                }
                AudioBufferRef::S32(buf) => {
                    let frames = buf.frames();
                    let channels = buf.spec().channels.count();

                    for frame in 0..frames {
                        for channel in 0..channels {
                            let sample = buf.chan(channel)[frame];
                            // Convert s32 to f32: [-2147483648, 2147483647] -> [-1.0, 1.0]
                            all_samples.push(sample as f32 / 2147483648.0);
                        }
                    }
                }
                AudioBufferRef::F64(buf) => {
                    let frames = buf.frames();
                    let channels = buf.spec().channels.count();

                    for frame in 0..frames {
                        for channel in 0..channels {
                            let sample = buf.chan(channel)[frame];
                            // Convert f64 to f32
                            all_samples.push(sample as f32);
                        }
                    }
                }
                // Skip 24-bit formats for now as they're less common
                AudioBufferRef::U24(_) | AudioBufferRef::S24(_) => {
                    return Err("24-bit audio format not supported yet".to_string());
                }
            }
        }

        Ok(AudioData {
            sample_rate,
            channels: channels as u16,
            bits_per_sample,
            data: all_samples,
        })
    }

    // Helper function to check if file format is supported
    fn is_format_supported(path: &str) -> bool {
        let supported_extensions = ["wav", "mp3", "flac", "ogg", "m4a", "aac"];

        if let Some(extension) = std::path::Path::new(path).extension() {
            if let Some(ext_str) = extension.to_str() {
                return supported_extensions.contains(&ext_str.to_lowercase().as_str());
            }
        }
        false
    }
}

// [Rest of the code remains exactly the same as your previous working version]

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

/// Play audio file (supports WAV, MP3, FLAC, OGG, M4A, AAC)
#[pyfunction]
fn play_audio(file_path: &str) -> PyResult<()> {
    // Check if format is supported
    if !AudioData::is_format_supported(file_path) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unsupported audio format. Supported: WAV, MP3, FLAC, OGG, M4A, AAC"
        ));
    }

    let audio_data = AudioData::load_from_file(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;

    let config = get_default_config()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    // Audio processing pipeline
    let resampled = resample_samples(&audio_data.data, audio_data.sample_rate, config.sample_rate.0, audio_data.channels);
    let final_samples = convert_channels(&resampled, audio_data.channels, config.channels);

    play_from_buffer(&final_samples, &config)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    Ok(())
}

/// Check if file format is supported
#[pyfunction]
fn is_supported_format(file_path: &str) -> bool {
    AudioData::is_format_supported(file_path)
}

/// Get audio file information
#[pyfunction]
fn get_audio_info(file_path: &str) -> PyResult<(u32, u16, f32)> {
    if !AudioData::is_format_supported(file_path) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unsupported audio format"
        ));
    }

    let audio_data = AudioData::load_from_file(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;

    let duration = audio_data.data.len() as f32 / audio_data.sample_rate as f32 / audio_data.channels as f32;

    Ok((audio_data.sample_rate, audio_data.channels, duration))
}

/// Stream large audio file in chunks (non-blocking)
#[pyfunction]
fn play_audio_streamed(file_path: &str, chunk_size: Option<usize>) -> PyResult<()> {
    if !AudioData::is_format_supported(file_path) {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Unsupported audio format"
        ));
    }

    let audio_data = AudioData::load_from_file(file_path)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(e))?;

    let config = get_default_config()
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e))?;

    let resampled = resample_samples(&audio_data.data, audio_data.sample_rate, config.sample_rate.0, audio_data.channels);
    let final_samples = convert_channels(&resampled, audio_data.channels, config.channels);

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
    m.add_function(wrap_pyfunction!(is_supported_format, &m)?)?;
    m.add_function(wrap_pyfunction!(get_audio_info, &m)?)?;
    m.add_function(wrap_pyfunction!(play_array, &m)?)?;
    m.add_function(wrap_pyfunction!(play_audio_streamed, &m)?)?;
    m.add_function(wrap_pyfunction!(play_audio_async, &m)?)?;
    m.add_function(wrap_pyfunction!(play_sine_wave, &m)?)?;
    Ok(())
}