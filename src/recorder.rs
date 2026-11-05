//! Recording from an input device.
//!
//! Same shape as the player: one stream, a callback that converts to f32, and a
//! buffer the Python side drains. The callback only appends to a preallocated
//! vector; it never allocates per callback in the normal case.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, StreamConfig};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::{error, error_with, into_array};

struct Capture {
    data: Vec<f32>,
    overflowed: bool,
}

/// Input device, opened once and kept open by the recorder.
pub struct Input {
    device: cpal::Device,
    config: StreamConfig,
    sample_format: SampleFormat,
    rate: u32,
    channels: u16,
}

impl Input {
    pub fn open(device_name: Option<&str>) -> Result<Self, String> {
        let host = cpal::default_host();
        let device = match device_name {
            Some(name) => host
                .input_devices()
                .map_err(|e| format!("Cannot enumerate input devices: {e}"))?
                .find(|d| d.name().map(|n| n == name).unwrap_or(false))
                .ok_or_else(|| format!("No input device named '{name}'"))?,
            None => host
                .default_input_device()
                .ok_or("No input device available")?,
        };

        let supported = device
            .default_input_config()
            .map_err(|e| format!("Cannot query input device: {e}"))?;

        Ok(Self {
            config: StreamConfig {
                channels: supported.channels(),
                sample_rate: supported.sample_rate(),
                buffer_size: cpal::BufferSize::Default,
            },
            sample_format: supported.sample_format(),
            rate: supported.sample_rate().0,
            channels: supported.channels(),
            device,
        })
    }

    fn build(&self, capture: Arc<Mutex<Capture>>) -> Result<cpal::Stream, String> {
        match self.sample_format {
            SampleFormat::F32 => build_input::<f32>(&self.device, &self.config, capture),
            SampleFormat::F64 => build_input::<f64>(&self.device, &self.config, capture),
            SampleFormat::I16 => build_input::<i16>(&self.device, &self.config, capture),
            SampleFormat::I32 => build_input::<i32>(&self.device, &self.config, capture),
            SampleFormat::U8 => build_input::<u8>(&self.device, &self.config, capture),
            SampleFormat::U16 => build_input::<u16>(&self.device, &self.config, capture),
            other => Err(format!("Unsupported input sample format: {other:?}")),
        }
    }
}

fn build_input<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    capture: Arc<Mutex<Capture>>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    device
        .build_input_stream(
            config,
            move |input: &[T], _: &cpal::InputCallbackInfo| {
                if let Ok(mut capture) = capture.lock() {
                    // Hard cap: ten minutes at 96 kHz stereo is ~230 MB. Beyond
                    // that the caller is not draining and we stop growing.
                    if capture.data.len() + input.len() > 230_000_000 {
                        capture.overflowed = true;
                        return;
                    }
                    capture
                        .data
                        .extend(input.iter().map(|s| f32::from_sample(*s)));
                }
            },
            |err| eprintln!("Input stream error: {err}"),
            None,
        )
        .map_err(|e| format!("Cannot open input stream: {e}"))
}

/// Records microphone input into an array.
#[pyclass(unsendable)]
pub struct Recorder {
    input: Option<Input>,
    capture: Arc<Mutex<Capture>>,
    stream: Option<cpal::Stream>,
}

#[pymethods]
impl Recorder {
    /// Opens an input device. Raises SoundRsError(code="device") when there is
    /// none, or when the OS denies microphone access.
    #[new]
    #[pyo3(signature = (device=None))]
    pub(crate) fn new(device: Option<String>) -> PyResult<Self> {
        let input =
            Input::open(device.as_deref()).map_err(|message| error_with("device", message))?;
        let capture = Arc::new(Mutex::new(Capture {
            data: Vec::new(),
            overflowed: false,
        }));
        Ok(Self {
            input: Some(input),
            capture,
            stream: None,
        })
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__<'py>(
        &mut self,
        py: Python<'py>,
        _ty: Bound<'py, PyAny>,
        _value: Bound<'py, PyAny>,
        _traceback: Bound<'py, PyAny>,
    ) {
        let _ = self.stop(py);
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.input.as_ref().map(|i| i.rate).unwrap_or(0)
    }

    #[getter]
    fn channels(&self) -> u16 {
        self.input.as_ref().map(|i| i.channels).unwrap_or(0)
    }

    #[getter]
    fn device(&self) -> String {
        self.input
            .as_ref()
            .and_then(|i| i.device.name().ok())
            .unwrap_or_else(|| "unknown".to_string())
    }

    /// Starts capturing. Samples accumulate until `stop()`.
    pub(crate) fn start(&mut self) -> PyResult<()> {
        if self.stream.is_some() {
            return Ok(());
        }
        let input = match &self.input {
            Some(input) => input,
            None => return Err(error_with("device", "Recorder is closed".to_string())),
        };
        {
            let mut capture = self
                .capture
                .lock()
                .map_err(|_| error("Recorder state poisoned".to_string()))?;
            capture.data.clear();
            capture.overflowed = false;
        }
        let stream = input
            .build(self.capture.clone())
            .map_err(|message| error_with("device", message))?;
        stream
            .play()
            .map_err(|e| error_with("device", format!("Capture failed: {e}")))?;
        self.stream = Some(stream);
        Ok(())
    }

    /// Stops capturing and returns `(array, sample_rate)`.
    pub(crate) fn stop<'py>(&mut self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, u32)> {
        if let Some(stream) = self.stream.take() {
            // Pause before dropping so no callback is in flight while the
            // buffer is drained.
            let _ = stream.pause();
            drop(stream);
        }
        let (samples, channels, rate) = {
            let input = match &self.input {
                Some(input) => input,
                None => return Err(error_with("device", "Recorder is closed".to_string())),
            };
            let mut capture = self
                .capture
                .lock()
                .map_err(|_| error("Recorder state poisoned".to_string()))?;
            if capture.overflowed {
                // Release ~230 MB before reporting, otherwise a failed take
                // keeps the buffer alive until the next start().
                capture.data = Vec::new();
                capture.overflowed = false;
                return Err(error_with(
                    "unsupported",
                    "Recording exceeded the 230 MB buffer; drain it more often".to_string(),
                ));
            }
            (
                std::mem::take(&mut capture.data),
                input.channels,
                input.rate,
            )
        };
        let audio = crate::audio::Audio {
            sample_rate: rate,
            channels,
            samples,
        };
        into_array(py, audio)
    }

    /// Records for `seconds` and returns `(array, sample_rate)`.
    pub(crate) fn record<'py>(
        &mut self,
        py: Python<'py>,
        seconds: f64,
    ) -> PyResult<(Bound<'py, PyAny>, u32)> {
        if !seconds.is_finite() {
            return Err(error_with(
                "usage",
                "record() needs a finite number of seconds".to_string(),
            ));
        }
        self.start()?;
        let wait = Duration::from_secs_f64(seconds.clamp(0.0, 3600.0));
        py.allow_threads(|| std::thread::sleep(wait));
        self.stop(py)
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        self.stream.take();
    }
}

/// Input devices as `(index, name, is_default)`.
pub(crate) fn input_devices() -> Result<Vec<(usize, String, bool)>, String> {
    let host = cpal::default_host();
    let default_name = host.default_input_device().and_then(|d| d.name().ok());
    let mut devices = Vec::new();
    for (index, name) in host
        .input_devices()
        .map_err(|e| format!("Cannot enumerate input devices: {e}"))?
        .filter_map(|device| device.name().ok())
        .enumerate()
    {
        let is_default = default_name.as_deref() == Some(name.as_str());
        devices.push((index, name, is_default));
    }
    Ok(devices)
}

#[cfg(test)]
mod tests {
    #[test]
    fn capture_buffer_cap_allows_long_recordings() {
        // The cap keeps a forgotten recorder from eating all memory while still
        // allowing a long take: 230 M samples is ~40 minutes at 48 kHz stereo.
        const CAP: usize = 230_000_000;
        let minutes = CAP as f64 / 48_000.0 / 2.0 / 60.0;
        assert!(minutes > 8.0, "{minutes} minutes");
    }
}
