//! Recording from an input device.
//!
//! The callback only converts samples into a bounded ring. A collector thread
//! owns the growing recording buffer, so allocation and synchronization stay
//! off the audio thread. Overflow and backend faults invalidate the take.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, Sample, SampleFormat, SizedSample, StreamConfig};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{error_with, into_array};

// Preserve the published 230 M sample limit (920 MB of float32 payload).
const MAX_CAPTURE_SAMPLES: usize = 230_000_000;
// Usually one second of input, capped at 4 MiB even for unusual device formats.
const MAX_RING_SAMPLES: usize = 1_048_576;

type Collector = std::thread::JoinHandle<Result<Vec<f32>, CaptureError>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum CaptureError {
    Overflow = 1,
    Limit,
    Device,
    Allocation,
    Worker,
}

impl CaptureError {
    fn into_pyerr(self) -> PyErr {
        let (code, message) = match self {
            Self::Overflow => (
                "stalled",
                "Recording input buffer overflowed; the collector could not keep up",
            ),
            Self::Limit => (
                "unsupported",
                "Recording exceeded the 230 M sample (920 MB) limit; use a shorter take",
            ),
            Self::Device => (
                "device",
                "Input stream failed while recording; the device may have disconnected",
            ),
            Self::Allocation => ("unsupported", "Cannot allocate the recording buffer"),
            Self::Worker => ("stalled", "Recording collector stopped unexpectedly"),
        };
        error_with(code, message.to_string())
    }
}

#[derive(Default)]
struct CaptureStatus {
    failure: AtomicU8,
}

impl CaptureStatus {
    fn fail(&self, error: CaptureError) {
        // Keep the first fault: a later device callback must not hide overflow.
        let _ = self
            .failure
            .compare_exchange(0, error as u8, Ordering::AcqRel, Ordering::Acquire);
    }

    fn failure(&self) -> Option<CaptureError> {
        match self.failure.load(Ordering::Acquire) {
            1 => Some(CaptureError::Overflow),
            2 => Some(CaptureError::Limit),
            3 => Some(CaptureError::Device),
            4 => Some(CaptureError::Allocation),
            5 => Some(CaptureError::Worker),
            _ => None,
        }
    }
}

struct Capture {
    producer: rtrb::Producer<f32>,
    status: Arc<CaptureStatus>,
}

impl Capture {
    fn push<T: Sample>(&mut self, input: &[T])
    where
        f32: FromSample<T>,
    {
        // No lock, allocation, formatting, Python call or wait in the callback.
        if self.status.failure().is_some() {
            return;
        }
        if self.producer.is_abandoned() {
            self.status.fail(CaptureError::Worker);
            return;
        }
        // Admit a whole callback, preserving channel/frame order on overflow.
        if input.len() > self.producer.slots() {
            self.status.fail(CaptureError::Overflow);
            return;
        }
        for sample in input {
            if self.producer.push(f32::from_sample(*sample)).is_err() {
                self.status.fail(CaptureError::Overflow);
                return;
            }
        }
    }
}

fn collect(
    mut consumer: rtrb::Consumer<f32>,
    status: Arc<CaptureStatus>,
    limit: usize,
) -> Result<Vec<f32>, CaptureError> {
    let mut data = Vec::new();
    loop {
        if let Some(error) = status.failure() {
            return Err(error);
        }
        // Check abandonment first, then refresh the available count. Otherwise
        // a final callback racing with shutdown could be missed.
        let finished = consumer.is_abandoned();
        let available = consumer.slots();
        if available == 0 {
            if finished {
                return status.failure().map_or(Ok(data), Err);
            }
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }
        let Some(needed) = data.len().checked_add(available).filter(|n| *n <= limit) else {
            status.fail(CaptureError::Limit);
            return Err(status.failure().unwrap_or(CaptureError::Limit));
        };
        if needed > data.capacity() {
            // Explicit growth prevents Vec's geometric reserve from exceeding
            // the sample budget. All allocation happens on this worker.
            let capacity = data
                .capacity()
                .max(4096)
                .saturating_mul(2)
                .max(needed)
                .min(limit);
            if data.try_reserve_exact(capacity - data.len()).is_err() {
                status.fail(CaptureError::Allocation);
                return Err(status.failure().unwrap_or(CaptureError::Allocation));
            }
        }
        if let Ok(chunk) = consumer.read_chunk(available) {
            let (first, second) = chunk.as_slices();
            data.extend_from_slice(first);
            data.extend_from_slice(second);
            chunk.commit_all();
        }
    }
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

        if supported.channels() == 0 || supported.sample_rate().0 == 0 {
            return Err(
                "Input device reported an invalid channel count or sample rate".to_string(),
            );
        }
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

    fn build(&self, capture: Capture) -> Result<cpal::Stream, String> {
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
    mut capture: Capture,
) -> Result<cpal::Stream, String>
where
    T: SizedSample,
    f32: FromSample<T>,
{
    let status = capture.status.clone();
    device
        .build_input_stream(
            config,
            move |input: &[T], _: &cpal::InputCallbackInfo| capture.push(input),
            move |_err| status.fail(CaptureError::Device),
            None,
        )
        .map_err(|e| format!("Cannot open input stream: {e}"))
}

/// Records microphone input into an array.
#[pyclass(unsendable)]
pub struct Recorder {
    input: Option<Input>,
    capture: Arc<CaptureStatus>,
    stream: Option<cpal::Stream>,
    collector: Option<Collector>,
    /// Frames the caller asked for in `record()`; extra callbacks are trimmed.
    expected_frames: Option<usize>,
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
        let capture = Arc::new(CaptureStatus::default());
        Ok(Self {
            input: Some(input),
            capture,
            stream: None,
            collector: None,
            expected_frames: None,
        })
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__<'py>(
        &mut self,
        py: Python<'py>,
        ty: Bound<'py, PyAny>,
        _value: Bound<'py, PyAny>,
        _traceback: Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let result = self.finish_capture(py);
        // Preserve an exception raised inside the with block; otherwise expose
        // a recording fault instead of silently discarding it on context exit.
        if ty.is_none() {
            result.map(|_| ()).map_err(CaptureError::into_pyerr)
        } else {
            Ok(())
        }
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
    pub(crate) fn start(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.stream.is_some() {
            return Ok(());
        }
        let input = match &self.input {
            Some(input) => input,
            None => return Err(error_with("device", "Recorder is closed".to_string())),
        };
        let channels = input.channels as usize;
        let capacity = (input.rate as usize)
            .saturating_mul(channels)
            .min(MAX_RING_SAMPLES);
        let capacity = capacity - capacity % channels;
        let (producer, consumer) = rtrb::RingBuffer::new(capacity);
        let status = Arc::new(CaptureStatus::default());
        let capture = Capture {
            producer,
            status: status.clone(),
        };
        let stream = input
            .build(capture)
            .map_err(|message| error_with("device", message))?;
        let collector_status = status.clone();
        let limit = MAX_CAPTURE_SAMPLES - MAX_CAPTURE_SAMPLES % channels;
        let collector = std::thread::Builder::new()
            .name("sound-rs-capture".to_string())
            .spawn(move || collect(consumer, collector_status, limit))
            .map_err(|e| {
                error_with(
                    "unsupported",
                    format!("Cannot start recording collector: {e}"),
                )
            })?;
        if let Err(error) = stream.play() {
            drop(stream);
            let _ = py.detach(move || collector.join());
            return Err(error_with("device", format!("Capture failed: {error}")));
        }
        self.capture = status;
        self.stream = Some(stream);
        self.collector = Some(collector);
        self.expected_frames = None;
        Ok(())
    }

    /// Stops capturing and returns `(array, sample_rate)`.
    pub(crate) fn stop<'py>(&mut self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, u32)> {
        let samples = self.finish_capture(py).map_err(CaptureError::into_pyerr)?;
        let input = match &self.input {
            Some(input) => input,
            None => return Err(error_with("device", "Recorder is closed".to_string())),
        };
        let audio = crate::audio::Audio {
            sample_rate: input.rate,
            channels: input.channels,
            samples,
        };
        into_array(py, audio)
    }

    /// Records for `seconds` and writes the take to a WAV file.
    /// `subtype` is `pcm16`, `pcm24` or `float32`.
    #[pyo3(signature = (path, seconds, subtype="pcm16"))]
    pub(crate) fn record_to(
        &mut self,
        py: Python<'_>,
        path: &str,
        seconds: f64,
        subtype: &str,
    ) -> PyResult<()> {
        let (audio, rate) = self.record(py, seconds)?;
        crate::write_array(path, &audio, rate, subtype)
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
        let seconds = seconds.clamp(0.0, 3600.0);
        let input = self
            .input
            .as_ref()
            .ok_or_else(|| error_with("device", "Recorder is closed".to_string()))?;
        let frames = (seconds * input.rate as f64).round() as usize;
        if frames
            .checked_mul(input.channels as usize)
            .is_none_or(|n| n > MAX_CAPTURE_SAMPLES)
        {
            return Err(CaptureError::Limit.into_pyerr());
        }
        self.start(py)?;
        self.expected_frames = Some(frames);
        let wait = Duration::from_secs_f64(seconds);
        let status = self.capture.clone();
        py.detach(move || {
            let start = Instant::now();
            while status.failure().is_none() {
                let remaining = wait.saturating_sub(start.elapsed());
                if remaining.is_zero() {
                    break;
                }
                std::thread::sleep(remaining.min(Duration::from_millis(5)));
            }
        });
        self.stop(py)
    }
}

impl Recorder {
    fn finish_capture(&mut self, py: Python<'_>) -> Result<Vec<f32>, CaptureError> {
        let frames = self.expected_frames.take();
        // CPAL streams stay on their owning thread. Dropping the stream releases
        // the ring producer; the collector then drains every final sample.
        drop(self.stream.take());
        let mut samples = match self.collector.take() {
            Some(collector) => py
                .detach(move || collector.join())
                .unwrap_or(Err(CaptureError::Worker))?,
            None => Vec::new(),
        };
        if let (Some(frames), Some(input)) = (frames, &self.input) {
            samples.truncate(frames.saturating_mul(input.channels as usize));
        }
        Ok(samples)
    }
}

impl Drop for Recorder {
    fn drop(&mut self) {
        drop(self.stream.take());
        if let Some(collector) = self.collector.take() {
            let _ = collector.join();
        }
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
    use super::*;

    fn capture(capacity: usize) -> (Capture, rtrb::Consumer<f32>, Arc<CaptureStatus>) {
        let (producer, consumer) = rtrb::RingBuffer::new(capacity);
        let status = Arc::new(CaptureStatus::default());
        (
            Capture {
                producer,
                status: status.clone(),
            },
            consumer,
            status,
        )
    }

    #[test]
    fn collector_drains_final_samples_after_callback_is_dropped() {
        let (mut capture, consumer, status) = capture(8);
        capture.push(&[0.25f32, -0.5, 0.75, -1.0]);
        drop(capture);
        assert_eq!(
            collect(consumer, status, 8).unwrap(),
            [0.25, -0.5, 0.75, -1.0]
        );
    }

    #[test]
    fn callback_converts_integer_samples_without_changing_order() {
        let (mut capture, mut consumer, status) = capture(4);
        capture.push(&[i16::MIN, 0, i16::MAX]);
        assert_eq!(consumer.pop().unwrap(), -1.0);
        assert_eq!(consumer.pop().unwrap(), 0.0);
        capture.push(&[16384i16, -16384]); // wraps the ring
        drop(capture);
        let samples = collect(consumer, status, 4).unwrap();
        assert_eq!(samples, [i16::MAX as f32 / 32768.0, 0.5, -0.5]);
    }

    #[test]
    fn full_ring_reports_overflow_instead_of_returning_partial_take() {
        let (mut capture, consumer, status) = capture(2);
        capture.push(&[0.25f32, 0.5]);
        capture.push(&[0.75f32]);
        assert_eq!(status.failure(), Some(CaptureError::Overflow));
        drop(capture);
        assert_eq!(collect(consumer, status, 20), Err(CaptureError::Overflow));
    }

    #[test]
    fn collector_enforces_sample_limit_and_accepts_exact_limit() {
        let (mut capture, consumer, status) = capture(4);
        capture.push(&[0.0f32; 4]);
        drop(capture);
        assert_eq!(collect(consumer, status, 4).unwrap().len(), 4);
        let (mut capture, consumer, status) = self::capture(4);
        capture.push(&[0.0f32; 4]);
        drop(capture);
        assert_eq!(collect(consumer, status, 3), Err(CaptureError::Limit));
    }

    #[test]
    fn backend_error_is_reported_and_callback_stops_writing() {
        let (mut capture, mut consumer, status) = capture(8);
        status.fail(CaptureError::Device);
        capture.push(&[1.0f32, 2.0]);
        assert!(consumer.pop().is_err());
        drop(capture);
        assert_eq!(collect(consumer, status, 8), Err(CaptureError::Device));
    }

    #[test]
    fn first_capture_failure_is_preserved() {
        let status = CaptureStatus::default();
        status.fail(CaptureError::Overflow);
        status.fail(CaptureError::Device);
        assert_eq!(status.failure(), Some(CaptureError::Overflow));
    }

    #[test]
    fn repeated_worker_shutdown_does_not_lose_pending_samples() {
        for _ in 0..50 {
            let (mut capture, consumer, status) = capture(64);
            let worker = std::thread::spawn(move || collect(consumer, status, 64));
            capture.push(&[0.25f32; 64]);
            drop(capture);
            assert_eq!(worker.join().unwrap().unwrap(), vec![0.25; 64]);
        }
    }

    #[test]
    fn collector_preserves_multichannel_samples_across_many_ring_wraps() {
        let samples: Vec<f32> = (0..150_000).map(|n| n as f32 / 150_000.0).collect();
        let (mut capture, consumer, status) = capture(6 * 1024);
        let worker_status = status.clone();
        let worker = std::thread::spawn(move || collect(consumer, worker_status, 150_000));
        let deadline = Instant::now() + Duration::from_secs(5);
        for chunk in samples.chunks(6 * 256) {
            while capture.producer.slots() < chunk.len() {
                assert!(Instant::now() < deadline, "collector stalled");
                assert_eq!(status.failure(), None);
                std::thread::yield_now();
            }
            capture.push(chunk);
        }
        drop(capture);
        assert_eq!(worker.join().unwrap().unwrap(), samples);
    }

    #[test]
    fn empty_capture_stops_without_waiting_for_input() {
        let (capture, consumer, status) = capture(8);
        drop(capture);
        assert!(collect(consumer, status, 8).unwrap().is_empty());
    }

    #[test]
    fn overflow_rejects_entire_callback_without_splitting_a_frame() {
        let (mut capture, mut consumer, status) = capture(6);
        capture.push(&[0.0f32, 1.0, 2.0, 3.0]);
        capture.push(&[4.0f32, 5.0, 6.0, 7.0]);
        assert_eq!(status.failure(), Some(CaptureError::Overflow));
        let samples: Vec<_> = std::iter::from_fn(|| consumer.pop().ok()).collect();
        assert_eq!(samples, [0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn backend_failure_finishes_collector_before_stream_is_dropped() {
        let (capture, consumer, status) = capture(8);
        let collector_status = status.clone();
        let worker = std::thread::spawn(move || collect(consumer, collector_status, 8));
        status.fail(CaptureError::Device);
        assert_eq!(worker.join().unwrap(), Err(CaptureError::Device));
        drop(capture);
    }

    #[test]
    fn dropping_recorder_joins_its_collector() {
        use std::sync::atomic::AtomicBool;
        let (mut capture, consumer, status) = capture(8);
        let finished = Arc::new(AtomicBool::new(false));
        let worker_finished = finished.clone();
        let worker_status = status.clone();
        let worker = std::thread::spawn(move || {
            let result = collect(consumer, worker_status, 8);
            worker_finished.store(true, Ordering::Release);
            result
        });
        capture.push(&[0.5f32; 8]);
        let recorder = Recorder {
            input: None,
            capture: status,
            stream: None,
            collector: Some(worker),
            expected_frames: None,
        };
        drop(capture);
        drop(recorder);
        assert!(finished.load(Ordering::Acquire));
    }

    #[test]
    fn recording_budget_matches_published_sample_limit() {
        assert_eq!(
            MAX_CAPTURE_SAMPLES * std::mem::size_of::<f32>(),
            920_000_000
        );
    }
}
