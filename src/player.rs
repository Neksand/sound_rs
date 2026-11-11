//! Persistent audio output: one stream stays open, buffers are fed to it.
//!
//! Two sources feed the same callback: an in-memory buffer (already at the
//! device rate) and a decoder thread that streams a file through a ring
//! buffer, so a two-hour file costs a few megabytes instead of gigabytes.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, StreamConfig};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::audio::{self, convert_channels, resample_all};
use crate::{error, error_with};

/// How much decoded audio the decoder thread may run ahead of the device.
const RING_SECONDS: f64 = 1.0;

pub struct Core {
    pub channels: usize,
    pub rate: u32,
    pub volume: f32,
    pub playing: bool,
    pub finished: bool,
    /// Explicit pause intent survives a final callback or an empty decoder result.
    paused: bool,
    /// Consumed device samples (interleaved).
    pub position: usize,
    /// Total device samples this source will produce, once known.
    pub total: Option<usize>,
    /// Samples already played before the current position counter started,
    /// so seeking does not restart the clock at zero.
    pub base: usize,
    /// Track length in seconds, when the container knows it.
    pub duration: Option<f64>,
    pub memory: Option<Arc<Vec<f32>>>,
    pub consumer: Option<rtrb::Consumer<f32>>,
    pub error: Option<String>,
    pub error_code: &'static str,
    /// Seek requested by Python, in seconds.
    pub seek_to: Option<f64>,
    /// Seconds of audio still in flight when the last sample was written.
    pub drain: Duration,
    /// Callbacks that found the buffer empty while playing.
    pub underruns: u64,
    /// Last measured device latency, in milliseconds.
    pub latency_ms: Option<f64>,
    /// Paths waiting to play after the current one.
    pub queue: std::collections::VecDeque<String>,
    /// Set when a fed stream has been closed by the caller.
    pub end_of_stream: bool,
    /// The file currently owned by the decoder, including queued handovers.
    current_path: Option<String>,
    /// The backend error callback only stores this flag; it never locks or formats.
    device_failed: Arc<AtomicBool>,
}

impl Core {
    pub fn new(channels: usize, rate: u32) -> Self {
        Self {
            channels,
            rate,
            volume: 1.0,
            playing: false,
            finished: false,
            paused: false,
            position: 0,
            total: None,
            base: 0,
            duration: None,
            memory: None,
            consumer: None,
            error: None,
            error_code: "format",
            seek_to: None,
            drain: Duration::ZERO,
            underruns: 0,
            latency_ms: None,
            queue: std::collections::VecDeque::new(),
            end_of_stream: false,
            current_path: None,
            device_failed: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn seconds(&self) -> f64 {
        self.position as f64 / self.channels as f64 / self.rate as f64
    }

    pub fn duration(&self) -> Option<f64> {
        self.duration.or_else(|| {
            self.total
                .map(|total| total as f64 / self.channels as f64 / self.rate as f64)
        })
    }

    pub fn is_playing(&self) -> bool {
        self.playing
            && !self.paused
            && !self.finished
            && !self.device_failed.load(Ordering::Relaxed)
    }

    fn request_seek(&mut self, seconds: f64) -> Result<(), &'static str> {
        if !seconds.is_finite() {
            return Err("seek() needs a finite number of seconds");
        }
        if self.memory.is_none() && self.current_path.is_none() {
            return Err("seek() requires a file or an in-memory source");
        }
        let seconds = self
            .duration()
            .map(|duration| seconds.max(0.0).min(duration.max(0.0)))
            .unwrap_or_else(|| seconds.max(0.0));
        let frames = seconds * self.rate as f64;
        if !frames.is_finite() || frames >= (usize::MAX / self.channels.max(1)) as f64 {
            return Err("seek() position is too large");
        }
        let target = (frames as usize) * self.channels;
        // A seek after natural completion starts playback again; a paused
        // source remains paused and retains its queue.
        self.playing |= self.finished && !self.paused;
        self.finished = false;
        self.drain = Duration::ZERO;
        if self.memory.is_some() {
            self.position = target.min(self.total.unwrap_or(usize::MAX));
            self.base = self.position;
        } else {
            self.seek_to = Some(seconds);
            self.base = target;
            self.position = target;
            self.total = None;
        }
        Ok(())
    }

    fn enqueue(&mut self, path: &str) -> Result<(), &'static str> {
        if self.current_path.is_none() || self.error.is_some() {
            return Err("enqueue() requires a file started with play_file()");
        }
        if self.queue.len() >= 128 {
            return Err("Queue is full (128 entries); wait for playback to advance");
        }
        self.queue.push_back(path.to_string());
        Ok(())
    }

    fn pause(&mut self) {
        self.paused = true;
        self.playing = false;
    }

    fn take_queued_path(&mut self) -> Option<String> {
        // Completion and explicit pause are independent: a pause that wins
        // the lock after the final callback must still prevent queue handover.
        let next = if self.finished && !self.paused {
            self.queue.pop_front()
        } else {
            None
        };
        if let Some(path) = &next {
            self.current_path = Some(path.clone());
            self.consumer = None;
            self.position = 0;
            self.base = 0;
            self.total = None;
            self.duration = None;
            self.finished = false;
            self.end_of_stream = false;
            self.underruns = 0;
            self.drain = Duration::ZERO;
            self.playing = true;
        }
        next
    }

    fn resume(&mut self) -> Result<(), &'static str> {
        if self.memory.is_none() && self.current_path.is_none() && self.consumer.is_none() {
            return Err("resume() requires an active source");
        }
        if self.finished {
            if self.end_of_stream {
                return Err("A finished feed cannot restart; call begin_feed() again");
            }
            self.request_seek(0.0)?;
        }
        self.paused = false;
        self.playing = true;
        Ok(())
    }

    /// Called from the audio callback. Writes exactly `output.len()` samples:
    /// real audio while there is some, silence otherwise.
    pub fn fill<T>(&mut self, output: &mut [T], latency: Option<Duration>)
    where
        T: SizedSample + FromSample<f32>,
    {
        let silence = |output: &mut [T]| {
            for slot in output.iter_mut() {
                *slot = T::from_sample(0.0f32);
            }
        };

        if self.device_failed.load(Ordering::Relaxed) {
            self.playing = false;
        }
        if !self.playing || self.paused || self.seek_to.is_some() {
            silence(output);
            return;
        }
        let volume = self.volume;

        let written = if let Some(data) = &self.memory {
            let end = data.len().min(self.position + output.len());
            let chunk = &data[self.position.min(data.len())..end];
            for (slot, sample) in output.iter_mut().zip(chunk) {
                *slot = T::from_sample(*sample * volume);
            }
            chunk.len()
        } else if let Some(consumer) = &mut self.consumer {
            let mut written = 0;
            while written < output.len() {
                match consumer.pop() {
                    Ok(sample) => {
                        output[written] = T::from_sample(sample * volume);
                        written += 1;
                    }
                    Err(_) => break,
                }
            }
            written
        } else {
            0
        };

        self.position += written;
        silence(&mut output[written..]);
        if latency.is_some() {
            self.latency_ms = latency.map(|value| value.as_secs_f64() * 1000.0);
        }

        let source_ended = self.total.is_some_and(|total| self.position >= total)
            || (self.end_of_stream && self.consumer.as_ref().is_some_and(|ring| ring.is_empty()));
        if source_ended {
            if let Some(total) = self.total {
                self.position = total;
            }
            self.finished = true;
            self.playing = false;
            // The timestamp refers to the beginning of this output buffer.
            // Only the written prefix must reach the device; its silent tail
            // does not extend the track. An empty source has nothing to drain.
            self.drain = if self.position == 0 {
                Duration::ZERO
            } else {
                latency.unwrap_or(Duration::from_millis(100))
                    + Duration::from_secs_f64(
                        written as f64 / self.channels.max(1) as f64 / self.rate as f64,
                    )
            };
            return;
        }

        // Starvation after audio has started is a real glitch; the very first
        // callbacks before the decoder delivers its first packet are not.
        if written < output.len() && self.position > 0 {
            if self
                .total
                .map(|total| self.position < total)
                .unwrap_or(true)
            {
                self.underruns += 1;
            }
            self.latency_ms = latency.map(|value| value.as_secs_f64() * 1000.0);
        }
    }
}

/// The output device, opened once and kept open by the `Player`.
pub struct Output {
    pub device: cpal::Device,
    pub config: StreamConfig,
    pub sample_format: SampleFormat,
    host: cpal::Host,
}

impl Output {
    pub fn open(device_name: Option<&str>) -> Result<Self, String> {
        let host = cpal::default_host();
        let device = match device_name {
            Some(name) => host
                .output_devices()
                .map_err(|e| format!("Cannot enumerate output devices: {e}"))?
                .find(|d| d.name().map(|n| n == name).unwrap_or(false))
                .ok_or_else(|| format!("No output device named '{name}'"))?,
            None => host
                .default_output_device()
                .ok_or("No output device available")?,
        };

        let supported = device
            .default_output_config()
            .map_err(|e| format!("Cannot query output device: {e}"))?;
        let config = StreamConfig {
            channels: supported.channels(),
            sample_rate: supported.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };
        validate_output_config(&config)?;

        Ok(Self {
            config,
            sample_format: supported.sample_format(),
            device,
            host,
        })
    }

    pub fn channels(&self) -> usize {
        self.config.channels as usize
    }

    pub fn rate(&self) -> u32 {
        self.config.sample_rate.0
    }

    pub fn name(&self) -> String {
        self.device.name().unwrap_or_else(|_| "unknown".to_string())
    }

    /// Builds the output stream whose callback pulls from `core`.
    pub fn build_stream(&self, core: Arc<Mutex<Core>>) -> Result<cpal::Stream, String> {
        let _ = &self.host;
        let device = &self.device;
        let config = &self.config;

        macro_rules! build {
            ($ty:ty) => {
                build_stream::<$ty>(device, config, core.clone())
            };
        }

        match self.sample_format {
            SampleFormat::F32 => build!(f32),
            SampleFormat::F64 => build!(f64),
            SampleFormat::I16 => build!(i16),
            SampleFormat::I32 => build!(i32),
            SampleFormat::U8 => build!(u8),
            SampleFormat::U16 => build!(u16),
            other => Err(format!("Unsupported output sample format: {other:?}")),
        }
    }
}

fn validate_output_config(config: &StreamConfig) -> Result<(), &'static str> {
    if config.channels == 0 || config.sample_rate.0 == 0 {
        Err("Output device returned zero channels or sample rate")
    } else {
        Ok(())
    }
}

fn build_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    core: Arc<Mutex<Core>>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let device_failed = core
        .lock()
        .map_err(|_| "Player state poisoned")?
        .device_failed
        .clone();
    device
        .build_output_stream(
            config,
            move |output: &mut [T], info: &cpal::OutputCallbackInfo| {
                let stamp = info.timestamp();
                let latency = stamp.playback.duration_since(&stamp.callback);
                fill_output(&core, output, latency);
            },
            move |_err| device_failed.store(true, Ordering::Relaxed),
            None,
        )
        .map_err(|e| format!("Cannot open output stream: {e}"))
}

fn fill_output<T>(core: &Mutex<Core>, output: &mut [T], latency: Option<Duration>)
where
    T: SizedSample + FromSample<f32>,
{
    // Controls and decoder handovers may briefly hold the state lock. The
    // callback must return on time: contention produces silence, without
    // advancing the source or updating its underrun counter under that lock.
    if let Ok(mut core) = core.try_lock() {
        core.fill(output, latency);
    } else {
        output.fill(T::from_sample(0.0f32));
    }
}

/// Reads packets from the file, converts them to the device layout, resamples,
/// and pushes them into the ring buffer until the file ends or it is replaced.
fn decode_stream(
    path: String,
    core: Arc<Mutex<Core>>,
    generation: Arc<AtomicU64>,
    my_generation: u64,
    rate: u32,
    channels: u16,
) -> Result<(), String> {
    let mut source = audio::Stream::open(&path)?;
    let mut resampler = audio::Resampler::new(source.sample_rate, rate, channels as usize);

    let capacity = (rate as f64 * RING_SECONDS) as usize * channels as usize;
    let (mut producer, consumer) = rtrb::RingBuffer::<f32>::new(capacity);
    {
        let mut core = core.lock().map_err(|_| "Player state poisoned")?;
        core.consumer = Some(consumer);
        core.current_path = Some(path);
    }

    let mut pushed = 0usize;
    let mut skip = source.delay_frames();
    let mut at_end = false;

    'decode: loop {
        if generation.load(Ordering::Relaxed) != my_generation {
            return Ok(());
        }

        // Keep the request visible until the replacement ring is installed:
        // the callback must not consume stale samples while seeking.
        let seek = core.lock().map_err(|_| "Player state poisoned")?.seek_to;
        if let Some(seconds) = seek {
            source.seek(seconds)?;
            skip = 0;
            resampler.reset();
            let (new_producer, new_consumer) = rtrb::RingBuffer::<f32>::new(capacity);
            let mut core = core.lock().map_err(|_| "Player state poisoned")?;
            if core.seek_to == Some(seconds) {
                producer = new_producer;
                core.consumer = Some(new_consumer);
                core.position = core.base;
                core.total = None;
                core.finished = false;
                core.seek_to = None;
                pushed = 0;
                at_end = false;
            }
            continue;
        }

        if at_end {
            let next = {
                let mut guard = core.lock().map_err(|_| "Player state poisoned")?;
                if guard.seek_to.is_some() {
                    continue;
                }
                guard.take_queued_path()
            };
            if let Some(next) = next {
                source = audio::Stream::open(&next)?;
                resampler = audio::Resampler::new(source.sample_rate, rate, channels as usize);
                skip = source.delay_frames();
                let (new_producer, new_consumer) = rtrb::RingBuffer::<f32>::new(capacity);
                producer = new_producer;
                let duration = audio::duration_seconds(&next).ok().flatten();
                let mut core = core.lock().map_err(|_| "Player state poisoned")?;
                core.consumer = Some(new_consumer);
                core.duration = duration;
                // Preserve a pause or seek issued while the next file opened.
                if let Some(seconds) = core.seek_to {
                    core.request_seek(seconds)?;
                }
                pushed = 0;
                at_end = false;
            } else {
                std::thread::sleep(Duration::from_millis(5));
            }
            continue;
        }

        let Some(chunk) = source.next_chunk(&mut skip)? else {
            for sample in resampler.flush() {
                while producer.push(sample).is_err() {
                    if generation.load(Ordering::Relaxed) != my_generation {
                        return Ok(());
                    }
                    if core
                        .lock()
                        .map_err(|_| "Player state poisoned")?
                        .seek_to
                        .is_some()
                    {
                        continue 'decode;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                pushed += 1;
            }
            let mut core = core.lock().map_err(|_| "Player state poisoned")?;
            if core.seek_to.is_some() {
                continue;
            }
            core.total = Some(core.base.saturating_add(pushed));
            if pushed == 0 {
                core.finished = true;
                core.playing = false;
            }
            at_end = true;
            continue;
        };

        let converted = convert_channels(&chunk, source.channels, channels);
        for sample in resampler.process(&converted) {
            while producer.push(sample).is_err() {
                if generation.load(Ordering::Relaxed) != my_generation {
                    return Ok(());
                }
                // A paused device cannot drain a full ring. Seek must retire
                // that ring instead of waiting for resume to make room.
                if core
                    .lock()
                    .map_err(|_| "Player state poisoned")?
                    .seek_to
                    .is_some()
                {
                    continue 'decode;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            pushed += 1;
        }
    }
}

#[pyclass(unsendable)]
pub struct Player {
    output: Output,
    core: Arc<Mutex<Core>>,
    stream: Option<cpal::Stream>,
    generation: Arc<AtomicU64>,
    thread: Option<std::thread::JoinHandle<Result<(), String>>>,
    feed_producer: Option<rtrb::Producer<f32>>,
    feed_rate: u32,
}

#[pymethods]
impl Player {
    /// Opens the output device once; `play_*` calls reuse it.
    #[new]
    #[pyo3(signature = (device=None))]
    pub(crate) fn new(device: Option<String>) -> PyResult<Self> {
        let output =
            Output::open(device.as_deref()).map_err(|message| error_with("device", message))?;
        let core = Arc::new(Mutex::new(Core::new(output.channels(), output.rate())));
        let stream = output
            .build_stream(core.clone())
            .map_err(|message| error_with("device", message))?;
        stream
            .play()
            .map_err(|e| error_with("device", format!("Playback failed: {e}")))?;
        Ok(Self {
            output,
            core,
            stream: Some(stream),
            generation: Arc::new(AtomicU64::new(0)),
            thread: None,
            feed_producer: None,
            feed_rate: 0,
        })
    }

    /// Device name the player is using.
    #[getter]
    fn device(&self) -> String {
        self.output.name()
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.output.rate()
    }

    #[getter]
    fn channels(&self) -> u16 {
        self.output.channels() as u16
    }

    /// Stream a file. Returns immediately; audio is decoded as it plays.
    pub(crate) fn play_file(&mut self, _py: Python<'_>, path: &str) -> PyResult<()> {
        self.feed_producer = None;
        self.stop();
        let generation = self
            .generation
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);

        let duration = crate::audio::duration_seconds(path).ok().flatten();
        let mut core = self.lock()?;
        core.current_path = Some(path.to_string());
        core.duration = duration;
        // Container duration is advisory. Only the decoder knows the actual
        // number of output samples; using an estimate can truncate playback.
        core.total = None;
        core.base = 0;
        core.memory = None;
        core.consumer = None;
        core.position = 0;
        core.finished = false;
        core.error = None;
        core.error_code = "format";
        core.drain = Duration::ZERO;
        core.seek_to = None;
        core.underruns = 0;
        core.end_of_stream = false;
        core.playing = true;
        drop(core);

        let core = self.core.clone();
        let failure_slot = self.core.clone();
        let generation_counter = self.generation.clone();
        let rate = self.output.rate();
        let channels = self.output.channels() as u16;
        let path = path.to_string();
        let handle = std::thread::Builder::new()
            .name("sound_rs-decode".into())
            .spawn(move || {
                let result =
                    decode_stream(path, core, generation_counter, generation, rate, channels);
                if let Err(message) = &result {
                    if let Ok(mut core) = failure_slot.lock() {
                        core.error = Some(message.clone());
                        core.error_code = if message.starts_with("Cannot open") {
                            "io"
                        } else {
                            "format"
                        };
                        core.playing = false;
                    }
                }
                result
            })
            .map_err(|e| error(format!("Cannot start decoder: {e}")))?;
        self.thread = Some(handle);
        Ok(())
    }

    /// Play an in-memory array at the device format. Needs `numpy`.
    pub(crate) fn play_array(&mut self, array: Bound<'_, PyAny>, sample_rate: u32) -> PyResult<()> {
        let (samples, channels) = crate::numpy_samples(&array)?;
        self.play_samples(samples, channels, sample_rate)
    }

    pub(crate) fn play_samples(
        &mut self,
        samples: Vec<f32>,
        src_channels: u16,
        sample_rate: u32,
    ) -> PyResult<()> {
        validate_samples(&samples, src_channels, sample_rate)
            .map_err(|message| error_with("usage", message.into()))?;
        self.feed_producer = None;
        self.stop();
        let channels = self.output.channels() as u16;
        let ready = if sample_rate == self.output.rate() && src_channels == channels {
            samples
        } else {
            let folded = convert_channels(&samples, src_channels, channels);
            resample_all(&folded, sample_rate, self.output.rate(), channels as usize)
        };
        let total = ready.len();

        let mut core = self.lock()?;
        core.memory = Some(Arc::new(ready));
        core.consumer = None;
        core.position = 0;
        core.base = 0;
        core.duration = Some(total as f64 / channels as f64 / self.output.rate() as f64);
        core.total = Some(total);
        core.finished = false;
        core.error = None;
        core.error_code = "format";
        core.drain = Duration::ZERO;
        core.seek_to = None;
        core.underruns = 0;
        core.end_of_stream = false;
        core.playing = true;
        Ok(())
    }

    /// Append a path to the queue. When the current track ends, the next one
    /// starts automatically, with its own rate and channel layout.
    fn enqueue(&mut self, path: &str) -> PyResult<()> {
        if self
            .thread
            .as_ref()
            .is_none_or(|thread| thread.is_finished())
        {
            return Err(error_with(
                "usage",
                "enqueue() requires a file started with play_file()".to_string(),
            ));
        }
        self.lock()?
            .enqueue(path)
            .map_err(|message| error_with("usage", message.into()))
    }

    /// Paths waiting behind the current track.
    #[getter]
    fn queue(&self) -> PyResult<Vec<String>> {
        Ok(self.lock()?.queue.iter().cloned().collect())
    }

    fn clear_queue(&self) -> PyResult<()> {
        self.lock()?.queue.clear();
        Ok(())
    }

    /// Start a fed stream: samples arrive through `feed()` instead of a file.
    fn begin_feed(&mut self, sample_rate: u32) -> PyResult<()> {
        validate_sample_rate(sample_rate).map_err(|message| error_with("usage", message.into()))?;
        self.stop();
        let channels = self.output.channels() as u16;
        let capacity = (self.output.rate() as f64 * RING_SECONDS) as usize * channels as usize;
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(capacity);
        {
            let mut core = self.lock()?;
            core.memory = None;
            core.consumer = Some(consumer);
            core.position = 0;
            core.base = 0;
            core.total = None;
            core.duration = None;
            core.finished = false;
            core.error = None;
            core.error_code = "format";
            core.drain = Duration::ZERO;
            core.underruns = 0;
            core.end_of_stream = false;
            core.playing = true;
        }
        self.feed_producer = Some(producer);
        self.feed_rate = sample_rate;
        Ok(())
    }

    /// Push samples into a fed stream. Blocks while the ring is full.
    /// Expects float32 in `(frames,)` or `(channels, frames)`.
    pub(crate) fn feed(&mut self, py: Python<'_>, array: Bound<'_, PyAny>) -> PyResult<()> {
        let device_failed = self.lock()?.device_failed.clone();
        if self.feed_producer.is_none() {
            return Err(error_with(
                "usage",
                "Call begin_feed() before feed(), and not after stop()".to_string(),
            ));
        }
        let (samples, channels) = crate::numpy_samples(&array)?;
        validate_samples(&samples, channels, self.feed_rate)
            .map_err(|message| error_with("usage", message.into()))?;
        let device_channels = self.output.channels() as u16;
        let mut ready = convert_channels(&samples, channels, device_channels);
        if self.feed_rate != self.output.rate() {
            ready = resample_all(
                &ready,
                self.feed_rate,
                self.output.rate(),
                device_channels as usize,
            );
        }

        let producer = match &mut self.feed_producer {
            Some(producer) => producer,
            None => {
                return Err(error_with(
                    "usage",
                    "Call begin_feed() before feed(), and not after stop()".to_string(),
                ))
            }
        };

        let result =
            py.detach(|| feed_samples(producer, &ready, Duration::from_secs(5), &device_failed));
        result.map_err(|message| {
            let code = if device_failed.load(Ordering::Relaxed) {
                "device"
            } else {
                "stalled"
            };
            error_with(code, message.to_string())
        })
    }

    /// No more samples will arrive; playback finishes when the ring drains.
    fn end_feed(&mut self) -> PyResult<()> {
        if self.feed_producer.is_none() {
            // Closing an already-closed feed is harmless, but closing a file
            // stream must not turn temporary decoder starvation into EOF.
            if self.lock()?.end_of_stream {
                return Ok(());
            }
            return Err(error_with(
                "usage",
                "Call begin_feed() before end_feed()".to_string(),
            ));
        }
        self.feed_producer = None;
        let mut core = self.lock()?;
        core.end_of_stream = true;
        Ok(())
    }

    /// Pause playback; the stream stays open.
    fn pause(&self) -> PyResult<()> {
        self.lock()?.pause();
        Ok(())
    }

    /// Resume from where it was paused.
    fn resume(&self) -> PyResult<()> {
        let mut core = self.lock()?;
        if core.current_path.is_some()
            && self
                .thread
                .as_ref()
                .is_none_or(|thread| thread.is_finished())
        {
            return Err(error_with(
                "usage",
                "The decoder has stopped; call play_file() again".into(),
            ));
        }
        core.resume()
            .map_err(|message| error_with("usage", message.into()))
    }

    /// Stop and rewind to the start.
    fn stop(&mut self) {
        // Retire the decoder thread of the current track, if any.
        self.feed_producer = None;
        self.generation.fetch_add(1, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
        if let Ok(mut core) = self.core.lock() {
            core.playing = false;
            core.finished = false;
            core.paused = false;
            core.position = 0;
            core.consumer = None;
            core.memory = None;
            core.total = None;
            core.seek_to = None;
            core.duration = None;
            core.queue.clear();
            core.end_of_stream = false;
            core.drain = Duration::ZERO;
            core.current_path = None;
        }
    }

    /// Jump to `seconds`. Works for both streamed files and arrays.
    fn seek(&mut self, seconds: f64) -> PyResult<()> {
        let mut core = self.lock()?;
        if core.memory.is_none()
            && core.current_path.is_some()
            && self
                .thread
                .as_ref()
                .is_none_or(|thread| thread.is_finished())
        {
            return Err(error_with(
                "usage",
                "The decoder has stopped; call play_file() again".into(),
            ));
        }
        core.request_seek(seconds)
            .map_err(|message| error_with("usage", message.into()))
    }

    #[getter]
    fn position(&self) -> PyResult<f64> {
        Ok(self.lock()?.seconds())
    }

    #[getter]
    fn duration(&self) -> PyResult<Option<f64>> {
        Ok(self.lock()?.duration())
    }

    #[getter]
    fn is_playing(&self) -> PyResult<bool> {
        Ok(self.lock()?.is_playing())
    }

    /// Callbacks that found the buffer empty while the source was still
    /// running. A healthy stream keeps this at zero.
    #[getter]
    fn underruns(&self) -> PyResult<u64> {
        Ok(self.lock()?.underruns)
    }

    /// Last device latency reported by the backend, in milliseconds.
    #[getter]
    fn latency(&self) -> PyResult<Option<f64>> {
        Ok(self.lock()?.latency_ms)
    }

    #[getter]
    fn volume(&self) -> PyResult<f32> {
        Ok(self.lock()?.volume)
    }

    #[setter]
    pub(crate) fn set_volume(&self, volume: f32) -> PyResult<()> {
        if !volume.is_finite() {
            return Err(error_with("usage", "volume must be finite".to_string()));
        }
        self.lock()?.volume = volume.clamp(0.0, 4.0);
        Ok(())
    }

    /// Block until the current source finishes playing.
    pub(crate) fn wait(&mut self, py: Python<'_>) -> PyResult<()> {
        let core = self.core.clone();
        py.detach(move || {
            let drain = wait_loop(core);
            std::thread::sleep(drain.unwrap_or(Duration::from_millis(20)));
        });
        let mut core = self.lock()?;
        if let Some(message) = core.error.take() {
            return Err(error_with(core.error_code, message));
        }
        Ok(())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__<'py>(
        &mut self,
        _py: Python<'py>,
        _ty: Bound<'py, PyAny>,
        _v: Bound<'py, PyAny>,
        _tb: Bound<'py, PyAny>,
    ) {
        self.stop();
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        self.stop();
        if let Some(stream) = self.stream.take() {
            drop(stream);
        }
    }
}

impl Player {
    fn lock(&self) -> PyResult<std::sync::MutexGuard<'_, Core>> {
        let core = self
            .core
            .lock()
            .map_err(|_| error("Player state poisoned".to_string()))?;
        if core.device_failed.load(Ordering::Relaxed) {
            return Err(error_with(
                "device",
                "Output stream failed; create a new Player".to_string(),
            ));
        }
        Ok(core)
    }
}

fn validate_sample_rate(sample_rate: u32) -> Result<(), &'static str> {
    if sample_rate == 0 {
        Err("sample_rate must be greater than zero")
    } else {
        Ok(())
    }
}

fn validate_samples(samples: &[f32], channels: u16, sample_rate: u32) -> Result<(), &'static str> {
    validate_sample_rate(sample_rate)?;
    if channels == 0 || !samples.len().is_multiple_of(channels as usize) {
        return Err("samples must contain complete frames with at least one channel");
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("audio samples must be finite");
    }
    Ok(())
}

/// Ring backpressure is independent of Python and can be tested without a device.
fn feed_samples(
    producer: &mut rtrb::Producer<f32>,
    samples: &[f32],
    stall: Duration,
    device_failed: &AtomicBool,
) -> Result<(), &'static str> {
    for &sample in samples {
        if device_failed.load(Ordering::Relaxed) {
            return Err("Output stream failed; create a new Player");
        }
        if producer.push(sample).is_ok() {
            continue;
        }
        // Only time a full ring. A large feed can take arbitrarily long while
        // the callback is making progress; the GIL stays released throughout.
        let blocked_since = Instant::now();
        while producer.push(sample).is_err() {
            if device_failed.load(Ordering::Relaxed) {
                return Err("Output stream failed; create a new Player");
            }
            if blocked_since.elapsed() > stall {
                return Err("feed() could not drain: is playback paused?");
            }
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    Ok(())
}

/// Waits without holding the GIL, with a stall watchdog.
fn wait_loop(shared: Arc<Mutex<Core>>) -> Option<Duration> {
    {
        const STALL: Duration = Duration::from_secs(2);
        let mut last = 0;
        let mut last_change = Instant::now();
        loop {
            let (playing, finished, position, drain, error, queued) = match shared.lock() {
                Ok(mut core) => {
                    if core.device_failed.load(Ordering::Relaxed) {
                        core.playing = false;
                        return Some(Duration::ZERO);
                    }
                    if core.paused {
                        return Some(core.drain);
                    }
                    (
                        core.playing,
                        core.finished,
                        core.position,
                        core.drain,
                        core.error.is_some(),
                        !core.queue.is_empty(),
                    )
                }
                Err(_) => return None,
            };
            if error {
                return Some(Duration::ZERO);
            }
            // A paused track returns with its queue intact. A completed track
            // with queued work waits for the worker's atomic handover.
            if ((!playing || finished) && !queued) || (!playing && !finished) {
                return Some(drain);
            }
            if position != last {
                last = position;
                last_change = Instant::now();
            } else if last_change.elapsed() > STALL {
                if let Ok(mut core) = shared.lock() {
                    core.error_code = "stalled";
                    core.playing = false;
                    core.error = Some(
                        "Playback stalled for 2s: the output device stopped responding".to_string(),
                    );
                }
                return Some(Duration::ZERO);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

impl core::fmt::Debug for Player {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Player")
            .field("device", &self.output.name())
            .field("channels", &self.output.channels())
            .field("rate", &self.output.rate())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pause_between_completion_and_queue_handover_retains_the_queue() {
        let mut core = Core::new(1, 1000);
        core.current_path = Some("first.wav".into());
        let (mut producer, consumer) = rtrb::RingBuffer::<f32>::new(1);
        producer.push(0.5).unwrap();
        core.consumer = Some(consumer);
        core.total = Some(1);
        core.playing = true;
        core.enqueue("second.wav").unwrap();

        // These steps run under the same state lock in production. Pause
        // wins the race before the worker can select the next queued file.
        core.fill(&mut [0.0f32; 1], Some(Duration::ZERO));
        assert!(core.finished);
        core.pause();
        assert_eq!(core.take_queued_path(), None);
        assert!(!core.playing);
        assert_eq!(core.current_path.as_deref(), Some("first.wav"));
        assert_eq!(core.queue.front().map(String::as_str), Some("second.wav"));
    }

    #[test]
    fn seeking_an_explicitly_paused_completed_file_keeps_it_paused() {
        let mut core = Core::new(1, 1000);
        core.current_path = Some("current.wav".into());
        core.finished = true;
        core.pause();
        core.request_seek(0.0).unwrap();
        assert!(!core.playing);
        assert_eq!(core.seek_to, Some(0.0));
    }

    #[test]
    fn wait_on_a_paused_completed_queue_returns_without_a_stall() {
        let mut core = Core::new(1, 1000);
        core.current_path = Some("first.wav".into());
        core.finished = true;
        core.enqueue("second.wav").unwrap();
        core.pause();
        let shared = Arc::new(Mutex::new(core));
        assert_eq!(wait_loop(shared.clone()), Some(Duration::ZERO));
        let mut core = shared.lock().unwrap();
        assert!(core.error.is_none());
        assert_eq!(core.queue.len(), 1);
        core.resume().unwrap();
        assert!(!core.paused);
        assert!(core.playing);
        assert_eq!(core.seek_to, Some(0.0));
    }

    #[test]
    fn output_callback_returns_silence_without_waiting_for_the_control_lock() {
        let mut core = Core::new(1, 1000);
        core.memory = Some(Arc::new(vec![0.5; 8]));
        core.total = Some(8);
        core.playing = true;
        let shared = Arc::new(Mutex::new(core));
        let held = shared.lock().unwrap();
        let callback_core = shared.clone();
        let (send, receive) = std::sync::mpsc::channel();
        let callback = std::thread::spawn(move || {
            let mut output = [1.0f32; 8];
            fill_output(&callback_core, &mut output, Some(Duration::ZERO));
            send.send(output).unwrap();
        });
        let output = receive.recv_timeout(Duration::from_millis(250));
        // Release before assertions so a regression cannot strand the test
        // thread in a blocking lock or make the process hang during cleanup.
        drop(held);
        callback.join().unwrap();
        assert_eq!(output.unwrap(), [0.0; 8]);
        assert_eq!(shared.lock().unwrap().position, 0);

        let mut output = [0.0f32; 8];
        fill_output(&shared, &mut output, Some(Duration::ZERO));
        assert_eq!(output, [0.5; 8]);
        assert_eq!(shared.lock().unwrap().position, 8);
    }

    #[test]
    fn output_configuration_rejects_zero_channels_and_rates() {
        for (channels, rate) in [(0, 48_000), (2, 0), (0, 0)] {
            let config = StreamConfig {
                channels,
                sample_rate: cpal::SampleRate(rate),
                buffer_size: cpal::BufferSize::Default,
            };
            assert!(validate_output_config(&config).is_err());
        }
        let config = StreamConfig {
            channels: 2,
            sample_rate: cpal::SampleRate(48_000),
            buffer_size: cpal::BufferSize::Default,
        };
        assert!(validate_output_config(&config).is_ok());
    }

    struct TestDecoder {
        core: Arc<Mutex<Core>>,
        generation: Arc<AtomicU64>,
        thread: Option<std::thread::JoinHandle<Result<(), String>>>,
    }

    impl TestDecoder {
        fn start(path: &str, mut core: Core) -> Self {
            core.current_path = Some(path.to_string());
            let rate = core.rate;
            let channels = core.channels as u16;
            let core = Arc::new(Mutex::new(core));
            let generation = Arc::new(AtomicU64::new(0));
            let worker_core = core.clone();
            let worker_generation = generation.clone();
            let path = path.to_string();
            let thread = std::thread::spawn(move || {
                decode_stream(path, worker_core, worker_generation, 0, rate, channels)
            });
            Self {
                core,
                generation,
                thread: Some(thread),
            }
        }

        fn poll(&self, mut condition: impl FnMut(&mut Core) -> bool) -> bool {
            let started = Instant::now();
            while started.elapsed() < Duration::from_secs(2) {
                if condition(&mut self.core.lock().unwrap()) {
                    return true;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            false
        }
    }

    impl Drop for TestDecoder {
        fn drop(&mut self) {
            self.generation.fetch_add(1, Ordering::Relaxed);
            let result = self.thread.take().unwrap().join();
            if !std::thread::panicking() {
                assert_eq!(result.unwrap(), Ok(()));
            }
        }
    }

    #[test]
    fn a_paused_decoder_processes_seek_even_when_its_ring_is_full() {
        let path =
            std::env::temp_dir().join(format!("sound_rs_paused_seek_{}.wav", std::process::id()));
        let samples: Vec<_> = (0..3000).map(|i| i as f32 / 4000.0).collect();
        crate::writer::write(
            path.to_str().unwrap(),
            &samples,
            1,
            1000,
            crate::writer::Subtype::Float32,
        )
        .unwrap();
        let mut core = Core::new(1, 1000);
        core.duration = Some(3.0);
        core.queue.push_back("next.wav".into());
        let worker = TestDecoder::start(path.to_str().unwrap(), core);
        assert!(worker.poll(|core| core
            .consumer
            .as_ref()
            .is_some_and(|ring| ring.slots() == 1000)));
        worker.core.lock().unwrap().request_seek(1.5).unwrap();
        assert!(worker.poll(|core| core.seek_to.is_none()
            && core.consumer.as_ref().is_some_and(|ring| ring.slots() > 0)));
        {
            let mut core = worker.core.lock().unwrap();
            assert!(!core.playing);
            assert_eq!(core.position, 1500);
            assert_eq!(core.queue.front().map(String::as_str), Some("next.wav"));
            assert_eq!(
                *core.consumer.as_ref().unwrap().peek().unwrap(),
                samples[1500]
            );
            core.resume().unwrap();
            let mut output = [0.0f32; 8];
            core.fill(&mut output, Some(Duration::ZERO));
            assert_eq!(&output, &samples[1500..1508]);
            assert_eq!(core.position, 1508);
        }
        drop(worker);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn seek_after_queue_completion_uses_the_current_file() {
        let path_a =
            std::env::temp_dir().join(format!("sound_rs_queue_a_{}.wav", std::process::id()));
        let path_b =
            std::env::temp_dir().join(format!("sound_rs_queue_b_{}.wav", std::process::id()));
        for (path, value) in [(&path_a, 0.25), (&path_b, 0.75)] {
            crate::writer::write(
                path.to_str().unwrap(),
                &[value; 100],
                1,
                1000,
                crate::writer::Subtype::Float32,
            )
            .unwrap();
        }
        let mut core = Core::new(1, 1000);
        core.playing = true;
        core.queue.push_back(path_b.to_str().unwrap().to_string());
        let worker = TestDecoder::start(path_a.to_str().unwrap(), core);
        let mut played = Vec::new();
        assert!(worker.poll(|core| {
            let mut output = [0.0f32; 64];
            core.fill(&mut output, Some(Duration::ZERO));
            played.extend(output);
            core.current_path.as_deref() == path_b.to_str() && core.finished
        }));
        assert_eq!(played.iter().filter(|sample| **sample == 0.25).count(), 100);
        assert_eq!(played.iter().filter(|sample| **sample == 0.75).count(), 100);
        worker.core.lock().unwrap().request_seek(0.05).unwrap();
        assert!(worker.poll(|core| core.seek_to.is_none()
            && core.consumer.as_ref().is_some_and(|ring| ring.slots() > 0)));
        {
            let mut core = worker.core.lock().unwrap();
            let mut output = [0.0f32; 1];
            core.fill(&mut output, Some(Duration::ZERO));
            assert_eq!(output, [0.75]);
            assert_eq!(core.position, 51);
            assert!(core.queue.is_empty());
        }
        drop(worker);
        std::fs::remove_file(path_a).unwrap();
        std::fs::remove_file(path_b).unwrap();
    }

    #[test]
    fn audio_validation_rejects_zero_rates_incomplete_frames_and_nonfinite_samples() {
        assert!(validate_sample_rate(0).is_err());
        assert!(validate_samples(&[], 0, 48_000).is_err());
        assert!(validate_samples(&[0.5], 2, 48_000).is_err());
        assert!(validate_samples(&[0.5], 1, 0).is_err());
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(validate_samples(&[bad], 1, 48_000).is_err());
        }
        assert!(validate_samples(&[], 2, 48_000).is_ok());
        assert!(validate_samples(&[f32::MAX, -f32::MAX], 2, 48_000).is_ok());
    }

    #[test]
    fn device_failure_silences_output_and_wakes_waiters() {
        let mut core = Core::new(1, 1000);
        core.memory = Some(Arc::new(vec![0.5; 8]));
        core.playing = true;
        core.device_failed.store(true, Ordering::Relaxed);
        let mut output = [1.0f32; 8];
        core.fill(&mut output, None);
        assert_eq!(output, [0.0; 8]);
        assert!(!core.is_playing());
        assert_eq!(core.position, 0);
        let shared = Arc::new(Mutex::new(core));
        assert_eq!(wait_loop(shared), Some(Duration::ZERO));
    }

    #[test]
    fn device_failure_interrupts_backpressure_without_waiting_for_the_watchdog() {
        let (mut producer, _consumer) = rtrb::RingBuffer::<f32>::new(1);
        let failed = Arc::new(AtomicBool::new(false));
        let signal = failed.clone();
        let callback = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            signal.store(true, Ordering::Relaxed);
        });
        assert_eq!(
            feed_samples(&mut producer, &[0.5, 0.5], Duration::from_secs(5), &failed),
            Err("Output stream failed; create a new Player")
        );
        callback.join().unwrap();
    }

    #[test]
    fn resume_restarts_completed_files_but_rejects_finished_feeds() {
        let mut core = Core::new(1, 1000);
        assert!(core.resume().is_err());
        core.current_path = Some("current.wav".into());
        core.finished = true;
        core.position = 100;
        core.resume().unwrap();
        assert_eq!(core.seek_to, Some(0.0));
        assert_eq!(core.position, 0);
        assert!(core.playing);
        core.finished = true;
        core.end_of_stream = true;
        assert!(core.resume().is_err());
    }

    #[test]
    fn invalid_seeks_leave_playback_state_unchanged() {
        for seconds in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
            let mut core = Core::new(2, 48_000);
            core.current_path = Some("track.wav".to_string());
            core.position = 100;
            core.base = 100;
            core.playing = true;
            assert!(core.request_seek(seconds).is_err());
            assert_eq!(core.position, 100);
            assert_eq!(core.base, 100);
            assert_eq!(core.seek_to, None);
            assert!(core.playing);
        }
    }

    #[test]
    fn decoder_seek_handles_saturating_timestamps_without_panicking() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tone.m4a");
        let mut source = audio::Stream::open(path.to_str().unwrap()).unwrap();
        assert!(source.delay > 0, "fixture must exercise priming addition");
        source.seek(f64::MAX).unwrap();
    }

    #[test]
    fn a_paused_queue_does_not_trigger_the_stall_watchdog() {
        let mut core = Core::new(1, 1000);
        core.current_path = Some("current.wav".into());
        core.enqueue("next.wav").unwrap();
        let shared = Arc::new(Mutex::new(core));
        assert_eq!(wait_loop(shared.clone()), Some(Duration::ZERO));
        let core = shared.lock().unwrap();
        assert!(core.error.is_none());
        assert_eq!(core.queue.len(), 1);
    }

    #[test]
    fn paused_seek_preserves_pause_and_queue() {
        let mut core = Core::new(2, 48_000);
        let (_producer, consumer) = rtrb::RingBuffer::<f32>::new(8);
        core.consumer = Some(consumer);
        core.current_path = Some("track.wav".to_string());
        core.queue.push_back("next.wav".to_string());
        core.request_seek(1.0).unwrap();
        assert!(!core.playing);
        assert_eq!(core.seek_to, Some(1.0));
        assert_eq!(core.position, 96_000);
        assert_eq!(core.queue.front().map(String::as_str), Some("next.wav"));
    }

    #[test]
    fn pending_seek_silences_stale_ring_samples() {
        let mut core = Core::new(1, 1000);
        let (mut producer, consumer) = rtrb::RingBuffer::<f32>::new(8);
        producer.push(0.5).unwrap();
        core.consumer = Some(consumer);
        core.current_path = Some("track.wav".to_string());
        core.playing = true;
        core.request_seek(1.0).unwrap();
        let mut out = [1.0f32; 8];
        core.fill(&mut out, None);
        assert_eq!(out, [0.0; 8]);
        assert_eq!(core.position, 1000);
        assert_eq!(core.consumer.as_ref().unwrap().slots(), 1);
        assert_eq!(core.underruns, 0);
    }

    #[test]
    fn seek_and_enqueue_reject_sources_without_a_decoder() {
        for feed in [false, true] {
            let mut core = Core::new(1, 1000);
            if feed {
                let (_producer, consumer) = rtrb::RingBuffer::<f32>::new(8);
                core.consumer = Some(consumer);
            }
            assert!(core.request_seek(0.1).is_err());
            assert!(core.enqueue("next.wav").is_err());
            assert!(core.queue.is_empty());
        }
        let mut core = Core::new(1, 1000);
        core.memory = Some(Arc::new(vec![0.5; 100]));
        core.total = Some(100);
        assert!(core.enqueue("next.wav").is_err());
        assert!(core.request_seek(0.05).is_ok());
        assert_eq!(core.position, 50);
    }

    #[test]
    fn memory_seek_clamps_and_restarts_only_finished_playback() {
        let mut core = Core::new(2, 1000);
        core.memory = Some(Arc::new(vec![0.5; 200]));
        core.total = Some(200);
        core.request_seek(0.05).unwrap();
        assert_eq!(core.position, 100);
        assert!(!core.playing);
        core.request_seek(10.0).unwrap();
        assert_eq!(core.position, 200);
        core.finished = true;
        core.request_seek(-1.0).unwrap();
        assert_eq!(core.position, 0);
        assert!(core.playing && !core.finished);
    }

    #[test]
    fn drain_waits_for_the_written_prefix_not_the_silent_tail() {
        for samples in [2, 8] {
            let mut core = Core::new(1, 1000);
            core.memory = Some(Arc::new(vec![0.5; samples]));
            core.total = Some(samples);
            core.playing = true;
            core.fill(&mut [0.0f32; 8], Some(Duration::from_millis(10)));
            assert_eq!(core.drain, Duration::from_millis(10 + samples as u64));
        }
    }

    #[test]
    fn empty_sources_finish_without_a_drain_delay() {
        for fed in [false, true] {
            let mut core = Core::new(1, 1000);
            if fed {
                let (producer, consumer) = rtrb::RingBuffer::<f32>::new(8);
                drop(producer);
                core.consumer = Some(consumer);
                core.end_of_stream = true;
            } else {
                core.memory = Some(Arc::new(Vec::new()));
                core.total = Some(0);
            }
            core.playing = true;
            let mut output = [1.0f32; 8];
            core.fill(&mut output, Some(Duration::from_millis(10)));
            assert!(core.finished && !core.playing);
            assert_eq!(core.drain, Duration::ZERO);
            assert_eq!(output, [0.0; 8]);
            assert_eq!(core.underruns, 0);
        }
    }

    #[test]
    fn closed_feed_finishes_on_its_last_partial_or_full_callback() {
        for samples in [2, 8] {
            let mut core = Core::new(1, 1000);
            let (mut producer, consumer) = rtrb::RingBuffer::<f32>::new(8);
            for _ in 0..samples {
                producer.push(0.5).unwrap();
            }
            drop(producer);
            core.consumer = Some(consumer);
            core.end_of_stream = true;
            core.playing = true;
            let mut output = [1.0f32; 8];
            core.fill(&mut output, Some(Duration::from_millis(10)));
            assert!(core.finished && !core.playing);
            assert_eq!(core.position, samples);
            assert_eq!(core.drain, Duration::from_millis(10 + samples as u64));
            assert_eq!(core.underruns, 0);
            assert!(output[..samples].iter().all(|sample| *sample == 0.5));
            assert!(output[samples..].iter().all(|sample| *sample == 0.0));
        }
    }

    #[test]
    fn closing_a_feed_does_not_discard_buffered_samples() {
        let mut core = Core::new(1, 1000);
        let (mut producer, consumer) = rtrb::RingBuffer::<f32>::new(10);
        for _ in 0..10 {
            producer.push(0.5).unwrap();
        }
        drop(producer);
        core.consumer = Some(consumer);
        core.end_of_stream = true;
        core.playing = true;
        core.fill(&mut [0.0f32; 8], None);
        assert!(core.playing && !core.finished);
        core.fill(&mut [0.0f32; 8], None);
        assert!(core.finished && !core.playing);
        assert_eq!(core.position, 10);
        assert_eq!(core.underruns, 0);
    }

    #[test]
    fn watchdog_reports_stalled_code_and_stops_playback() {
        let shared = Arc::new(Mutex::new(Core::new(1, 1000)));
        shared.lock().unwrap().playing = true;
        assert_eq!(wait_loop(shared.clone()), Some(Duration::ZERO));
        let core = shared.lock().unwrap();
        assert_eq!(core.error_code, "stalled");
        assert!(core.error.is_some());
        assert!(!core.playing);
    }

    #[test]
    fn feed_timeout_measures_stalls_instead_of_total_playback_time() {
        let (mut producer, mut consumer) = rtrb::RingBuffer::<f32>::new(1);
        let reader = std::thread::spawn(move || {
            let mut received = Vec::new();
            loop {
                match consumer.pop() {
                    Ok(sample) => {
                        received.push(sample);
                        std::thread::sleep(Duration::from_millis(20));
                    }
                    Err(_) if consumer.is_abandoned() => return received,
                    Err(_) => std::thread::sleep(Duration::from_millis(1)),
                }
            }
        });
        let samples: Vec<_> = (0..30).map(|value| value as f32).collect();
        let result = feed_samples(
            &mut producer,
            &samples,
            Duration::from_millis(200),
            &AtomicBool::new(false),
        );
        drop(producer);
        let received = reader.join().unwrap();
        assert_eq!(result, Ok(()));
        assert_eq!(received, samples);
    }

    #[test]
    fn feed_times_out_when_the_consumer_does_not_advance() {
        let (mut producer, _consumer) = rtrb::RingBuffer::<f32>::new(1);
        let result = feed_samples(
            &mut producer,
            &[1.0, 2.0],
            Duration::from_millis(20),
            &AtomicBool::new(false),
        );
        assert!(result.is_err());
    }

    #[test]
    fn memory_source_plays_volume_and_stops_at_the_end() {
        let mut core = Core::new(2, 48_000);
        core.memory = Some(Arc::new(vec![1.0f32, -1.0, 1.0, -1.0]));
        core.total = Some(4);
        core.volume = 0.5;
        core.playing = true;

        let mut out = [9.0f32; 8];
        core.fill(&mut out, None);
        assert_eq!(&out[..4], &[0.5, -0.5, 0.5, -0.5]);
        assert!(out[4..].iter().all(|&s| s == 0.0), "tail must be silence");
        assert_eq!(core.position, 4);
        assert!(core.finished && !core.playing);
    }

    #[test]
    fn memory_source_resumes_where_it_left_off() {
        let mut core = Core::new(1, 1000);
        core.memory = Some(Arc::new((0..100).map(|i| i as f32).collect()));
        core.total = Some(100);
        core.playing = true;

        let mut out = [0.0f32; 10];
        core.fill(&mut out, None);
        assert_eq!(
            &out[..10],
            &(0..10).map(|i| i as f32).collect::<Vec<_>>()[..]
        );

        core.playing = false; // paused
        core.fill(&mut out, None);
        assert!(out.iter().all(|&s| s == 0.0), "paused playback is silent");

        core.playing = true;
        core.fill(&mut out, None);
        assert_eq!(out[0], 10.0, "resume continues from the paused position");
    }

    #[test]
    fn underruns_are_counted_only_while_playing() {
        let mut core = Core::new(1, 48_000);
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(4);
        drop(producer);
        core.consumer = Some(consumer);

        core.memory = Some(Arc::new(vec![0.5f32, 0.5]));
        core.total = None;
        core.playing = true;

        let mut out = [0.0f32; 4];
        core.fill(&mut out, Some(Duration::from_millis(12))); // two samples, then silence
        assert_eq!(
            core.underruns, 1,
            "starving after audio started is an underrun"
        );

        core.memory = None;
        core.fill(&mut out, Some(Duration::from_millis(12)));
        assert_eq!(core.underruns, 2);
        assert_eq!(core.latency_ms, Some(12.0));

        core.playing = false;
        core.fill(&mut out, None);
        assert_eq!(core.underruns, 2, "paused silence is not an underrun");
    }

    #[test]
    fn pause_freezes_the_position() {
        let mut core = Core::new(1, 48_000);
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(4);
        drop(producer);
        core.consumer = Some(consumer);
        core.playing = false;

        let mut out = [9.0f32; 4];
        core.fill(&mut out, None);
        assert!(out.iter().all(|&s| s == 0.0));
        assert_eq!(core.position, 0);
    }
    #[test]
    fn decoder_respects_a_pause_after_the_final_callback() {
        let path_a =
            std::env::temp_dir().join(format!("review_pause_a_{}.wav", std::process::id()));
        let path_b =
            std::env::temp_dir().join(format!("review_pause_b_{}.wav", std::process::id()));
        for (path, value) in [(&path_a, 0.25), (&path_b, 0.75)] {
            crate::writer::write(
                path.to_str().unwrap(),
                &[value; 100],
                1,
                1000,
                crate::writer::Subtype::Float32,
            )
            .unwrap();
        }
        let mut core = Core::new(1, 1000);
        core.playing = true;
        core.queue.push_back(path_b.to_str().unwrap().to_string());
        let worker = TestDecoder::start(path_a.to_str().unwrap(), core);
        assert!(worker.poll(|core| core.total == Some(100)));
        {
            let mut core = worker.core.lock().unwrap();
            core.fill(&mut [0.0f32; 100], Some(Duration::ZERO));
            assert!(core.finished);
            core.pause();
        }
        let handed_over = worker.poll(|core| core.current_path.as_deref() == path_b.to_str());
        drop(worker);
        std::fs::remove_file(path_a).unwrap();
        std::fs::remove_file(path_b).unwrap();
        assert!(
            !handed_over,
            "explicit pause after final callback must preserve queue"
        );
    }
}
