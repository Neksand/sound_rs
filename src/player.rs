//! Persistent audio output: one stream stays open, buffers are fed to it.
//!
//! Two sources feed the same callback: an in-memory buffer (already at the
//! device rate) and a decoder thread that streams a file through a ring
//! buffer, so a two-hour file costs a few megabytes instead of gigabytes.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, StreamConfig};
use pyo3::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::audio::{self, convert_channels, resample_all};
use crate::error;

/// How much decoded audio the decoder thread may run ahead of the device.
const RING_SECONDS: f64 = 1.0;

pub struct Core {
    pub channels: usize,
    pub rate: u32,
    pub volume: f32,
    pub playing: bool,
    pub finished: bool,
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
    /// Seek requested by Python, in seconds.
    pub seek_to: Option<f64>,
    /// Seconds of audio still in flight when the last sample was written.
    pub drain: Duration,
}

impl Core {
    pub fn new(channels: usize, rate: u32) -> Self {
        Self {
            channels,
            rate,
            volume: 1.0,
            playing: false,
            finished: false,
            position: 0,
            total: None,
            base: 0,
            duration: None,
            memory: None,
            consumer: None,
            error: None,
            seek_to: None,
            drain: Duration::ZERO,
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
        self.playing && !self.finished
    }

    /// Called from the audio callback. Writes exactly `output.len()` samples:
    /// real audio while there is some, silence otherwise.
    pub fn fill<T>(
        &mut self,
        output: &mut [T],
        latency: Option<Duration>,
    ) where
        T: SizedSample + FromSample<f32>,
    {
        let silence = |output: &mut [T]| {
            for slot in output.iter_mut() {
                *slot = T::from_sample(0.0f32);
            }
        };

        if !self.playing {
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

        // The source is done once everything it will ever produce was consumed.
        if let Some(total) = self.total {
            if self.position >= total {
                self.finished = true;
                self.playing = false;
                self.drain = latency.unwrap_or(Duration::from_millis(100))
                    + Duration::from_secs_f64(
                        (output.len() - written) as f64
                            / self.channels.max(1) as f64
                            / self.rate as f64,
                    );
            }
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

        Ok(Self {
            config: StreamConfig {
                channels: supported.channels(),
                sample_rate: supported.sample_rate(),
                buffer_size: cpal::BufferSize::Default,
            },
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

fn build_stream<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    core: Arc<Mutex<Core>>,
) -> Result<cpal::Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    device
        .build_output_stream(
            config,
            move |output: &mut [T], info: &cpal::OutputCallbackInfo| {
                let stamp = info.timestamp();
                let latency = stamp.playback.duration_since(&stamp.callback);
                if let Ok(mut core) = core.lock() {
                    core.fill(output, latency);
                }
            },
            move |err| eprintln!("Audio stream error: {err}"),
            None,
        )
        .map_err(|e| format!("Cannot open output stream: {e}"))
}

/// Reads packets from the file, converts them to the device layout, resamples,
/// and pushes them into the ring buffer until the file ends or it is replaced.
#[allow(clippy::too_many_arguments)]
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
    }

    let mut pushed = 0usize;
    let mut skip = source.delay_frames();

    loop {
        // Retired by a newer track, a stop, or a drop of the Player.
        if generation.load(Ordering::Relaxed) != my_generation {
            return Ok(());
        }

        // Controls are polled once per packet; packets are milliseconds apart.
        let seek = {
            let mut core = core.lock().map_err(|_| "Player state poisoned")?;
            core.seek_to.take()
        };
        if let Some(seconds) = seek {
            skip = source.seek(seconds)?;
            resampler.reset();
            let (new_producer, new_consumer) = rtrb::RingBuffer::<f32>::new(capacity);
            producer = new_producer;
            {
                let mut core = core.lock().map_err(|_| "Player state poisoned")?;
                core.consumer = Some(new_consumer);
                core.position = core.base;
                core.total = None;
                core.finished = false;
            }
            pushed = 0;
            continue;
        }

        let Some(chunk) = source.next_chunk(&mut skip)? else {
            // Drain the resampler's internal buffer, then mark the end.
            for sample in resampler.flush() {
                while producer.push(sample).is_err() {
                    if generation.load(Ordering::Relaxed) != my_generation {
                        return Ok(());
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                pushed += 1;
            }
            {
                let mut core = core.lock().map_err(|_| "Player state poisoned")?;
                core.total = Some(core.base + pushed);
                if pushed == 0 && core.base == 0 {
                    core.finished = true;
                    core.playing = false;
                }
            }

            // Stay parked at the end so a later seek can resume. The thread
            // exits when the track is replaced or the player is dropped.
            loop {
                if generation.load(Ordering::Relaxed) != my_generation {
                    return Ok(());
                }
                let seek = core
                    .lock()
                    .map_err(|_| "Player state poisoned")?
                    .seek_to
                    .take();
                match seek {
                    Some(seconds) => {
                        skip = source.seek(seconds)?;
                        resampler.reset();
                        let (new_producer, new_consumer) = rtrb::RingBuffer::<f32>::new(capacity);
                        producer = new_producer;
                        let mut core = core.lock().map_err(|_| "Player state poisoned")?;
                        core.consumer = Some(new_consumer);
                        // Audio that was still in the old ring must not count
                        // towards the new play position.
                        core.position = core.base;
                        core.total = None;
                        core.finished = false;
                        pushed = 0;
                        break;
                    }
                    None => std::thread::sleep(Duration::from_millis(5)),
                }
            }
            continue;
        };

        let converted = convert_channels(&chunk, source.channels, channels);
        for sample in resampler.process(&converted) {
            while producer.push(sample).is_err() {
                if generation.load(Ordering::Relaxed) != my_generation {
                    return Ok(());
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
}

#[pymethods]
impl Player {
    /// Opens the output device once; `play_*` calls reuse it.
    #[new]
    #[pyo3(signature = (device=None))]
    pub(crate) fn new(device: Option<String>) -> PyResult<Self> {
        let output = Output::open(device.as_deref()).map_err(error)?;
        let core = Arc::new(Mutex::new(Core::new(output.channels(), output.rate())));
        let stream = output.build_stream(core.clone()).map_err(error)?;
        stream.play().map_err(|e| error(format!("Playback failed: {e}")))?;
        Ok(Self {
            output,
            core,
            stream: Some(stream),
            generation: Arc::new(AtomicU64::new(0)),
            thread: None,
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
    pub(crate) fn play_file(&mut self, py: Python<'_>, path: &str) -> PyResult<()> {
        let _ = py.allow_threads(|| ());
        self.stop();
        let generation = self
            .generation
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);

        let duration = crate::audio::duration_seconds(path).ok().flatten();
        let mut core = self.lock()?;
        core.duration = duration;
        core.base = 0;
        core.memory = None;
        core.consumer = None;
        core.position = 0;
        core.total = None;
        core.finished = false;
        core.error = None;
        core.seek_to = None;
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
                let result = decode_stream(path, core, generation_counter, generation, rate, channels);
                if let Err(message) = &result {
                    if let Ok(mut core) = failure_slot.lock() {
                        core.error = Some(message.clone());
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
    pub(crate) fn play_array(&mut self, py: Python<'_>, array: Bound<'_, PyAny>, sample_rate: u32) -> PyResult<()> {
        let (samples, channels) = crate::numpy_samples(py, &array, sample_rate)?;
        self.play_samples(samples, channels, sample_rate)
    }

    pub(crate) fn play_samples(
        &mut self,
        samples: Vec<f32>,
        src_channels: u16,
        sample_rate: u32,
    ) -> PyResult<()> {
        self.stop();
        let channels = self.output.channels() as u16;
        let ready = if sample_rate == self.output.rate() && src_channels == channels {
            samples
        } else {
            let folded = convert_channels(&samples, src_channels, channels);
            resample_all(
                &folded,
                sample_rate.max(1),
                self.output.rate(),
                channels as usize,
            )
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
        core.seek_to = None;
        core.playing = true;
        Ok(())
    }

    /// Pause playback; the stream stays open.
    fn pause(&self) -> PyResult<()> {
        self.lock()?.playing = false;
        Ok(())
    }

    /// Resume from where it was paused.
    fn resume(&self) -> PyResult<()> {
        let mut core = self.lock()?;
        if core.finished {
            core.position = 0;
            core.finished = false;
        }
        core.playing = true;
        Ok(())
    }

    /// Stop and rewind to the start.
    fn stop(&mut self) {
        // Retire the decoder thread of the current track, if any.
        self.generation.fetch_add(1, Ordering::Relaxed);
        if let Some(handle) = self.thread.take() {
            let _ = handle.join();
        }
        if let Ok(mut core) = self.core.lock() {
            core.playing = false;
            core.finished = false;
            core.position = 0;
            core.consumer = None;
            core.memory = None;
            core.total = None;
            core.seek_to = None;
        }
    }

    /// Jump to `seconds`. Works for both streamed files and arrays.
    fn seek(&self, seconds: f64) -> PyResult<()> {
        let seconds = seconds.max(0.0);
        let mut core = self.lock()?;
        let ch = core.channels;
        let rate = core.rate;
        let target = (seconds * rate as f64) as usize * ch;
        core.finished = false;
        if core.memory.is_some() {
            core.position = target.min(core.total.unwrap_or(usize::MAX));
            core.base = core.position;
        } else {
            core.seek_to = Some(seconds);
            core.base = target;
            core.position = target;
            core.total = None;
        }
        Ok(())
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

    #[getter]
    fn volume(&self) -> PyResult<f32> {
        Ok(self.lock()?.volume)
    }

    #[setter]
    fn set_volume(&self, volume: f32) -> PyResult<()> {
        self.lock()?.volume = volume.clamp(0.0, 4.0);
        Ok(())
    }

    /// Block until the current source finishes playing.
    pub(crate) fn wait(&mut self, py: Python<'_>) -> PyResult<()> {
        let core = self.core.clone();
        let drain = py.allow_threads(move || wait_loop(core));
        if let Some(message) = self.lock().map(|mut c| c.error.take()).ok().flatten() {
            return Err(error(message));
        }
        std::thread::sleep(drain.unwrap_or(Duration::from_millis(20)));
        Ok(())
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__<'py>(&mut self, _py: Python<'py>, _ty: Bound<'py, PyAny>, _v: Bound<'py, PyAny>, _tb: Bound<'py, PyAny>) {
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
        self.core
            .lock()
            .map_err(|_| error("Player state poisoned".to_string()))
    }
}

/// Waits without holding the GIL, with a stall watchdog.
fn wait_loop(shared: Arc<Mutex<Core>>) -> Option<Duration> {
    {
        const STALL: Duration = Duration::from_secs(2);
        let mut last = 0;
        let mut last_change = Instant::now();
        loop {
            let (playing, finished, position, drain, error) = match shared.lock() {
                Ok(core) => (
                    core.playing,
                    core.finished,
                    core.position,
                    core.drain,
                    core.error.is_some(),
                ),
                Err(_) => return None,
            };
            if error {
                return Some(Duration::ZERO);
            }
            if !playing || finished {
                return Some(drain);
            }
            if position != last {
                last = position;
                last_change = Instant::now();
            } else if last_change.elapsed() > STALL {
                if let Ok(mut core) = shared.lock() {
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
        assert_eq!(&out[..10], &(0..10).map(|i| i as f32).collect::<Vec<_>>()[..]);

        core.playing = false; // paused
        core.fill(&mut out, None);
        assert!(out.iter().all(|&s| s == 0.0), "paused playback is silent");

        core.playing = true;
        core.fill(&mut out, None);
        assert_eq!(out[0], 10.0, "resume continues from the paused position");
    }

    #[test]
    fn paused_stream_is_silent_and_does_not_advance() {
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
}
