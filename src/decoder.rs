//! Incremental decoding for files larger than memory.
//!
//! `Decoder` keeps one container reader and decoder alive, returns fixed-size
//! blocks, and optionally folds to mono or resamples with retained filter
//! state. It is deliberately separate from `Player`, which adds a device and a
//! real-time ring.

use pyo3::prelude::*;
use std::path::PathBuf;

use crate::audio::{self, convert_channels, Resampler};

struct DecoderCore {
    stream: audio::Stream,
    skip: u32,
    mono: bool,
    target_rate: u32,
    source_channels: u16,
    target_channels: u16,
    resampler: Option<Resampler>,
    pending: Vec<f32>,
    eof: bool,
    duration: Option<f64>,
}

impl DecoderCore {
    fn open(path: &str, sample_rate: Option<u32>, mono: bool) -> Result<Self, String> {
        let stream = audio::Stream::open(path)?;
        let source_channels = stream.channels;
        let target_channels = if mono { 1 } else { source_channels };
        let target_rate = sample_rate.unwrap_or(stream.sample_rate);
        let skip = stream.delay_frames();
        let duration = audio::duration_seconds(path).ok().flatten();
        let mut core = Self {
            stream,
            skip,
            mono,
            target_rate,
            source_channels,
            target_channels,
            resampler: None,
            pending: Vec::new(),
            eof: false,
            duration,
        };
        if core.source_channels > 0 {
            core.ensure_resampler()?;
        } else {
            core.prime()?;
        }
        Ok(core)
    }

    fn ensure_resampler(&mut self) -> Result<(), String> {
        if self.resampler.is_some() || self.target_rate == self.stream.sample_rate {
            return Ok(());
        }
        let resampler = Resampler::new(
            self.stream.sample_rate,
            self.target_rate,
            self.target_channels as usize,
        );
        if !resampler.is_active() {
            return Err("Cannot create resampler for this decoder".to_string());
        }
        self.resampler = Some(resampler);
        Ok(())
    }

    fn channels(&self) -> u16 {
        self.target_channels
    }

    fn duration(&self) -> Option<f64> {
        self.duration
    }

    fn read(&mut self, frames: usize) -> Result<Vec<f32>, String> {
        if frames == 0 {
            return Ok(Vec::new());
        }
        if self.target_channels == 0 {
            self.prime()?;
        }
        let channels = self.target_channels as usize;
        if channels == 0 {
            return Ok(Vec::new());
        }
        let wanted = frames
            .checked_mul(channels)
            .ok_or("Requested block is too large")?;
        let mut out = std::mem::take(&mut self.pending);

        while out.len() < wanted && !self.eof {
            let Some(chunk) = self.stream.next_chunk(&mut self.skip)? else {
                self.eof = true;
                if let Some(resampler) = &mut self.resampler {
                    out.extend(resampler.flush());
                }
                break;
            };
            if self.source_channels == 0 {
                self.source_channels = self.stream.channels;
                self.target_channels = if self.mono { 1 } else { self.source_channels };
                self.ensure_resampler()?;
            }
            let source_channels = self.source_channels.max(1);
            let converted = if self.mono {
                convert_channels(&chunk, source_channels, 1)
            } else if source_channels != self.target_channels {
                convert_channels(&chunk, source_channels, self.target_channels)
            } else {
                chunk
            };
            if let Some(resampler) = &mut self.resampler {
                out.extend(resampler.process(&converted));
            } else {
                out.extend(converted);
            }
        }

        let take = wanted.min(out.len());
        let rest = out.split_off(take);
        self.pending = rest;
        Ok(out)
    }

    /// AAC-in-MP4 has no channel layout in the container. Decode one packet so
    /// the public channel count and resampler are correct before the first
    /// block is requested.
    fn prime(&mut self) -> Result<(), String> {
        if self.source_channels != 0 || self.eof {
            return Ok(());
        }
        let Some(chunk) = self.stream.next_chunk(&mut self.skip)? else {
            self.eof = true;
            return Ok(());
        };
        self.source_channels = self.stream.channels;
        self.target_channels = if self.mono { 1 } else { self.source_channels };
        self.ensure_resampler()?;
        let source_channels = self.source_channels.max(1);
        let converted = if self.mono {
            convert_channels(&chunk, source_channels, 1)
        } else {
            chunk
        };
        if let Some(resampler) = &mut self.resampler {
            self.pending.extend(resampler.process(&converted));
        } else {
            self.pending.extend(converted);
        }
        Ok(())
    }
}

/// Incremental decoder. Yields `(frames, channels)` float32 blocks.
#[pyclass(unsendable)]
pub struct Decoder {
    core: Option<DecoderCore>,
    block_frames: usize,
}

#[pymethods]
impl Decoder {
    #[new]
    #[pyo3(signature = (file_path, block_frames=48_000, sample_rate=None, mono=false))]
    fn new(
        file_path: PathBuf,
        block_frames: usize,
        sample_rate: Option<u32>,
        mono: bool,
    ) -> PyResult<Self> {
        if block_frames == 0 {
            return Err(crate::error_with(
                "usage",
                "block_frames must be positive".to_string(),
            ));
        }
        if sample_rate == Some(0) {
            return Err(crate::error_with(
                "usage",
                "sample_rate must be positive".to_string(),
            ));
        }
        let path = crate::path_string(file_path)?;
        let core = DecoderCore::open(&path, sample_rate, mono).map_err(crate::error)?;
        Ok(Self {
            core: Some(core),
            block_frames,
        })
    }

    #[getter]
    fn sample_rate(&self) -> u32 {
        self.core.as_ref().map(|core| core.target_rate).unwrap_or(0)
    }

    #[getter]
    fn channels(&self) -> u16 {
        self.core.as_ref().map(DecoderCore::channels).unwrap_or(0)
    }

    #[getter]
    fn duration(&self) -> Option<f64> {
        self.core.as_ref().and_then(DecoderCore::duration)
    }

    #[getter]
    fn block_frames(&self) -> usize {
        self.block_frames
    }

    /// Read up to `frames` frames. Returns `(0, channels)` at EOF.
    #[pyo3(signature = (frames=None))]
    fn read<'py>(&mut self, py: Python<'py>, frames: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
        let wanted = frames.unwrap_or(self.block_frames);
        let core = self
            .core
            .as_mut()
            .ok_or_else(|| crate::error_with("usage", "Decoder is closed".to_string()))?;
        let samples = core.read(wanted).map_err(crate::error)?;
        decoded_array(py, samples, core.channels())
    }

    fn close(&mut self) {
        self.core = None;
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let core = self
            .core
            .as_mut()
            .ok_or_else(|| crate::error_with("usage", "Decoder is closed".to_string()))?;
        let samples = core.read(self.block_frames).map_err(crate::error)?;
        if samples.is_empty() {
            return Ok(None);
        }
        Ok(Some(decoded_array(py, samples, core.channels())?))
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__<'py>(
        &mut self,
        _py: Python<'py>,
        _ty: Bound<'py, PyAny>,
        _value: Bound<'py, PyAny>,
        _traceback: Bound<'py, PyAny>,
    ) {
        self.close();
    }
}

#[cfg(feature = "numpy")]
fn decoded_array<'py>(
    py: Python<'py>,
    samples: Vec<f32>,
    channels: u16,
) -> PyResult<Bound<'py, PyAny>> {
    use numpy::{PyArray1, PyArrayMethods};

    let channels = channels.max(1) as usize;
    let frames = samples.len() / channels;
    PyArray1::from_vec(py, samples)
        .reshape((frames, channels))
        .map(Bound::into_any)
        .map_err(|error| crate::error(error.to_string()))
}

#[cfg(not(feature = "numpy"))]
fn decoded_array<'py>(
    _py: Python<'py>,
    _samples: Vec<f32>,
    _channels: u16,
) -> PyResult<Bound<'py, PyAny>> {
    Err(crate::error_with(
        "unsupported",
        "NumPy support not enabled. Build with the 'numpy' feature.".to_string(),
    ))
}
