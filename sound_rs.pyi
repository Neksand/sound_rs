from os import PathLike
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

_Path = Union[str, PathLike[str]]

__version__: str

__all__ = [
    "Player",
    "SoundRsError",
    "get_audio_info",
    "is_supported_format",
    "list_devices",
    "load",
    "Recorder",
    "list_input_devices",
    "load_bytes",
    "load_many",
    "play_array",
    "play_audio",
    "write",
    "write_bytes",
    "play_sine_wave",
    "cli",
    "__version__",
    "Decoder",
]

class SoundRsError(RuntimeError):
    code: str

def play_audio(file_path: _Path, device: Optional[str] = None) -> None:
    """Play a file, blocking until it ends. Releases the GIL while playing."""

def is_supported_format(file_path: _Path) -> bool:
    """True if the container can be probed. Does not decode the audio."""

def get_audio_info(file_path: _Path) -> Tuple[int, int, float]:
    """`(sample_rate, channels, duration_seconds)` from the container header."""

def load(
    file_path: _Path,
    sample_rate: Optional[int] = None,
    mono: bool = False,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> Tuple[npt.NDArray[np.float32], int]:
    """Decode to a float32 array of shape `(frames, channels)` and its rate.

    `sample_rate` resamples, `mono` folds to one channel, `offset` and
    `duration` (seconds) decode only that window. Times must be finite and
    non-negative; a requested sample rate must be positive.
    """

def write(
    file_path: _Path,
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    subtype: str = "pcm16",
) -> None:
    """Write `(frames,)` or `(frames, channels)` float32 to a WAV file.

    `subtype` is `pcm16`, `pcm24` or `float32`. Samples outside -1..1 are clipped.
    Non-finite samples and invalid WAV dimensions raise SoundRsError(code="usage").
    File failures use code="io". Encoding and disk I/O release the GIL.
    """

def load_bytes(data: bytes) -> Tuple[npt.NDArray[np.float32], int]:
    """Decode audio that is already in memory."""

def write_bytes(
    audio: npt.NDArray[np.float32],
    sample_rate: int,
    subtype: str = "pcm16",
) -> bytes:
    """Encode `(frames,)` or `(frames, channels)` float32 as the bytes of a WAV.

    Same layout, subtypes and validation as `write`, but nothing is saved.
    """

def load_many(
    paths: Sequence[_Path],
    threads: Optional[int] = None,
    sample_rate: Optional[int] = None,
    mono: bool = False,
    on_error: str = "raise",
) -> List[Optional[Tuple[npt.NDArray[np.float32], int]]]:
    """Decode several files in parallel, in the order given.

    `on_error="raise"` raises for the first file that fails; `"skip"` puts
    `None` in that slot instead.
    """

def play_array(array: npt.NDArray[np.float32], sample_rate: int) -> None:
    """Play a 1D mono `(samples,)` or 2D `(channels, samples)` float32 array."""

def play_sine_wave(frequency: float, duration: float, sample_rate: int) -> None: ...

def cli() -> int:
    """Entry point for the `soundrs` console command (reads sys.argv)."""

def list_input_devices() -> List[Tuple[int, str, bool]]:
    """Input devices as `(index, name, is_default)`."""

class Recorder:
    """Records from an input device."""

    def __init__(self, device: Optional[str] = None) -> None: ...
    def __enter__(self) -> Recorder: ...
    def __exit__(self, exc_type, exc_value, traceback) -> None: ...
    @property
    def device(self) -> str: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def channels(self) -> int: ...
    def start(self) -> None: ...
    def stop(self) -> Tuple[npt.NDArray[np.float32], int]: ...
    def record(self, seconds: float) -> Tuple[npt.NDArray[np.float32], int]:
        """Record for 0 to 3600 seconds; longer takes use start() and stop()."""
    def record_to(
        self, path: _Path, seconds: float, subtype: str = "pcm16"
    ) -> None:
        """Record to memory, then write a WAV; validates `path` and `subtype` first.

        `seconds` is limited to 0..3600; longer takes use start() and stop().
        """

def list_devices() -> List[Tuple[int, str, bool]]:
    """Output devices as `(index, name, is_default)`."""

class Decoder:
    """Incremental decoder for files larger than memory.

    Yields `(frames, channels)` float32 blocks. `sample_rate` resamples with
    retained filter state and `mono` folds channels.
    """

    def __init__(
        self,
        file_path: _Path,
        block_frames: int = 48000,
        sample_rate: Optional[int] = None,
        mono: bool = False,
    ) -> None: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def channels(self) -> int: ...
    @property
    def duration(self) -> Optional[float]: ...
    @property
    def block_frames(self) -> int: ...
    def read(self, frames: Optional[int] = None) -> npt.NDArray[np.float32]:
        """Read up to `frames` frames; returns an empty block at EOF."""
    def close(self) -> None: ...
    def __iter__(self) -> Decoder: ...
    def __next__(self) -> npt.NDArray[np.float32]: ...
    def __enter__(self) -> Decoder: ...
    def __exit__(self, *args: object) -> None: ...

class Player:
    """One output stream kept open, so playback starts without re-opening the device."""

    def __init__(self, device: Optional[str] = None) -> None: ...
    @property
    def device(self) -> str: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def channels(self) -> int: ...
    @property
    def volume(self) -> float: ...
    @volume.setter
    def volume(self, value: float) -> None: ...
    @property
    def position(self) -> float: ...
    @property
    def duration(self) -> Optional[float]: ...
    @property
    def is_playing(self) -> bool: ...
    @property
    def paused(self) -> bool: ...
    def play_file(self, path: _Path) -> None:
        """Stream a file; returns immediately, audio is decoded as it plays."""
    def play_array(self, array: npt.NDArray[np.float32], sample_rate: int) -> None: ...
    def play_samples(self, samples: Sequence[float], src_channels: int, sample_rate: int) -> None:
        """Play interleaved samples with an explicit source channel count."""
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def stop(self) -> None: ...
    def seek(self, seconds: float) -> None: ...
    def wait(self) -> None:
        """Block until the current source finishes."""
    @property
    def underruns(self) -> int: ...
    @property
    def latency(self) -> Optional[float]: ...
    def enqueue(self, path: _Path) -> None: ...
    @property
    def queue(self) -> List[str]: ...
    def clear_queue(self) -> None: ...
    def begin_feed(self, sample_rate: int) -> None: ...
    def feed(self, array: npt.NDArray[np.float32]) -> None: ...
    def end_feed(self) -> None: ...
    def __enter__(self) -> "Player": ...
    def __exit__(self, *args: object) -> None: ...
