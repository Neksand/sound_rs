from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

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
    "play_sine_wave",
    "cli",
    "__version__",
]

class SoundRsError(RuntimeError):
    code: str

def play_audio(file_path: str, device: Optional[str] = None) -> None:
    """Play a file, blocking until it ends. Releases the GIL while playing."""

def is_supported_format(file_path: str) -> bool:
    """True if the container can be probed. Does not decode the audio."""

def get_audio_info(file_path: str) -> Tuple[int, int, float]:
    """`(sample_rate, channels, duration_seconds)` from the container header."""

def load(
    file_path: str,
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
    file_path: str,
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

def load_many(
    paths: Sequence[str],
    threads: Optional[int] = None,
    sample_rate: Optional[int] = None,
    mono: bool = False,
) -> List[Tuple[npt.NDArray[np.float32], int]]:
    """Decode several files in parallel, in the order given."""

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
    def record(self, seconds: float) -> Tuple[npt.NDArray[np.float32], int]: ...
    def record_to(
        self, path: str, seconds: float, subtype: str = "pcm16"
    ) -> None: ...

def list_devices() -> List[Tuple[int, str, bool]]:
    """Output devices as `(index, name, is_default)`."""

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
    def play_file(self, path: str) -> None:
        """Stream a file; returns immediately, audio is decoded as it plays."""
    def play_array(self, array: npt.NDArray[np.float32], sample_rate: int) -> None: ...
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
    def enqueue(self, path: str) -> None: ...
    @property
    def queue(self) -> List[str]: ...
    def clear_queue(self) -> None: ...
    def begin_feed(self, sample_rate: int) -> None: ...
    def feed(self, array: npt.NDArray[np.float32]) -> None: ...
    def end_feed(self) -> None: ...
    def __enter__(self) -> "Player": ...
    def __exit__(self, *args: object) -> None: ...
