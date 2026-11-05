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
    "load_bytes",
    "load_many",
    "play_array",
    "play_audio",
    "play_sine_wave",
    "cli",
    "__version__",
]

class SoundRsError(RuntimeError): ...

def play_audio(file_path: str, device: Optional[str] = None) -> None:
    """Play a file, blocking until it ends. Releases the GIL while playing."""

def is_supported_format(file_path: str) -> bool:
    """True if the container can be probed. Does not decode the audio."""

def get_audio_info(file_path: str) -> Tuple[int, int, float]:
    """`(sample_rate, channels, duration_seconds)` from the container header."""

def load(file_path: str) -> Tuple[npt.NDArray[np.float32], int]:
    """Decode to a float32 array of shape `(frames, channels)` and its rate."""

def load_bytes(data: bytes) -> Tuple[npt.NDArray[np.float32], int]:
    """Decode audio that is already in memory."""

def load_many(
    paths: Sequence[str], threads: Optional[int] = None
) -> List[Tuple[npt.NDArray[np.float32], int]]:
    """Decode several files in parallel, in the order given."""

def play_array(array: npt.NDArray[np.float32], sample_rate: int) -> None:
    """Play a 1D mono `(samples,)` or 2D `(channels, samples)` float32 array."""

def play_sine_wave(frequency: float, duration: float, sample_rate: int) -> None: ...

def cli() -> int:
    """Entry point for the `soundrs` console command (reads sys.argv)."""

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
    def __enter__(self) -> "Player": ...
    def __exit__(self, *args: object) -> None: ...
