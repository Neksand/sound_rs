# API reference

Functions for files and arrays, plus persistent `Player` and `Recorder` classes. Library failures use `sound_rs.SoundRsError`; Python argument conversion can raise `TypeError` or `OverflowError`.

## Module

`sound_rs.__version__` — the installed version as a string, for example `"0.2.1"`.

`class sound_rs.SoundRsError(RuntimeError)` — library failures carry a stable `.code`; see [error codes](guarantees.html#errors).

## play_audio

```
sound_rs.play_audio(file_path: str, device: str | None = None) -> None
```

Plays a file and blocks until it ends, releasing the GIL while waiting for playback. Opens the output device for this call; startup cost depends on the backend and device. Use [Player](#player) for repeated playback.

- `file_path` — a file with a supported container and codec.

- `device` — exact device name from `list_devices()`; the default device when omitted.

Raises `SoundRsError` for an unreadable file, a missing device, or a device that disconnects mid-playback.

```
import sound_rs

sound_rs.play_audio("music.flac")
sound_rs.play_audio("music.flac", device="MacBook Pro Speakers")
```

## is_supported_format

```
sound_rs.is_supported_format(file_path: str) -> bool
```

Probes the container header. Does not decode or verify codec support and packet integrity. A missing or unrecognized file returns `False`. Probe time depends on the container and storage.

## get_audio_info

```
sound_rs.get_audio_info(file_path: str) -> tuple[int, int, float]
```

Returns `(sample_rate, channels, duration_seconds)`. Uses the header when complete; missing rate or channel information can require decoding a packet.

If the header has no frame count, the file is decoded to measure duration. The returned duration is a number; this function does not return `None`.

## load

```
sound_rs.load(file_path: str, sample_rate: int | None = None,
              mono: bool = False, offset: float = 0.0,
              duration: float | None = None) -> tuple[numpy.ndarray, int]
```

Decodes to float32 with shape `(frames, channels)`, returning the array and rate. Optional `sample_rate` resamples, `mono` folds channels, and `offset`/`duration` select a window in seconds. Times must be finite and non-negative; a requested rate must be positive. Requires NumPy. Memory includes the returned audio plus decoder and conversion/resampling temporaries.

Raises `SoundRsError` for an unreadable file or when the build has no NumPy support.

`offset` and `duration` are finite, non-negative seconds. `duration=0` returns an empty array. A requested `sample_rate` must be positive; invalid options raise `SoundRsError(code="usage")` before file access. `mono=True` averages channels before resampling.

## load_many

```
sound_rs.load_many(paths: Sequence[str], threads: int | None = None,
                   sample_rate: int | None = None, mono: bool = False)
    -> list[tuple[numpy.ndarray, int]]
```

Decodes in parallel with a Rayon pool, releasing the GIL. Order matches `paths`. `threads=None` or `threads=0` uses the automatic worker count. All files are processed, then the first failure in input order is raised. A requested sample rate must be positive, including for an empty batch. Optional rate and mono conversion apply to every file; the whole batch is retained in memory.

```
clips = sound_rs.load_many(["a.flac", "b.flac"], threads=2)
```

## play_array

```
sound_rs.play_array(array: numpy.ndarray, sample_rate: int) -> None
```

Plays a float32 array: `(frames,)` for mono or `(channels, frames)` for multichannel. Converts to the device layout and resamples if the rate differs, then blocks until playback ends.

Raises `TypeError` for an unsupported dtype/dimensionality, `SoundRsError(code="usage")` for invalid rates, channel counts or non-finite samples, and `SoundRsError(code="device")` for device failures.

## play_sine_wave

```
sound_rs.play_sine_wave(frequency: float, duration: float, sample_rate: int) -> None
```

Generates a mono sine tone at half scale and plays it. Useful for checking that audio output works at all. Frequency and duration must be finite and non-negative, and the sample rate positive; invalid values raise `SoundRsError(code="usage")` before a device opens.

```
sound_rs.play_sine_wave(440.0, 1.0, 44100)
```

## cli

```
sound_rs.cli() -> int
```

Entry point for the `soundrs` console script; reads `sys.argv` and returns an exit code. You do not call it from Python — see [Command line](cli.html).

## write

```
sound_rs.write(file_path: str, audio, sample_rate: int, subtype: str = "pcm16") -> None
```

Writes float32 audio to a WAV file. The array is `(frames,)` for mono or `(frames, channels)` — the same layout `load` returns, so `sound_rs.write(path, *sound_rs.load(src))` round-trips. Samples outside −1…1 are clipped. Subtypes: `pcm16`, `pcm24`, `float32`. Samples must be finite and the rate positive. See [Writing WAV files](writing.html) for memory and RIFF limits.

Only `.wav` is accepted so far; any other extension raises `SoundRsError` with `code="usage"`. Returns nothing.

The sample rate and channel count must be positive and fit the WAV header. NaN and infinity, incomplete frames, and files too large for classic RIFF are rejected with `code="usage"` before opening the destination. A file-write failure has `code="io"`. Encoding and disk I/O release the GIL after copying the array. Read-only and strided float32 arrays are accepted; float64 remains a `TypeError`.

```
audio, rate = sound_rs.load("take.flac")
sound_rs.write("take.wav", audio, rate, subtype="pcm24")
```

## load_bytes

```
sound_rs.load_bytes(data: bytes) -> tuple[numpy.ndarray, int]
```

Decodes bytes to a float32 `(frames, channels)` array and its sample rate. AAC-in-MP4 priming is read from the buffer's edit list. MP3 uses Symphonia here, while a full-file `load` uses minimp3; samples and alignment can differ. Malformed data can raise `SoundRsError(code="format")`; an unexpected EOF may yield a decodable prefix.

## list_devices

```
sound_rs.list_devices() -> list[tuple[int, str, bool]]
```

Output devices as `(index, name, is_default)`. The index is only for display — pass the name to `play_audio` or `Player`.

## Recorder

```
sound_rs.Recorder(device: str | None = None)
```

Opens an input device. Raises `SoundRsError` with `code="device"` when there is no microphone or the OS denies access.

| Member | Returns | Notes |
|---|---|---|
| start() | None | Starts capturing |
| stop() | (ndarray, rate) | Stops and returns what was captured |
| record(seconds) | (ndarray, rate) | start + wait + stop |
| record_to(path, seconds, subtype="pcm16") | None | Capture in memory, then write a WAV |
| device / sample_rate / channels | str / int / int | Device properties |

## list_input_devices

```
sound_rs.list_input_devices() -> list[tuple[int, str, bool]]
```

Input devices as `(index, name, is_default)`. The name is what `Recorder(device=...)` accepts.

## Player

```
sound_rs.Player(device: str | None = None)
```

Opens one output stream and keeps it open. Device name matching is exact; the default device is used when omitted. Create, use and destroy it on the same thread. A `with` block stops playback on exit; the device stream closes when the object is destroyed.

### Methods

| Signature | Returns | Notes |
|---|---|---|
| play_file(path) | None | Streams from disk, returns immediately |
| play_array(array, rate) | None | 1D or 2D float32 |
| play_samples(samples, src_channels, sample_rate) | None | Interleaved Python sequence, converted to float32; no NumPy required |
| pause() | None | Position freezes |
| resume() | None | Continues an active source; a finished feed requires begin_feed again |
| stop() | None | Clears source/queue and resets position; keeps device open |
| seek(seconds) | None | Finite file/array target, clamped to known duration; preserves pause |
| wait() | None | Waits for completion/drain or returns while paused/stopped; GIL released |
| enqueue(path) | None | Requires active file source; up to 128 waiting entries |
| clear_queue() | None | Drops everything queued |
| begin_feed(rate) | None | Switches to samples pushed from Python |
| feed(array) | None | Requires begin_feed; blocks on full ring, times out after five seconds without progress |
| end_feed() | None | No more samples; finishes when the ring drains |

`play_samples` is an existing Python-exposed method. Supply complete interleaved frames: `[left0, right0, left1, right1]` with `src_channels=2`. It copies the input and prepares it for the device, then returns while playback continues. Prefer `play_array` for NumPy arrays.

### Properties

| Name | Type | Notes |
|---|---|---|
| device | str | Name the player opened |
| sample_rate | int | Device rate |
| channels | int | Device channel count |
| volume | float | Read/write, clamped to 0…4 |
| position | float | Seconds played |
| duration | float \| None | Seconds;  None  until known |
| is_playing | bool | False  when paused, stopped or finished |
| queue | list[str] | Paths waiting behind the current track |
| underruns | int | Ring-starvation callbacks after audio started; excludes silence from lock contention |
| latency | float \| None | Last device latency in milliseconds |

```
import sound_rs

with sound_rs.Player() as player:
    player.play_file("track.flac")
    while player.position < 10:
        pass                      # do work while it plays
    player.seek(120.0)
    player.wait()
```

## Exceptions

| Raised by | When |
|---|---|
| SoundRsError | Unreadable file, no output device, device disconnected, stalled stream, no NumPy support |
| TypeError | An array that is not a 1D/2D float32 ndarray |
| OverflowError | Python integer outside a Rust argument's representable range |

`SoundRsError` subclasses `RuntimeError`, so existing `except RuntimeError` handlers keep working.
