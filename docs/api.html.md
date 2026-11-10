# API reference

Nine functions, one class. Everything raises `sound_rs.SoundRsError`, a subclass of `RuntimeError`, on failure.

## Module

`sound_rs.__version__` — the installed version as a string, for example `"0.2.1"`.

`class sound_rs.SoundRsError(RuntimeError)` — the only exception type the library raises. Catch it instead of the individual causes.

## play_audio

```
sound_rs.play_audio(file_path: str, device: str | None = None) -> None
```

Plays a file and blocks until it ends. Releases the GIL, so other threads keep running. Opens the output device for this call, which costs about 0.1–0.2 s before the first sound — use [Player](#player) for repeated playback.

- `file_path` — any container the probe accepts.

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

Probes the container header. Does not decode, does not check that the audio data is intact, and does not raise for a missing file — it returns `False`. About 1 ms on any file size.

## get_audio_info

```
sound_rs.get_audio_info(file_path: str) -> tuple[int, int, float]
```

Returns `(sample_rate, channels, duration_seconds)`. Reads the header, so the cost does not depend on file length. For AAC the container has no channel layout, so one packet is decoded to learn it (~0.15 ms).

Duration is `None` in the Python sense only when the header has no frame count; in that case the file is decoded to measure it.

## load

```
sound_rs.load(file_path: str) -> tuple[numpy.ndarray, int]
```

Decodes the whole file to float32 with shape `(frames, channels)`, and returns it with the sample rate. Requires NumPy. Peak memory is the decoded audio: 10 minutes of stereo is 211 MB.

Raises `SoundRsError` for an unreadable file or when the build has no NumPy support.

## load_many

```
sound_rs.load_many(paths: Sequence[str], threads: int | None = None)
    -> list[tuple[numpy.ndarray, int]]
```

Decodes in parallel with a Rayon pool, releasing the GIL. Order matches `paths`. `threads=None` uses one worker per core. The first file that fails raises, and the rest of the batch is abandoned.

```
clips = sound_rs.load_many(["a.flac", "b.flac"], threads=2)
```

## play_array

```
sound_rs.play_array(array: numpy.ndarray, sample_rate: int) -> None
```

Plays a float32 array: `(frames,)` for mono or `(channels, frames)` for multichannel. Converts to the device layout and resamples if the rate differs, then blocks until playback ends.

Raises `TypeError` for anything that is not a 1D or 2D float32 ndarray, and `SoundRsError` for a device failure.

## play_sine_wave

```
sound_rs.play_sine_wave(frequency: float, duration: float, sample_rate: int) -> None
```

Generates a mono sine tone at half scale and plays it. Useful for checking that audio output works at all.

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

Writes float32 audio to a WAV file. The array is `(frames,)` for mono or `(frames, channels)` — the same layout `load` returns, so `sound_rs.write(path, *sound_rs.load(src))` round-trips. Samples outside −1…1 are clipped. Subtypes: `pcm16`, `pcm24`, `float32`.

Only `.wav` is accepted so far; any other extension raises `SoundRsError` with `code="usage"`. Returns nothing.

```
audio, rate = sound_rs.load("take.flac")
sound_rs.write("take.wav", audio, rate, subtype="pcm24")
```

## load_bytes

```
sound_rs.load_bytes(data: bytes) -> tuple[numpy.ndarray, int]
```

Decodes audio that is already in memory. Same shape, rate and gapless trimming as `load`; the MP4 edit list is read from the buffer as well. Corrupt data raises `SoundRsError` with `code="format"`.

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
| record_to(path, seconds, subtype) | None | record and write a WAV in one call |
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

Opens one output stream and keeps it open. Device name matching is exact; the default device is used when omitted. The stream closes when the object is deleted or when the `with` block exits.

### Methods

| Signature | Returns | Notes |
|---|---|---|
| play_file(path) | None | Streams from disk, returns immediately |
| play_array(array, rate) | None | 1D or 2D float32 |
| pause() | None | Position freezes |
| resume() | None | Continues, or replays if finished |
| stop() | None | Rewinds to zero |
| seek(seconds) | None | Frame-accurate, clamped to the track |
| wait() | None | Blocks until the source ends; GIL released |
| enqueue(path) | None | Plays after the current track; up to 128 entries |
| clear_queue() | None | Drops everything queued |
| begin_feed(rate) | None | Switches to samples pushed from Python |
| feed(array) | None | Blocks while the ring is full; call from one thread |
| end_feed() | None | No more samples; finishes when the ring drains |

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
| underruns | int | Callbacks that starved after audio started; healthy streams stay at 0 |
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

`SoundRsError` subclasses `RuntimeError`, so existing `except RuntimeError` handlers keep working.
