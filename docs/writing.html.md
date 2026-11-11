# Writing WAV files

`write` saves a float32 array as a WAV file. It accepts the `(frames, channels)` layout returned by `load`, or a one-dimensional mono array. It requires NumPy.

```
import sound_rs

audio, rate = sound_rs.load("source.flac")
sound_rs.write("copy.wav", audio, rate, subtype="pcm24")
```

## Subtypes and sample values

| Subtype | Encoding | Bytes per sample |
|---|---|---|
| pcm16 | Signed 16-bit PCM (default) | 2 |
| pcm24 | Signed 24-bit PCM | 3 |
| float32 | 32-bit IEEE float | 4 |

All three accept finite float32 samples and clip values outside −1…1. Integer PCM quantizes the samples; no dither is added. Strided and read-only arrays are accepted through a checked copy. Float64 input is rejected rather than converted silently.

## Validation and errors

The destination must have a `.wav` extension, the rate must be positive, and the channel count and byte rate must fit the WAV header. Invalid values or unsupported subtypes raise `SoundRsError(code="usage")` before opening the destination. File creation/write failures use `code="io"`; Python dtype or integer conversion can raise `TypeError` or `OverflowError`.

An existing destination is overwritten. Writes are not atomic: an I/O failure can leave an incomplete file. The format is classic RIFF WAV, not RF64; files beyond its size limits are rejected.

## Saving a recording

```
import sound_rs

with sound_rs.Recorder() as recorder:
    recorder.record_to("take.wav", 5.0, subtype="pcm16")
```

`record_to` captures into memory and then calls the WAV writer. It does not stream samples to disk while recording. See [capture limits](recording.html#limits).

## Memory and concurrency

The writer copies the input and builds the complete encoded WAV buffer before writing it. Budget for the array, copied samples and encoded bytes together. Encoding and file I/O release the GIL after Python array preparation. Incremental writing and compressed output are not implemented.
