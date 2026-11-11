# Guarantees

A library is trustworthy when its promises are written down and tested. This is the list, and the ones that are deliberately not on it.

## Errors

Library failures use `sound_rs.SoundRsError` (a `RuntimeError`) with a stable `.code`. Python argument conversion can raise `TypeError` or `OverflowError`; float64 arrays are rejected rather than silently converted.

| Code | Meaning |
|---|---|
| io | The file could not be opened, read or written. |
| format | The container or codec was not understood, or no audio was decodable. |
| device | No output/input device, or the device refused to open. |
| stalled | Playback made no progress for two seconds; the device probably disappeared. |
| unsupported | The build lacks a feature (NumPy) or hit a documented limit. |
| usage | Invalid values, WAV dimensions or call order, for example  feed()  before  begin_feed() . |

Codes are part of the API: renaming one is a breaking change.

## No panics on bad input

Corrupt files return an error, they do not abort the interpreter. This is enforced in `cargo test` by mutating every committed fixture ~200 times per file (bit flips, truncation, appended noise) and by feeding random bytes; every case must return `Ok` with finite samples or a non-empty error.

## Threads and the GIL

- Decoding, playback, recording and `wait()` release the GIL.

- One `Player` is one stream: call its methods from one thread at a time.

- Different players can be used from different threads, each with its own device handle.

## Determinism

Decoding is deterministic: the same file produces the same float32 samples on every run. Across CPUs the integer decoder's SIMD paths can differ by a bit, so the regression hashes are taken at 10⁻⁴ (about three LSB at 16-bit scale). `tests/decode_regression.py` pins a SHA-256 of the PCM for five fixtures; a decoder change that moves a single sample fails CI.

## Memory

- Streaming playback uses a one-second ring buffer: memory does not grow with file length (43 MB peak on a 10-minute file, measured).

- `load` allocates the decoded audio once; `load_many` allocates all results at once, so size your batches.

- The recorder caps its buffer at 230 M samples instead of growing without bound.

- A 100-cycle playback soak grows RSS by less than 20 MB (measured: 0.0 MB).

## Audio quality

- Resampling uses rubato's FFT resampler: a 20 kHz tone downsampled to 24 kHz is filtered below 10% of its level instead of folding back as an alias.

- Gapless: encoder delay and padding are trimmed for MP3, OGG and AAC/MP4.

- WAV and FLAC decode to PCM identical to libsndfile; MP3 matches to 1.6·10⁻⁵ with a 47-sample offset; AAC matches ffmpeg to 8·10⁻⁸.

## What is not promised

- **A latency figure.** Latency depends on the OS, the device and its buffer; `Player.latency` reports what the backend measured, nothing more.

- **Zero underruns everywhere.** `Player.underruns` counts them; on a loaded machine or a Bluetooth sink they will appear.

- **Real-time safety of the Python API.** Calling `feed()` from Python is fine, but Python's scheduler is not a real-time scheduler.

- **Bit-identical output across decoders.** Different MP3 decoders trim delay differently; we match length exactly and samples to 1.6·10⁻⁵.
