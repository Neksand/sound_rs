# Guarantees

A library is trustworthy when its promises are written down and tested. This is the list, and the ones that are deliberately not on it.

## Errors

Library failures use `sound_rs.SoundRsError` (a `RuntimeError`) with a stable `.code`. Python argument conversion can raise `TypeError` or `OverflowError`; float64 arrays are rejected rather than silently converted.

| Code | Meaning |
|---|---|
| io | The file could not be opened, read or written. |
| format | The container or codec was not understood, or no audio was decodable. |
| device | No output/input device, open failure or reported backend fault. |
| stalled | Playback/feed made no progress, or the recording collector could not keep up. |
| unsupported | The build lacks a feature (NumPy) or hit a documented limit. |
| usage | Invalid values, WAV dimensions or call order, for example  feed()  before  begin_feed() . |

Codes are part of the API: renaming one is a breaking change.

## No panics on bad input

The contract is to report malformed input without a panic crossing the Python boundary. Regression coverage in `cargo test` includes mutating every committed fixture ~200 times per file (bit flips, truncation, appended noise) and by feeding random bytes; every case must return `Ok` with finite samples or a non-empty error. This bounded test corpus is not exhaustive fuzzing or proof that arbitrary untrusted media is safe.

## Threads and the GIL

- Bulk decoding/writing, playback waits, feed backpressure and timed capture waits release the GIL. Python array preparation, player joins and device teardown can still hold it.

- Create, use and destroy each `Player` or `Recorder` on the same thread. Sequential cross-thread use is also invalid.

- Separate threads can own separate instances, subject to backend/device sharing limits.

## Determinism

`tests/decode_regression.py` checks exact dimensions/rates for six fixtures, SHA-256 hashes of samples rounded to 10⁻⁴ for five, and an RMS tolerance for MP3. Missing fixtures fail the check. These are regression bounds, not a guarantee that every sample change is detected or that decoder output is bit-identical across CPUs and dependency versions.

## Memory

- Streaming playback uses a one-second ring; container, decoder, filter and process memory are additional. This bounds playback buffering, not total process RSS.

- `load` retains its result plus decoding/conversion temporaries; `load_many` retains the whole batch. The initial reservation ceiling is not a total decode-memory limit.

- Recording limits captured payload to 230 M samples (920 MB), with ring/process overhead additional. `record_to` and `write` also build an encoded WAV buffer.

- Historical streaming and soak RSS measurements are observations on one system; [benchmark results](benchmarks.html) are not memory guarantees.

## Audio quality

- Resampling uses rubato's FFT resampler: a 20 kHz tone downsampled to 24 kHz is filtered below 10% of its level instead of folding back as an alias.

- Encoder-delay/padding metadata is used when available; MP3 decoder paths differ and AAC can retain tail padding. See [format limits](formats.html#gapless).

- Historical test-corpus comparisons found identical WAV/FLAC PCM, MP3 agreement within 1.6·10⁻⁵ after a 47-sample alignment adjustment, and AAC agreement within 8·10⁻⁸. These are fixture results, not universal codec guarantees.

## What is not promised

- **A latency figure.** Latency depends on the OS, the device and its buffer; `Player.latency` reports what the backend measured, nothing more.

- **Zero glitches or a complete glitch counter.** `Player.underruns` counts ring starvation after playback starts. Callback lock contention produces silence without updating this counter; backend/device glitches are not fully observable.

- **Real-time safety of the Python API.** Calling `feed()` from Python is fine, but Python's scheduler is not a real-time scheduler.

- **Bit-identical output across decoders.** Different MP3 decoders trim delay differently; fixture comparisons do not establish equivalence for every file.
