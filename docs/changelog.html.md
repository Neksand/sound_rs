# Changelog

## Unreleased — media validation

- Bounds-check MP4 box lengths before arithmetic and slicing.

- Cap metadata-driven initial decode reservations at 32 MiB; this is not a limit on total decoded audio memory.

- Validate WAV dimensions, sizes and finite samples before touching the destination; include odd PCM24 padding in RIFF size and classify write errors as `io`.

- Use checked NumPy read borrows; reject zero or overflowing channel counts.

- Reject invalid load/synthesis parameters and non-finite CLI volume. WAV encoding and disk I/O release the GIL.

## Unreleased — playback lifecycle

- Drain the final written audio prefix, complete empty feeds and time out only stalled producers. Paused wait returns without discarding the queue.

- Preserve pause/queue across seek, track the active queued file, and reject invalid source transitions, non-finite samples/volume, incomplete frames and zero rates.

- Report backend failures through an atomic flag and avoid blocking on callback state contention. Keep hardware recovery and continuous feed resampling as future work.

## Unreleased — bounded recording

- Move capture allocation and synchronization out of the input callback into a collector behind a bounded ring; preserve the 230 M sample storage limit.

- Report overflow, allocation and device faults; drain final input on shutdown and release the GIL while joining the collector.

- Make the documented `record_to(..., subtype="pcm16")` default effective at runtime and include context manager methods in the type stub.

## Unreleased — streaming resampler

- Drain filter delay at EOF even when input ends on a full block, remove startup delay for very short tracks, and trim output padding to the exact converted frame count.

- Test numerical output against a full filter pass across packet sizes, sample rates, stereo channels and impulses at the end of a track. Full-decode fixtures are unchanged.

Short history. Full notes with diffs live on [GitHub Releases](https://github.com/Neksand/sound_rs/releases).

## 0.4.0

- `sound_rs.write(path, audio, rate, subtype=)` writes WAV in PCM16, PCM24 or float32. Arrays are `(frames, channels)`, matching `load`, so a load/write round-trip needs no transpose.

- `Recorder.record_to(path, seconds, subtype=)` records and saves in one call.

- Non-contiguous arrays (`audio[:, 0]`) are accepted by `write` and `play_array` instead of raising.

## 0.3.2

- `load_bytes` trims AAC priming too: the MP4 edit list is read from the buffer, so bytes decode to the same frame count as a file.

- `record(seconds)` never returns more than the requested frames; the extra device buffer at the end is cut.

## 0.3.1

- PyPI carries the corrected README: roadmap, and one figure for time-to-first-sound.

## 0.3.0

- Stable error codes, mutation fuzzing, deterministic decode hashes, underrun and latency reporting.

- `load(..., sample_rate, mono, offset, duration)`, `load_bytes`, `Recorder`, `Player.enqueue`, `begin_feed/feed`.

- Benchmark harness v2 and abi3 wheels for five platforms.

## 0.2.2

- `soundrs` console command: `play`, `info`, `devices`, `docs`, `--version`, `--help`.

- Documentation link in the package metadata.

## 0.2.1

- `sound_rs.__version__`.

- Metadata for discovery: keywords, classifiers, documentation links; wheels and `requires-python` now agree on Python 3.10+.

- Small internal cleanups; no behaviour change.

## 0.2.0

- `Player` with play, pause, resume, stop, seek, volume, position, duration.

- Streaming decode: constant memory, ~11 ms to first sound on a 10-minute file.

- rubato resampler replaces linear interpolation; downsampling no longer aliases.

- MP3 decoded with minimp3 (1.1× faster than libsndfile, previously slower).

- `load_many`, `list_devices`, device selection, typed stubs, `SoundRsError`.

## 0.1.8

- Gapless trimming for MP3 and OGG; AAC priming read from the MP4 edit list.

- Metadata without decoding; header-only duration.

- Playback waits for the exact device drain time instead of a fixed 100 ms.

## 0.1.7

- Wheels for Linux, Windows and macOS across Python 3.10–3.14, published from CI.
