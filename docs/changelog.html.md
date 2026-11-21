# Changelog

## 0.7.0

- `sound_rs.probe(path)` returns a metadata dictionary with codec, bit depth, frames and duration, without decoding the whole file. Missing fields are `None`.

- `play_array(..., layout="frames")` accepts the same `(frames, channels)` layout as `load`; the default remains channels-first for compatibility.

## 0.6.0

- `soundrs mcp` starts a Model Context Protocol stdio server with `audio_info` and `convert` tools. It never opens an audio device and works in headless containers.

- The MCP server has no extra runtime dependency and is covered by a stdio integration test.

## 0.5.1

- Publish an x86_64 `musllinux_1_2` wheel and smoke-test it in Alpine; the wheel bundles libasound, so `import sound_rs` and decoding need no system packages.

- Release validation checks the bundled ALSA runtime instead of installing a host copy.

## 0.5.0

- `sound_rs.Decoder`: bounded-memory incremental reads with optional resampling and mono folding, iteration, `read()` and context management.

- `write_bytes()` encodes WAV without touching disk; `load_many( on_error="skip")` keeps batches going past corrupt files.

- CLI: `soundrs info --json` and `soundrs convert`, with typed exit codes and no new dependencies.

- AIFF, CAF and ALAC-in-M4A decoding with lossless fixtures; `pathlib.Path` is accepted wherever a file path is taken.

- One MP3 decoder: Symphonia replaces minimp3 for full-file, bytes, windows and streaming. The 47-sample LAME/Xing offset is gone; MP3 regression uses a sample hash. Full-file MP3 decode is about 26% slower on the 60-second benchmark (~2,000× realtime).

- Continuous fed resampling with retained filter state; `end_feed()` flushes the tail. `Player.paused` reports pause state.

- Fail `play_file()` before replacing playback on an unreadable path; raise an error when a whole-buffer resampler cannot run; validate `record_to()` destinations before capture; reject `record()` outside 0…3600 seconds.

- Linux wheels bundle the ALSA runtime; source builds need ALSA development headers. Release validation checks the bundled library instead of installing one.

- CI runs Python 3.13, tests without the default `numpy` feature, and runs `cargo audit` without ignores (192 packages, no advisories).

Short history. Full notes with diffs live on [GitHub Releases](https://github.com/Neksand/sound_rs/releases).

## 0.4.2

- PyPI carries the corrected README: the release line points at the current version, and every link is absolute so it resolves on the package page.

## 0.4.1

- Python inputs and container boundaries are validated before arithmetic: invalid times, rates, dimensions and truncated WAV headers raise `usage` or `io` instead of overflowing or producing a broken file.

- Playback lifecycle: the final written prefix is drained, a user pause survives queue handover, and backend faults are signalled instead of swallowed.

- Recording captures through a bounded ring into a collector thread; overflow and device faults raise instead of returning an incomplete take.

- Streaming resampling keeps an exact converted frame count, including the filter tail.

- PyO3 0.29.3 and numpy 0.29 (past the RustSec advisory range); blocking paths moved to `py.detach`.

- New checks: 31 Python tests, generated-site validation, headless benchmark mode.

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
