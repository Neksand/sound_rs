# AGENTS.md

Guidance for coding agents working on `sound_rs`. Humans: see `README.md`.

## What this is

A Python audio library written in Rust (PyO3 + maturin): playback, streaming, recording,
decoding to NumPy, and now WAV writing. No ffmpeg, no system decoders.

## Layout

| Path | Owns |
|---|---|
| `src/lib.rs` | module registration, most PyO3 functions, NumPy conversion, `SoundRsError` |
| `src/audio.rs` | decoding (symphonia/minimp3), gapless trim, MP4 edit list, resampling (rubato), channel conversion |
| `src/player.rs` | the persistent output stream: `Player`, queue, fed streams, underruns |
| `src/recorder.rs` | input devices and `Recorder` |
| `src/writer.rs` | WAV encoding (`write`) |
| `src/cli.rs` | the `soundrs` command |
| `tests/` | smoke tests, decode fixtures, PCM hashes |
| `bench/` | benchmark harness (`run.py`) |
| `docs/` | static docs site; content lives in `docs/_content.py` |

## Commands

```bash
cargo test                                   # unit + fuzz + properties
cargo clippy --all-targets -- -D warnings    # must be clean
cargo fmt --all -- --check
maturin develop --release                    # build into the active venv
python tests/smoke.py                        # API end-to-end, no audio needed
SOUND_RS_PLAY=1 python tests/smoke.py        # + real playback and recording
python tests/decode_regression.py            # PCM hashes; must not change
python docs/check_api.py                     # every public name is documented
python docs/check_site.py                    # local links and generated indexes
python -m unittest discover -s tests -p 'test_*.py'
python3 docs/_build.py                       # regenerate docs, llms.txt, .md pages
python bench/run.py --compare                # benchmark
python bench/run.py --compare --no-audio     # no speaker/microphone access
```

CI is configured for Rust tests on Linux, Windows and macOS; headless Python checks run on
Python 3.12 on each OS and Python 3.10/3.14 on Linux. Hardware playback/recording and full benchmarks are
separate optional runs; never report them passed merely because a headless job passed.

## Invariants — do not break

1. **Keep registered APIs synchronized:** Rust functions/classes/members, `sound_rs.pyi`,
   and `docs/api.html` (via `docs/_content.py`). `docs/check_api.py` compares registrations
   and members with the stub and checks scoped documentation; it does not verify signatures.
2. **Library failures use `SoundRsError` with a `.code`** from `io`, `format`, `device`,
   `stalled`, `unsupported`, `usage`. Python argument conversion can raise `TypeError` or
   `OverflowError`; preserve strict float32 dtype rejection. No panic may cross the FFI boundary.
3. **Audio data callbacks must not allocate, block or touch Python.** Playback copies and
   applies gain from an `Arc<Vec<f32>>` or an `rtrb` ring; capture converts into a bounded
   ring. Silence on playback lock contention is not counted in `Player.underruns`.
4. **Release the GIL around bulk decode/write and playback/recording waits** using the
   PyO3 `py.detach` API.
   Do not claim every blocking path releases it: array preparation, player joins and device
   teardown still have ownership constraints. Create, use and destroy each Player/Recorder
   on the same thread; sequential access from another thread is also invalid.
5. **Arrays are float32.** `load`/`write` use `(frames, channels)`; `play_array` takes
   `(channels, frames)`. Do not silently accept float64.
6. **Preserve decode regressions.** Five fixtures compare PCM hashes rounded to 1e-4;
   MP3 checks dimensions/rate and RMS tolerance. These checks do not detect every sample
   change or prove cross-CPU identity. Verify against ffmpeg before updating expectations.
7. **Gate:** nothing is pushed, tagged, published or deployed without the maintainer's
   explicit approval after seeing the report.

## Adding a feature

1. Write the failing test first (`cargo test` for logic, `tests/smoke.py` for the Python
   surface).
2. Implement in the module that owns the concern; do not add a layer.
3. Add the stub to `sound_rs.pyi` and the section to `docs/_content.py`.
4. `python3 docs/_build.py` and `python docs/check_api.py`.
5. Update `README.md` features and the changelog page.
6. Run the full command list above.

## Style

- Match the surrounding code; comments explain *why*, not *what*.
- No new crates or system dependencies without a written reason. Linux already requires
  ALSA (runtime for wheels, development headers for source builds).
- Prefer deleting code to adding it. A helper used once is not a helper.

## Traps seen before

- `pyo3` methods called from Rust need `pub(crate)`; console scripts need `sys.argv`, not
  `std::env::args()`.
- cpal streams are `!Send`; `Player`/`Recorder` are `unsendable` pyclasses.
- Container seeks land on packet boundaries; `Stream::seek` records the difference and drops
  it on the next packet. Do not "simplify" that away.
- Full-file MP3 goes through minimp3; MP3 windows, streaming and `load_bytes` use Symphonia.
  The paths can differ in samples/alignment. AAC bytes already read MP4 edit-list priming.
- `pub(crate)` inside `#[pymethods]` is still exposed to Python. Check the generated API,
  not just Rust visibility; keep `Player.play_samples` documented and typed.
