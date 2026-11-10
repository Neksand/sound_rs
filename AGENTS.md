# AGENTS.md

Guidance for coding agents working on `sound_rs`. Humans: see `README.md`.

## What this is

A Python audio library written in Rust (PyO3 + maturin): playback, streaming, recording,
decoding to NumPy, and now WAV writing. No ffmpeg, no system decoders.

## Layout

| Path | Owns |
|---|---|
| `src/lib.rs` | the PyO3 surface: every `#[pyfunction]`, the module, `SoundRsError` |
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
python3 docs/_build.py                       # regenerate docs, llms.txt, .md pages
python bench/run.py --compare                # benchmark
```

CI runs all of the above on Linux, Windows and macOS. If a check fails locally, it fails there.

## Invariants — do not break

1. **Every public name is in three places:** the Rust `#[pyfunction]`, `sound_rs.pyi`, and
   `docs/api.html` (via `docs/_content.py`). `docs/check_api.py` enforces the last two.
2. **Every failure is `SoundRsError` with a `.code`** from `io`, `format`, `device`,
   `stalled`, `unsupported`, `usage`. No panic may cross the FFI boundary.
3. **The audio callback never allocates and never touches Python.** `Core::fill` only copies
   from an `Arc<Vec<f32>>` or an `rtrb` ring.
4. **The GIL is released around blocking work** (decode, playback, record, wait) with
   `py.allow_threads`.
5. **Arrays are float32.** `load`/`write` use `(frames, channels)`; `play_array` takes
   `(channels, frames)`. Do not silently accept float64.
6. **Decode is deterministic.** If a change moves a sample, `tests/decode_regression.py`
   fails — verify against ffmpeg before updating the hashes.
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
- No new crates without a written reason. No system dependencies, ever.
- Prefer deleting code to adding it. A helper used once is not a helper.

## Traps seen before

- `pyo3` methods called from Rust need `pub(crate)`; console scripts need `sys.argv`, not
  `std::env::args()`.
- cpal streams are `!Send`; `Player`/`Recorder` are `unsendable` pyclasses.
- Container seeks land on packet boundaries; `Stream::seek` records the difference and drops
  it on the next packet. Do not "simplify" that away.
- MP3 goes through minimp3, everything else through symphonia. `load_bytes` cannot use the
  fast path because minimp3 wants a file.
