# Engineering audit

Date: 2026-11-11. Baseline: `c1898635c49534e51437ad853951477094ed0665`, version 0.4.0.
This is a source and local-validation audit, not a certification of real-time or production safety.
See [history](ENGINEERING_HISTORY.md), [implementation plan](../IMPLEMENTATION_PLAN.md),
[PR roadmap](../PR_ROADMAP.md), and [final report](FINAL_ENGINEERING_REPORT.md).

## Executive assessment

The repository already contains a useful audio engine: incremental file playback, a persistent
Player, bounded playback rings, recording, parallel/windowed decoding, NumPy conversion and
three WAV subtypes. Reimplementing these APIs or adding an effects framework would miss the
highest-value work. The initial tests all pass, but targeted reproductions expose media-boundary,
playback-state, capture-callback and streaming-resampler defects.

The improvement program therefore preserves the single-crate design, codecs, array layouts,
published version and regression expectations. It prioritizes correctness and reproducible
validation before additional DSP, editing, synthesis, hot-swap or a public Rust API.

## Architecture and ownership

| Module | Responsibility and constraints |
| --- | --- |
| `lib.rs` | PyO3 boundary, error conversion, NumPy ownership, blocking-work GIL release |
| `audio.rs` | Symphonia decoding, minimp3 full-file MP3 path, gapless trimming, windows, channel conversion, rubato resampling |
| `player.rs` | CPAL output stream on its Python owner thread; decoder worker and one-second `rtrb` ring; source controls and queue |
| `recorder.rs` | CPAL input stream on its owner thread; baseline callback appends to shared Vec; proposed collector moves growth off callback |
| `writer.rs` | PCM16/PCM24/float32 WAV serialization and file output |
| `cli.rs` | Small command parser and Python console entry points |
| `docs/_content.py`, `_build.py` | Documentation source and reproducible HTML, Markdown, search, sitemap and llms output |

`Player` and `Recorder` are unsendable PyO3 classes because CPAL streams are not Send.
No unsafe Send/Sync implementation is justified. An `rlib` artifact does not make the private
engine modules a supported native Rust API. All new work keeps these ownership boundaries.

## Baseline feature and API matrix

“Covered” below describes the existing tests, not complete correctness. Hardware tests are optional.

| API or feature | Exists | Baseline coverage | Documentation / finding |
| --- | --- | --- | --- |
| `play_audio`, `Player.play_file` | Yes | Optional hardware smoke; decoder unit tests | Streaming already implemented; state/drain defects found |
| `Player.play_array`, `play_array` | Yes | Optional hardware smoke | Channels-first arrays; unchecked channels/rates |
| `Player.play_samples` | Yes, Python-exposed | No explicit boundary tests | Missing from stub and API reference despite `#[pymethods]` exposure |
| pause/resume/stop/position/duration/volume | Yes | Four Core tests, optional hardware smoke | Paused seek and lifecycle need new tests |
| seek | Yes | Decoder seek tests | Player-level overflow/state issues not covered |
| enqueue/queue/clear_queue | Yes | Optional file-queue smoke | Only file decoder consumes the queue; other source modes can stall |
| begin_feed/feed/end_feed | Yes | Optional five-block smoke | Absolute timeout, empty feed and drain defects |
| underruns/latency | Yes | One Core test, optional hardware | Observations, not latency guarantees |
| `Recorder`, context management | Yes | Arithmetic cap test, optional recording | Callback allocates/locks; context methods missing from stub |
| start/stop/record | Yes | Optional hardware | Backend errors only stderr; claimed cap bytes inconsistent |
| `Recorder.record_to` | Yes | Writer tested separately | Runtime subtype default missing; records in memory, then writes |
| `write` | Yes | Six Rust tests, Python PCM24 round-trip | RIFF padding/header arithmetic and error-classification bugs |
| `load` | Yes | Smoke, unit, properties and fixture regressions | Window/rate/mono options omitted in API signature; invalid times accepted |
| `load_bytes` | Yes | Bytes/file WAV and AAC checks, mutation tests | AAC priming already fixed in 0.3.2; MP3 path differs from file load |
| `load_many` | Yes | Batch order/shape smoke | `sample_rate=0` accepted; retains entire batch in RAM |
| `get_audio_info` | Yes | Header tests/smoke | May decode when header incomplete; not always constant-time |
| `is_supported_format` | Yes | Smoke | Probe success does not validate all packets or decoder support |
| device enumeration | Yes | Smoke | Device name is selection key; index is display-only |
| `play_sine_wave` | Yes | Optional hardware smoke | Invalid floats/rate may cause bad arithmetic/allocation |
| CLI and `__version__` | Yes | Seven parser tests, console smoke | Non-finite volume accepted; info may block under GIL |
| editing, FFT/STFT analysis, effects | No | None | Roadmap only; do not advertise as implemented |
| synthesis beyond playback sine | No | None | Defer until numeric API design and demand justify it |
| automatic device recovery | No | None | Reporting a fault is not reconnection/hot-swap |

## Format capabilities

| Format | Decode/play | Incremental file stream | Encoding | Evidence / limits |
| --- | --- | --- | --- | --- |
| PCM WAV | Yes | Yes | PCM16, PCM24, float32 | WAV/24-bit/six-channel fixtures and round-trips; classic RIFF size limits |
| FLAC | Yes | Yes | No | Fixture hash; built-in Symphonia codec |
| MP3 | Yes | Yes | No | Full-file minimp3; bytes/windows/streaming use Symphonia; exact length/RMS checked, samples differ across paths |
| AAC in MP4/M4A | Yes | Yes | No | Fixture hash, edit-list priming; residual tail padding can remain |
| Ogg/Vorbis | Implemented codec path | Yes | No | Dependency feature enabled; no committed format-specific golden fixture |
| Raw AAC / other containers | Backend-dependent | Backend-dependent | No | Not established by an extension or cargo feature alone; add dedicated fixtures before broad claims |

Metadata and seek depend on the container. An advertised extension is not proof that every codec,
channel layout or damaged file in that container is supported. Existing accepted truncation behavior
returns a decodable prefix; strict completeness validation is not presently a general guarantee.

## Reproduced findings and priorities

| ID | Priority | Finding | Reproduction / planned correction |
| --- | --- | --- | --- |
| M1 | P1 | Untrusted MP4 extended lengths overflow `offset + size` | A second box with `u64::MAX` size panics; compare against remaining bytes before arithmetic |
| M2 | P1 | Metadata can request huge upfront reservation | Extreme declared rate/channels requests ~1.69e16 samples; add absolute 32 MiB reservation ceiling |
| W1 | P1 | WAV arithmetic overflows before validation | 3,000 float channels overflow intermediate u16 product; huge rate overflows byte rate; checked header arithmetic |
| W2 | P2 | Odd PCM24 RIFF size excludes pad byte | One mono sample produces 48 bytes but RIFF declares 47; include pad in RIFF size only |
| W3 | P2 | Write I/O errors classified `format`; invalid samples accepted | Missing destination directory and NaN input; explicit usage/io errors, validation before file touch |
| A1 | P1 | NumPy channel count truncates to u16 | 65,536 channels become zero; check before copying and use checked readonly borrow guards |
| A2 | P2 | Invalid time/rate/sine values not consistently rejected | `load_many([], sample_rate=0)` succeeds; NaN times bypass intended logic; validate at boundary |
| P1 | P1 | Playback drain counts silence instead of written prefix | Full/partial Core callback tests; use backend latency plus actual audio duration |
| P2 | P1 | Empty closed feed never completes; final feed skips drain | Simulated ring/Core tests without devices |
| P3 | P1 | Healthy feed longer than five seconds times out | Small-ring producer/consumer with continuous progress; measure stall interval |
| P4 | P1 | Paused/file/queued seek corrupts state or overflows | Deterministic decoder/source-state tests; preserve pause/queue and check seek arithmetic |
| P5 | P2 | Wrong-mode queue/feed operations hang or end unrelated source | Explicit source-state checks and stable usage errors |
| R1 | P1 | Recorder callback locks and grows Vec | Source inspection, no preallocation despite module claim; bounded ring and collector |
| R2 | P1 | Recording backend faults do not invalidate take | Atomic fault propagation and simulated overflow/disconnect/shutdown tests |
| S1 | P1 | Streaming resampler loses tail or adds padded frames | 1 frame 44.1k→48k yields 960 instead of 2; 1024 yields 800 instead of 1115; account input/output and drain delay |
| Q1 | P2 | Regression script passes with missing fixtures | Enumerate expected fixtures, fail on missing input |
| Q2 | P2 | API checker only searches top-level stub names as substrings | Compare Rust registrations, class methods, AST stub and scoped API docs |
| Q3 | P2 | Python numerical suite runs only on Ubuntu | Expand headless platform/ABI matrix; release wheels need more than import |

## Reliability, memory and concurrency

Playback uses a bounded ring but full `load`/`load_many` intentionally retain complete results.
The reservation cap is only a defense against malformed metadata, not a total decode-memory limit.
WAV output still builds a complete encoded buffer; incremental file writing is future work.
Recording's historical limit is 230 million samples, approximately 920 MB of float32 payload,
not the 230 MB stated in old error text. Collector storage must enforce that limit explicitly.

No change may allocate, wait on Python or perform I/O in an audio data callback. The baseline
player takes a blocking mutex in its callback; the playback workstream changes this to
`try_lock` and silence on contention. This is not a proof of hard real-time safety.
Stream ownership remains on the creating thread. Decoder and collector lifetimes need stop/drop
tests, and blocking joins/drain waits should release the GIL where Python ownership permits.

## Audio quality and performance

Baseline six-fixture regression expectations remain the reference. The streaming-resampler fix
changes playback at unequal rates intentionally, verified numerically against a full rubato pass
with the same filter configuration, packet partition tests and tail impulses. It does not update
full-file fixture hashes. MP3 keeps the known approximately 47-frame alignment difference against
FFmpeg; an independent comparison reproduced it, but a robust fix needs nonperiodic mono/stereo
fixtures and explicit Xing-frame handling. No speculative hash update is justified.

The existing benchmark harness is reused. New measurements distinguish cold-process import +
decode cost, CPU and peak RSS; they are not directly comparable to all historic README workloads.
No speedup claim is made without repeatable evidence. Device latency, real-time factor and hardware
recording performance are not established by headless tests.

## Packaging, CI and documentation

Cargo 0.4.0 supplies the dynamic Python version. Five abi3 wheels exist on PyPI; current-head CI and
v0.4.0 release Actions were partly cancelled. Historical reports contain superseded limitations.
No version bump, merge, release, credential change or deployment is part of this PR program.

Documentation drift includes an overview claiming no recording, incomplete load signatures,
untyped Recorder context methods, an undocumented Python-exposed play_samples method, and claims
that no system libraries are needed despite the Linux ALSA runtime requirement. Generated HTML
must be rebuilt from source and checked with Markdown/search/sitemap/llms outputs together.

## Security and dependency review

Trust boundaries are external media headers, sample arrays, file destinations and native audio
backends. The code must validate lengths before arithmetic/allocation and keep failures within
the Python error model. Mutation testing is valuable evidence, not an exhaustive security proof.

The unchanged lock at baseline contains RustSec alerts for PyO3 0.25.1
([RUSTSEC-2026-0176](https://rustsec.org/advisories/RUSTSEC-2026-0176.html),
[RUSTSEC-2026-0177](https://rustsec.org/advisories/RUSTSEC-2026-0177.html)) and
slice-ring-buffer 0.3.4 through minimp3
([RUSTSEC-2025-0044](https://rustsec.org/advisories/RUSTSEC-2025-0044.html)).
The advisory review used the official RustSec database; cargo-audit was not installed.
No direct application use of the named PyO3 APIs was found, but that does not establish immunity.
The minimp3 transitive alert has no published patched version; do not equate unchanged regressions
with security clearance. A paired PyO3/NumPy upgrade is evaluated separately in the final report.

## Baseline validation

Environment: macOS 27.2 arm64, CPython 3.14.7, Rust 1.99.1, Cargo 1.99.1,
maturin 1.16.0, NumPy 2.5.4. A workspace virtual environment isolates Python tooling.

| Command | Baseline outcome |
| --- | --- |
| `cargo fmt --all -- --check` | Passed |
| `cargo clippy --all-targets -- -D warnings` | Passed |
| `cargo test --all-targets` | 39 passed |
| `maturin develop --release` | Built and installed macOS arm64 abi3 wheel |
| `python tests/smoke.py` | Passed |
| `python tests/decode_regression.py` | All six fixtures passed; expectations unchanged |
| `python docs/check_api.py` | Passed original limited 15-name check |
| `python bench/test_stats.py` | Passed |
| `python3 docs/_build.py` + Git diff | 17 pages; no baseline generated drift |
| full `python bench/run.py --compare` | Not run: includes device playback/soak; headless existing workers measured separately |
| microphone / physical disconnect / Windows or Linux hardware | Not performed locally |

## Recommended evolution and remaining debt

Complete the reproduced correctness fixes and portable tests before expanding the product.
Then address remaining decode resource limits, malformed spec changes, minimp3 dependency risk,
continuous feed resampling and device-backed soak/disconnect tests. Explicit frame-first array
editing functions may follow with numerical semantics and synchronized typing/docs. FFT/STFT,
filters, synthesis, AI adapters, dynamic plugins and a public Rust API require separate scoped
designs; no placeholder modules or mandatory ML dependencies should be introduced.
