# Pull request roadmap

Baseline: `c1898635c49534e51437ad853951477094ed0665`, `master`, version 0.4.0.
Prepared on 2026-11-11. No merge, tag, release or deployment is authorized by this series.

## Review order and branch strategy

The series is stacked deliberately: generated documentation, Python stubs and CI evolve across
the same workstreams, and the PyO3 migration must include the newly added blocking-work calls.
Each PR diff isolates one concern against the preceding validated tree. Merge in order; after
merging a base PR, retarget its successor to `master` before merging it. Do not squash the entire
program into one PR. If a base PR is squash- or rebase-merged, rebase its successors onto the new
master before retargeting so earlier commits are not reintroduced. The combined tree is validated
separately from each original workstream.

| Order | Title | Head branch | Initial base | Status / URL |
| --- | --- | --- | --- | --- |
| 1 | fix(media): validate container sizes and Python audio inputs | `fix/media-boundary-validation` | `master` | [Open PR #2](https://github.com/Neksand/sound_rs/pull/2) |
| 2 | fix(player): preserve source state and drain playback correctly | `fix/player-stream-lifecycle` | `fix/media-boundary-validation` | [Open PR #3](https://github.com/Neksand/sound_rs/pull/3) |
| 3 | fix(recorder): collect capture through a bounded ring | `fix/recorder-bounded-capture` | `fix/player-stream-lifecycle` | [Open PR #4](https://github.com/Neksand/sound_rs/pull/4) |
| 4 | fix(streaming): drain resampler tail with exact frame accounting | `fix/stream-resampler-tail` | `fix/recorder-bounded-capture` | [Open PR #5](https://github.com/Neksand/sound_rs/pull/5) |
| 5 | test: validate APIs, generated docs and portable wheels | `test/portable-validation` | `fix/stream-resampler-tail` | [Open PR #6](https://github.com/Neksand/sound_rs/pull/6) |
| 6 | fix(deps): upgrade PyO3 bindings past known advisory ranges | `fix/dependency-advisories` | `test/portable-validation` | [Open PR #7](https://github.com/Neksand/sound_rs/pull/7) |

## 1. Media and Python inputs

- **Problem/motivation:** malformed lengths or dimensions can overflow, produce invalid WAV
  headers or select incorrect errors; invalid time/rate inputs are inconsistent.
- **Scope/files:** `audio.rs`, `writer.rs`, `lib.rs`, `cli.rs`, validation tests, stubs, relevant
  API/guarantee/changelog pages, README and CI discovery.
- **API/compatibility:** names/layouts unchanged; invalid times, non-finite samples and
  unrepresentable dimensions now fail with usage; actual WAV I/O errors become io.
- **Acceptance/tests:** pre-fix regressions fail; 47 standalone Rust tests, 10 Python boundary
  tests, Clippy/fmt, smoke and six decode regressions pass; WAV rejection preserves destination.
- **Benchmarks:** compare full-decode cold-process medians/RSS; no speedup claim.
- **Risk:** callers relying on invalid/clamped inputs need correction; total decode memory is
  still proportional to content. No new crates. Dependencies: baseline only.

## 2. Playback lifecycle

- **Problem/motivation:** final audio can end early; empty/long feeds fail; seek and queue
  transitions can lose pause intent or play the wrong file; callback mutex can block.
- **Scope/files:** `player.rs`, small saturating seek arithmetic in `audio.rs`, hardware-free
  callback/ring/decoder tests, Player guide and changelog/README.
- **API/compatibility:** no names removed. Invalid modes/values fail promptly. Queueing requires
  a file source; stopped sources require restart. Paused wait returns with queue intact.
- **Acceptance/tests:** 65 standalone Rust tests, including real decoder-worker simulations,
  precise drain, long feeds, paused full-ring seek, pause-at-EOF race, backend faults and contention.
- **Benchmarks:** no latency/underrun improvement claim without physical devices.
- **Risk:** contention now emits silence instead of waiting; that silence is not included in
  ring-starvation underrun count. Continuous fed resampling/reconnect remain deferred.
- **Dependency:** previous PR provides the validated media/API foundation and shared docs state.

## 3. Recorder

- **Problem/motivation:** baseline callback grows a Vec under a mutex and backend faults only
  reach stderr; safe callback behavior and complete error reporting are required.
- **Scope/files:** `recorder.rs`, capture simulation tests, Recorder stubs, binding smoke,
  recording/guarantee/changelog docs and README.
- **API/compatibility:** preserve 230 M sample cap (920 MB payload); implement documented
  record_to subtype default; context exit propagates capture faults when no prior exception.
- **Acceptance/tests:** 51 standalone Rust tests; ring wrap/overflow, exact limit, first fault,
  final-drain and repeated worker shutdown; isolated release wheel smoke/decode checks.
- **Benchmarks:** structural callback allocation removal; no measured hardware CPU/latency claim.
- **Risk:** desktop backend teardown still requires hardware validation; large takes retain
  memory and record_to writes only after capture. Depends on prior series tree.

## 4. Streaming resampler

- **Problem/motivation:** block-boundary/short inputs lose filter tail or retain padded output.
- **Scope/files:** streaming Resampler in `audio.rs`, numerical/property tests and streaming docs.
- **API/compatibility:** same signatures; unequal-rate playback now produces exactly
  ceil(input frames × target/source rate) and retains the filter tail. Full decode unchanged.
- **Acceptance/tests:** 43 standalone Rust tests; 75 numerical/partition combinations, 64
  property cases, stereo/tail impulses/reset/double drain; baseline regression hashes untouched.
- **Benchmarks:** correctness fix, no claimed acceleration.
- **Risk:** audible output intentionally changes where previously truncated/padded; existing
  silent fallback on resampler construction error remains future work. Depends on prior tree.

## 5. Validation, docs and reports

- **Problem/motivation:** missing fixture files silently pass; API checker omits class members;
  Python numerical tests do not cover every platform; benchmarks can hang or hide failed runs.
- **Scope/files:** decode script, quality tests, API/site checkers, benchmark harness/statistics,
  CI/release workflows, stubs, truthful website/README/AGENTS content and engineering reports.
- **API/compatibility:** documents and types existing Player.play_samples; no new runtime API.
  Benchmark --no-audio and --timeout are additive; incomplete runs no longer yield summaries.
- **Acceptance/tests:** 21 checker/headless regressions, nine benchmark checks, actual headless
  benchmark, generated-site idempotence and links, API registration/member-name consistency; remote matrix
  covers three OSes plus Python 3.10/3.14 endpoints.
- **Risk:** CI configuration is only cross-platform evidence once remote jobs pass; no hardware
  result or publishing configuration is implied. Depends on all preceding behavior/docs.

## 6. PyO3 dependency maintenance

- **Problem/motivation:** PyO3 0.25.1 is in two known RustSec affected version ranges.
- **Scope/files:** Cargo.toml/lock, PyO3 attach/detach calls, explicit module GIL requirement,
  AGENTS/changelog and dependency advisory report.
- **API/compatibility:** PyO3 0.29.3 and numpy Rust crate 0.29.0; keep abi3-py310, GIL-required
  operation, names, array layouts, codec versions and fixtures. No direct crate additions.
- **Acceptance/tests:** isolated migration 39 Rust tests, Clippy/fmt/no-default-feature check,
  release wheel/isolated install and Python decode smoke; rerun complete combined suite.
- **Benchmarks:** paired before/after decode measurements in final report; no performance promise.
- **Risk:** minimp3 → slice-ring-buffer advisory remains unresolved and explicitly documented.
  Depends on previous PRs so every new blocking call is migrated together.

## Submission and CI record

This document initially records local preparation. Actual PR URLs and observed remote CI results
are added only after successful submission. Branches are pushed directly to the original repository
under the owner's authorized temporary session; no fork is created. Nothing is auto-merged.
