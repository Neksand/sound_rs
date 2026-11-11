# Engineering history audit

Audit date: 2026-11-11. Repository baseline: `c1898635c49534e51437ad853951477094ed0665` on `master` (`v0.4.0`). This report describes evidence retrieved during this audit, not a claim that historical checks were rerun locally.

## Current repository and release state

- Default branch: `master`.
- Cargo package version: `0.4.0`; Python version is dynamic and supplied by maturin.
- Latest GitHub release: [`v0.4.0`](https://github.com/Neksand/sound_rs/releases/tag/v0.4.0), published 2026-11-10.
- [PyPI JSON metadata](https://pypi.org/pypi/sound-rs/json) reports `0.4.0`, Python `>=3.10`, five non-yanked `cp310-abi3` wheels and one source distribution. These were uploaded on 2026-11-10. The wheels target macOS x86_64/arm64, manylinux x86_64/aarch64 and Windows amd64. Availability does not prove hardware correctness.
- Remote branches are `master`, `fix/audio-pipeline` and `release/0.3.0`. Both development branches are ancestors of `master`; neither contains unmerged work at the audit baseline.
- Git tags: `v0.1.7`, `v0.1.8`, `v0.2.0`, `v0.2.1`, `v0.2.2`, `v0.3.0`, `v0.3.1`, `v0.3.2`, `v0.4.0`.
- GitHub exposes seven releases, from `v0.2.0` through `v0.4.0`. A tag does not necessarily have a GitHub Release record. PyPI retains versions from `0.1.1` through `0.4.0`.
- One pull request exists in the retrieved history: [PR #1](https://github.com/Neksand/sound_rs/pull/1), merged. No open PRs or issue records were returned by repository-scoped searches.
- Repository rulesets returned an empty list. The branch-protection endpoint returned `403 Resource not accessible by integration`, so branch protection is **unknown**, not absent.

The web search index returned an obsolete PyPI page showing 0.1.5. The direct live JSON response above supersedes that cached page for the release assessment.

## Milestones and architecture evolution

| Period / release | Evidence | Change and lasting decision |
| --- | --- | --- |
| 2025-10-18 | `e88fff5`, `f1ac7dc` | Initial library and format expansion. |
| 2025-11-25 | `71bb89a` | First benchmark work. |
| 2026-11-03 | `b74b280`, `8638456` | Playback pipeline repair, removal of tokio/dead code, faster decoding and AAC/M4A support. Existing `fix/audio-pipeline` branch is historical. |
| 0.1.7 / 0.1.8 | `9e98725`, `dabe445`, `fa6e228`, `5e9b0b9` | Cross-platform wheels; block-copy playback, capped allocation, drain/error handling; gapless trimming, header-only metadata and MP4 edit-list priming. |
| 0.2.0 | `7476411` | Persistent Player, background incremental decoding into a bounded ring, rubato FFT resampling, minimp3 fast path, parallel decoding, devices and typed API. |
| 0.2.1 / 0.2.2 | `1042eaa`, `99cb1f5`, `92d9281` | Package metadata/version, command-line entry points and generated static documentation site. |
| 0.3.0 | `fbb52bf` through `c78c752`; PR #1 | Dataset windows/rate/mono, byte decoding, Recorder, queue/feed streams, stable error codes, underrun/latency diagnostics, properties/mutation tests, cross-platform CI and abi3 wheels. |
| 0.3.1 | `3f4b0f7` | Documentation-only release to update the PyPI long description. |
| 0.3.2 | `535d7e2` | AAC priming from byte buffers and exact upper frame bound for timed recording; documentation synchronized. |
| 0.4.0 | `d4db7b3`, `c189863` | WAV output in PCM16/PCM24/float32, `Recorder.record_to`, strided arrays, AGENTS.md and generated machine-readable documentation. Follow-up formatting commit establishes the audited head. |

The engine remains one Rust crate with separate audio, player, recorder, writer and CLI modules. Python is exposed with PyO3; no alternative engine or independent Rust public API was introduced. Although `rlib` is a build target, that alone does not establish a stable, supported Rust API.

## PR #1 and review history

[PR #1](https://github.com/Neksand/sound_rs/pull/1), “0.3.0: professional reliability + datasets, recording, queue, feed,” targeted `master` from `release/0.3.0`. It contained 15 commits and changed 51 files (+4,741/-323). GitHub reports it merged on 2026-11-05 at commit `f2dc2123a1fb913598184abc62f1e4a36bff6be2`.

The PR description asked to hold the branch until CI and the report were approved. The retrieved API records contain zero discussion comments, zero inline comments, zero review submissions and zero review threads. There is therefore no recorded GitHub review conversation to reinterpret or resolve. Review findings are instead documented in commit `46c7288` and `docs/reports/0.3.0-report.md`.

That commit addressed:

- Queue handover replacing the ring before the preceding track drained, cutting up to one second of audio.
- A fed stream spinning indefinitely on a full ring; a five-second timeout and producer teardown were added.
- State transitions allowing `wait()` to return between queued tracks.
- Seeking after EOF leaving the decoder parked.
- Windowed decoding and seek accounting for encoder delay/trailing padding.
- False underruns at fed-stream completion, latency observability and stable stall error codes.
- Multi-channel mono conversion, Recorder context management/non-finite duration rejection and valid empty load windows.

These are established behaviors to preserve and extend with precise regression tests. A new implementation must not recreate the 0.3.0 queue, feed, recorder or dataset APIs as if they were missing.

## Previously deferred work: actual status

The 0.3.0 report contains historical statements, including a pre-publication heading and a later addendum. Its deferred list is not a current backlog:

| Historical issue | Current status at audit baseline |
| --- | --- |
| Recorder retained a large buffer after overflow | Fixed during 0.3.0 follow-up (`e4a69d8`), as the report addendum says. |
| Timed recording could overshoot by one device buffer | Fixed in 0.3.2 by truncating to the requested maximum frame count. Input start latency can still yield fewer frames. |
| `load_bytes` did not trim AAC priming from MP4 edit lists | Fixed in 0.3.2. File and bytes fixture lengths are expected to match. |
| MP3 sample offset against FFmpeg/libmpeg123 | Still documented as accepted; length/RMS checks are not proof of sample alignment. |
| Real hardware exercised only on macOS | Still a limitation of historical evidence; CI compilation and headless tests do not substitute for Linux/Windows device tests. |

The old implementation plan also proposes standalone cargo-fuzz/ASan jobs that were not present in the current workflow. The actual implementation uses mutation tests within `cargo test`, not a continuously fuzzed standalone harness. Its old unchecked boxes should not be treated as proof that a feature is absent or as instructions to install unavailable skills.

## Historical performance evidence

Version 0.2.0 release notes report improvements from persistent streams and incremental decoding: about 43 MB versus 255 MB peak RSS for a ten-minute FLAC and 8.8 ms versus 492 ms time to first sound on the original test machine. The 0.3.0 report records 11.3 ms streaming startup, 43 MB peak RSS, zero observed underruns and 1.00x playback real-time factor; 60-second stereo decode medians were WAV 129.9 ms, FLAC 265.0 ms, MP3 240.8 ms and M4A 348.8 ms.

These are **historical author measurements**, stored in `docs/reports/0.3.0-report.md`, `bench/results.json` and release notes. They have not been independently reproduced in this history audit. Differences between machines, workloads and versions prevent treating them as current benchmark results or universal latency guarantees. Commits `d13a834` and `3f4b0f7` explicitly reconciled inconsistent documentation/PyPI performance claims.

## CI and release history

The complete returned Actions inventory contained 38 runs. Selected failures and recoveries were inspected through job metadata and logs:

| Run / commit | Observation | Resolution or implication |
| --- | --- | --- |
| [41701970504](https://github.com/Neksand/sound_rs/actions/runs/41701970504), `d9b3aa1` | Linux wheel checks failed installing virtual package `libasound2`. | `fac0062` added the appropriate ALSA package fallback. |
| [42015345427](https://github.com/Neksand/sound_rs/actions/runs/42015345427), `e4a69d8` | Python job failed because `maturin develop` requires a virtual environment. | `b7816fb` changed CI to build then install a wheel. |
| [42015691174](https://github.com/Neksand/sound_rs/actions/runs/42015691174), `b7816fb` | MP3 regression hash differed while the other five fixtures passed. | Quantization was initially relaxed in `6286485`. |
| [42016166316](https://github.com/Neksand/sound_rs/actions/runs/42016166316), `6286485` | MP3 hash still differed across CPU decoder paths. | `63cf7f1` uses exact dimensions/rate and RMS tolerance for MP3, retaining hashes for other fixtures. |
| [42016859766](https://github.com/Neksand/sound_rs/actions/runs/42016859766), `63cf7f1` | Aarch64 cross compilation could not configure ALSA pkg-config; other wheel smoke checks tried to install the `dist/` directory itself. | `f2dc212` uses a native ARM runner and `--find-links dist sound_rs`. |
| [42017949033](https://github.com/Neksand/sound_rs/actions/runs/42017949033), `f2dc212` | PR CI succeeded. | Historical green PR baseline. |
| [42017952998](https://github.com/Neksand/sound_rs/actions/runs/42017952998), `f2dc212` | Release dispatch succeeded. | Five-platform build/import evidence; publishing intentionally skipped on branch dispatch. |
| [42870361690](https://github.com/Neksand/sound_rs/actions/runs/42870361690), `d4db7b3` | Lint failed on two writer formatting differences. Linux/Windows Rust tests, Python and docs passed; macOS was cancelled. | `c189863` formats the writer. |
| [42870845954](https://github.com/Neksand/sound_rs/actions/runs/42870845954), `c189863` | Overall cancelled; lint, Linux/Windows tests, Python and docs succeeded. macOS job was cancelled. | Audited baseline did not have a fully green cross-platform CI run. |
| [42871469961](https://github.com/Neksand/sound_rs/actions/runs/42871469961), `v0.4.0` | Tests, sdist and four wheel jobs succeeded; macOS ARM cancelled; publish skipped. | Release notes describe a runner outage and local macOS wheel/manual upload. PyPI confirms files exist but does not validate their build provenance. |

The GitHub job records prove cancellation, not its external cause. The runner-outage explanation is the maintainer's release-note account.

At audited baseline `c189863`, CI ran Rust tests on Ubuntu, Windows and macOS; Python package build/smoke/decode/API/statistics checks ran **only on Ubuntu**. The release matrix built five wheels but its smoke step only imported the module; it did not run the full Python numerical suite on each platform. The baseline statement in AGENTS.md that all listed commands ran across all three OSes was therefore broader than that workflow. The proposed validation workstream expands these checks; its new configuration is separate from this historical run evidence.

## Constraints and opportunities for the next PRs

1. Preserve published Python conventions, the minimp3/file versus Symphonia/bytes split, seek trimming, callback constraints, stable error codes and all existing regression expectations.
2. Focus new reliability work on demonstrated current defects, not on already-resolved historical lists. Inspect WAV length/header arithmetic, input validation, player state transitions and bounded-memory guarantees before expanding features.
3. Strengthen wheel validation with device-independent numerical/API tests across platforms and the supported Python ABI endpoints; preserve the corrected native ARM build and wheel-install paths.
4. Keep generated documentation reproducible and make historical reports clearly historical. Correct present-tense claims where the actual CI or implementation differs.
5. Release workflow still uses a PyPI token and an environment gate. Trusted publishing/provenance is an opportunity requiring a separate maintainer account configuration; do not change secrets or publish during this engineering session.
6. Hardware disconnection/recovery, long-running input/output and callback timing need explicit device-backed testing before strong real-time or production claims.
7. Do not infer permission or protection settings from the inaccessible branch-protection endpoint. Do not merge, tag or release as part of PR preparation.

## Audit method and boundaries

Inspected the complete local Git history (46 commits), all remote branch tips and tags, historical implementation/report files, current workflow and package metadata, GitHub releases, repository-scoped PR/issue search, PR #1 comments/reviews/threads, all 38 returned Actions runs, failed-job metadata and representative failure logs, current-head CI and v0.4.0 release jobs, rulesets and attempted branch-protection access. Live PyPI JSON was fetched separately to avoid stale search-index metadata. No remote state was mutated by this audit.
