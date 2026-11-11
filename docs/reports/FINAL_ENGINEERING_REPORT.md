# Final engineering report

Date: 2026-11-11. Audited baseline: `c1898635c49534e51437ad853951477094ed0665` (0.4.0).
This report describes an implemented, locally validated and submitted PR series. Submission and remote CI
status are recorded separately below and in the [PR roadmap](../PR_ROADMAP.md).

## Executive outcome

sound_rs already had playback, recording, streaming, decoding, queues and WAV writing. The work
strengthens those existing capabilities instead of duplicating them: checked media/array inputs,
correct playback drain and source state, bounded capture callbacks, exact streaming resampling,
broader portable checks, synchronized docs and a compatible Python binding dependency upgrade.

The project version stays 0.4.0. No release, merge or deployment is performed. This is a reliability
and validation milestone, not a claim that every aspirational editing/DSP/AI feature is complete.

## Audit and design decisions

The [audit](ENGINEERING_AUDIT.md) covers architecture, every exposed API, formats, baseline
validation and reproduced findings; [history](ENGINEERING_HISTORY.md) records all remote branch
tips, tags/releases, PR #1, reviews/issues and Actions failures. Existing branches contain no
unmerged work at the baseline. Most architecture is retained: one crate, existing codecs/rtrb,
unsendable CPAL-owning classes and established array layouts.

No new direct dependency crate or runtime system dependency is introduced. The only architectural
addition is a recorder collector thread behind the existing rtrb primitive: allocation belongs off
the input callback. Playback source state gains explicit pause intent so natural completion cannot
override a user's pause during queue handover. A second independent review reproduced this race
after the initial lifecycle tests had passed; the final fix has both deterministic and live-worker
hardware-free regressions.

## Implementation and API status

| Area | Status | Delivered behavior |
| --- | --- | --- |
| MP4/WAV boundaries | Implemented | Bounds before arithmetic/slicing; correct odd padding; checked RIFF/header limits; explicit usage/io errors |
| NumPy/Python validation | Implemented | Checked read borrows and channels, strict dtype retained, valid times/rates/synthesis, write GIL release |
| Playback lifecycle | Implemented | Final written-prefix drain, empty/long feeds, source-mode errors, paused/queued seek, pause-at-EOF preservation, backend fault signal |
| Callback synchronization | Implemented | Output try_lock emits silence on contention; input callback uses bounded ring with atomic fault state |
| Recording | Implemented | Collector storage cap, final drain, overflow/backend fault reporting, context typing and record_to default |
| Streaming resampler | Implemented | Exact converted length, startup trim, tail drain and reset semantics |
| Quality gates | Implemented | Missing-fixture failure, source/stub/member/site checks, headless benchmark/timeouts, portable Python/release validation |
| PyO3 dependency remediation | Implemented | PyO3 0.29.3/numpy 0.29 with explicit GIL requirement and stable ABI |
| Editing/analysis/effects/new synthesis | Not implemented | Separate design milestones with acceptance criteria; no placeholder APIs |
| Device reconnect/hot-swap | Not implemented | Failures are reported; recovery and real hardware validation remain separate |
| Continuous fed resampling/incremental recording | Not implemented | Existing per-feed resampling and record-then-write limitations remain documented |
| Native public Rust/plugin/AI framework | Not implemented | No unsupported ABI or mandatory ML dependencies introduced |

No public callable is removed or deprecated. Player.play_samples was already exposed by PyO3;
it is now typed/documented, not a newly invented feature. Recorder.record_to now matches its
documented default. Invalid/non-finite inputs that were silently accepted or could panic now
raise usage errors; Python type conversion still raises TypeError/OverflowError where appropriate.
Finite out-of-range WAV samples retain existing clipping behavior. File/bytes MP3 paths retain
their existing sample differences. Decoder fixture expectations were not edited.

The type and documentation surface stays synchronized. CPAL streams remain on their creating
thread, without unsafe Send/Sync assertions. Final drain waits, capture collector joins, writing
and decoding release the GIL; some control paths and native backend teardown still hold it.

## Validation evidence

Local environment: Apple M5, macOS 27.2 arm64, CPython 3.14.7, Rust/Cargo 1.99.1,
maturin 1.16.0, NumPy 2.5.4. Python tooling uses a workspace virtual environment.

| Check | Executed command / evidence | Outcome |
| --- | --- | --- |
| Baseline Rust | cargo test --all-targets | 39 passed |
| Final combined Rust | cargo test --all-targets | 89 passed |
| Formatting | cargo fmt --all -- --check | Passed on workstreams; final stack rechecked before submission |
| Lints | cargo clippy --all-targets -- -D warnings | Passed |
| Optional feature build | cargo check --no-default-features | Passed with upgraded bindings |
| Extension | maturin develop --release | Built/installed macOS arm64 cp310-abi3 wheel |
| Isolated migration/recorder wheels | maturin build --release plus pip install into separate target | Passed |
| Python smoke and CLI | python tests/smoke.py | Passed |
| Boundary/checker regressions | python -m unittest discover -s tests -p 'test_*.py' -v | 31 passed (10 boundary, 21 quality) |
| Decoder regressions | python tests/decode_regression.py | Six fixtures passed, expectations unchanged |
| Benchmark statistics/harness | python bench/test_stats.py | Nine checks passed |
| API consistency | python docs/check_api.py | 16 module names and 36 class members checked |
| Site integrity | python docs/check_site.py | Links, anchors, Markdown, search, sitemap, llms checked |
| Site reproducibility | python3 docs/_build.py, repeat and compare Git diff | 18 generated pages checked |
| Workflow syntax | Ruby YAML parser for CI/release | Passed; actionlint unavailable |
| Full device benchmark | python bench/run.py --compare | Not run; contains audio/soak scenarios |
| Headless benchmark | python bench/run.py --compare --no-audio --reps 3 --timeout 60 | Completed; hardware scenarios explicitly skipped |
| Real speaker/microphone, disconnect, Windows/Linux hardware | Not performed locally | Not claimed passed |
| cargo audit | Tool unavailable | Official RustSec database manually reviewed instead |

The standalone workstreams passed 47 media, 65 Player, 51 Recorder, 43 resampler and 39 dependency
Rust tests before composition. Combined tests cover properties, mutation cases, state machines and
actual decoder threads driven by simulated callbacks. Tests reproduced the targeted old failures
before their fixes. Hardware independence does not establish desktop backend behavior or latency.

## Performance evidence

The existing bench/run.py worker/summary functions were reused for paired measurements: 60-second
44.1 kHz stereo corpus, one warmup, five fresh-process repetitions per scenario. Each decode timing
includes import/startup. Baseline uses a retained original extension, final uses the combined
extension; raw data are [before](benchmark-before.json) and [after](benchmark-after.json).

| Format | Before median | After median | After p95 | After p99 | After peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| WAV | 26.42 ms | 27.10 ms | 29.41 ms | 29.81 ms | 53.72 MiB |
| FLAC | 38.26 ms | 35.82 ms | 36.08 ms | 36.08 ms | 53.67 MiB |
| AAC/M4A | 43.79 ms | 43.52 ms | 45.17 ms | 45.35 ms | 55.80 MiB |
| MP3 | 34.58 ms | 34.94 ms | 35.13 ms | 35.16 ms | 53.64 MiB |

These are small diagnostic samples on one machine without CPU affinity/thermal isolation. Medians
vary slightly; no speedup or statistically established regression claim is justified. The
work optimizes correctness and callback allocation behavior, not decoder algorithms. Process RSS
includes interpreter, extension and audio buffers. The separate full headless 3s/600s run is in
[benchmark-headless.json](benchmark-headless.json); it is a different workload and must not be
compared directly with the table or historical README scores. Hardware startup, underruns,
recording CPU and real-time factor were not measured in this session.

## Documentation and contributor experience

Updated API signatures/options/errors, existing play_samples exposure, Recorder context methods,
source-state/feed/seek/drain behavior, capture limits and resampler semantics. Corrected inaccurate
capability, Linux dependency, memory, metadata and threading claims. Historical benchmark results
remain labeled as historical; contradictory duplicated README detail is removed.

All website changes originate in docs/_content.py or _build.py, then regenerate HTML/Markdown and
search data. llms.txt, sitemap and anchors are checked, not manually patched. AGENTS explains actual
CI and callback/error/thread conventions. New audit, history, implementation, PR and final reports
separate implemented, deferred and unverified work. No release history or version is invented.

## Security and remaining risks

PyO3 upgrade removes the two affected RustSec version ranges. The review compared 1,426 official
crate advisories against 202 baseline lock entries, not Python/system packages or unpublished
issues. The unresolved minimp3 → slice-ring-buffer 0.3.4 advisory
[RUSTSEC-2025-0044](https://rustsec.org/advisories/RUSTSEC-2025-0044.html) remains a release risk;
there is no published patched upstream version at review time. No exploitability proof or
complete security clearance is claimed. See the dependency PR's detailed advisory report.

Other priority work: malformed decoder spec/track changes and explicit total decode budgets;
the approximately 47-frame MP3 alignment difference (requires nonperiodic mono/stereo fixtures
before changes); stateful fed resampling; true incremental file recording; cross-platform physical
device soak/disconnect tests. Callback contention creates silence and is not reflected in the
ring-starvation underrun count. The recorder cap is audio payload, not total process RSS.

## Release-readiness assessment

| Use | Assessment |
| --- | --- |
| Development | Suitable for review/testing, with local validation and explicit limitations |
| Experimental trusted-media applications | Improved candidate; validate target device/backend and resource limits |
| Production / arbitrary hostile media | Not certified; unresolved native dependency advisory and hardware gaps remain |
| PyPI release | Not performed; maintainer must resolve/accept risks and verify full wheel/device matrix |
| Stable 1.0 | Not ready based on this work alone; API/state/resource/security guarantees need broader evidence |

## PR submission and remote CI

All six PRs were submitted to `Neksand/sound_rs` as `Neksand` on 2026-11-11.
The [roadmap](../PR_ROADMAP.md) describes review order, bases, scope and acceptance criteria.

| PR | Scope | Remote status |
| --- | --- | --- |
| [#2](https://github.com/Neksand/sound_rs/pull/2) | media | Open; latest CI results are on the linked PR |
| [#3](https://github.com/Neksand/sound_rs/pull/3) | player | Open; latest CI results are on the linked PR |
| [#4](https://github.com/Neksand/sound_rs/pull/4) | recorder | Open; latest CI results are on the linked PR |
| [#5](https://github.com/Neksand/sound_rs/pull/5) | resampler | Open; latest CI results are on the linked PR |
| [#6](https://github.com/Neksand/sound_rs/pull/6) | quality | Open; latest CI results are on the linked PR |
| [#7](https://github.com/Neksand/sound_rs/pull/7) | dependencies | Open; latest CI results are on the linked PR |

The first remote runs passed their completed Linux/Windows, Python, lint and documentation
checks, but macOS jobs were cancelled before starting: GitHub reported the `macos-14` runner
configuration unsupported. [GitHub retired these images on 2026-11-02](https://github.com/actions/runner-images#available-images).
The complete series now uses the supported `macos-15` image for CI and release wheel builds.
The PR checks are the authoritative status for each current head; local checks alone are not
reported as cross-platform CI success. Release publication and hardware tests were not run.

All pushes and PR creation used the owner-authorized temporary credential. No personal-account
fork was created. Nothing was merged, tagged, published or deployed.

## Recommendation

Review and merge the focused reliability series in order after CI. Prioritize the unresolved
native MP3 advisory and physical device validation next, then continuous streaming and resource
limits. Introduce editing and analysis APIs only with explicit numerical contracts and measured
use cases; a larger feature list is not a substitute for the remaining reliability evidence.
