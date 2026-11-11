# Implementation plan

Baseline and evidence: [engineering audit](reports/ENGINEERING_AUDIT.md),
[history](reports/ENGINEERING_HISTORY.md). Status is tracked in the [PR roadmap](PR_ROADMAP.md).

## Implemented workstreams

1. **Media and Python boundaries.** Checked MP4/WAV arithmetic; absolute reservation cap;
   checked NumPy borrows/channel counts; valid times/rates/sine inputs; explicit WAV errors;
   GIL release during writing. Acceptance: targeted old behavior fails, new unit/Python tests pass,
   valid file samples and existing decoder expectations remain unchanged.
2. **Playback lifecycle.** Correct final drain and empty feeds; timeout only on absent progress;
   checked source transitions/seek and backend failures. Acceptance: simulated callback/ring and
   decoder-worker tests prove completion, ordering, state preservation and stable error codes.
3. **Recorder.** Existing rtrb ring and collector thread; no callback allocation/mutex;
   whole-take failure after overflow/backend fault; bounded storage and deterministic shutdown.
   Acceptance: simulated input/overflow/wrap/fault/drop tests plus Python binding smoke.
4. **Streaming resampling.** Account exact frame ratio, trim startup/padded output and drain tail.
   Acceptance: multiple rates, packet partitions, stereo, short/full blocks, reset and tail impulses
   match a complete reference filter pass; decoder fixture hashes stay unchanged.
5. **Validation and documentation.** Fail on absent fixtures, compare actual registered APIs,
   validate generated links/indexes, headless benchmark mode, portable wheel numerical tests;
   correct capability claims and publish audit/history/final reports.
6. **Dependencies.** Trial the smallest compatible PyO3/NumPy upgrade fixing known PyO3 advisories.
   Acceptance: no direct dependency addition, ABI remains Python 3.10+, full validation unchanged;
   unremediated MP3 transitive risk remains explicit.

## Deferred milestones and acceptance criteria

| Priority | Milestone | Required evidence before implementation/claim |
| --- | --- | --- |
| P1 | Resolve minimp3 transitive advisory | Reviewed upstream fix or decoder replacement; nonperiodic mono/stereo golden audio, independent FFmpeg comparison, license/performance review |
| P1 | Decode resource limits and changing specs | Explicit allocation/frame budget and stable API errors; hostile metadata and multitrack fixtures |
| P1 | Device tests and recovery | Windows/Linux/macOS device runs; disconnect/reconnect/cancel tests; bounded shutdown; no automatic recovery claim until verified |
| P2 | Continuous fed resampling | Retained filter state across feed blocks, exact long-run frame accounting, impulse/sine boundary tests |
| P2 | Incremental WAV recording/writing | Bounded memory across long takes, correct finalized headers/cancellation, disk-failure tests |
| P2 | Frame-first editing and level analysis | Explicit units/ownership/empty/NaN semantics; trim/join/gain/fade/normalize/RMS/peak mathematical tests and meaningful benchmarks |
| P3 | FFT/STFT/filtering/synthesis | Document normalization/window/latency, frequencies/axes and channel handling; independent numerical references |
| P3 | Optional AI/native API integrations | Proven use case, no required models/GPU framework, compatibility design and maintenance ownership |
| P3 | Dynamic effects/plugin framework | Defer until concrete use; safe lifecycle/version/trust-boundary design before loading native code |

No release version is incremented. No empty APIs, artificial crate splits or unsupported feature
claims are part of this work. Prepared PRs remain reviewable and must not be auto-merged.
