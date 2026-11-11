# Dependency advisory review

Review date: 2026-11-11. Baseline: `c1898635c49534e51437ad853951477094ed0665`.
Official RustSec database commit: `28d73fc95e5972af3403225a217def57dcaba3c4`.

## Remediated version ranges

| Dependency | Before | After | Reason |
| --- | --- | --- | --- |
| PyO3 family | 0.25.1 | 0.29.3 | Fix versions affected by RUSTSEC-2026-0176 and RUSTSEC-2026-0177 |
| numpy Rust crate | 0.25.0 | 0.29.0 | Compatible PyO3 family; Python NumPy remains required only for array APIs |

Primary advisories: [iterator arithmetic](https://rustsec.org/advisories/RUSTSEC-2026-0176.html)
and [closure thread safety](https://rustsec.org/advisories/RUSTSEC-2026-0177.html).
No direct project calls to the named vulnerable PyO3 APIs were found; upgrading still removes
the affected dependency versions. Six lockfile packages change version, three are removed,
and no new direct dependency or system requirement is introduced.

Migration replaces `Python::with_gil` with `Python::attach` and `allow_threads` with `detach`.
Explicit `gil_used=true` preserves the old GIL requirement instead of silently claiming
free-threaded safety. `abi3-py310`, Python signatures and decoder versions remain unchanged.
No version bump or release publication is included.

## Unresolved transitive advisory

`sound_rs -> minimp3 0.6.1 -> slice-ring-buffer 0.3.4` remains in the affected version range of
[RUSTSEC-2025-0044](https://rustsec.org/advisories/RUSTSEC-2025-0044.html), describing memory-safety
flaws in safe APIs. minimp3 0.6.1 is the latest published release at review time; no patched
slice-ring-buffer release is listed. Replacing the decoder without independent audio verification
would violate the project's regression contract.

No direct project use of the four named vulnerable APIs was found; minimp3 uses a
`SliceRingBuffer<u8>` and generic extension. This is not a complete call-graph or exploitability
proof. Do not describe the package as cleared for arbitrary hostile media. Next step: review an
upstream fix or separately evaluate a replacement with nonperiodic mono/stereo MP3 fixtures,
FFmpeg sample alignment, licensing and performance evidence.

## Method and validation

`cargo-audit` was unavailable. Manual version-range triage parsed all 1,426 crate advisories
against 202 baseline lock entries (199 after migration). Python packages, system libraries,
unpublished vulnerabilities and OS audio backends were outside this check.

Isolated migration branch: formatting, Clippy with warnings denied, all-target compilation,
39 Rust tests, build without default features, macOS arm64 cp310-abi3 wheel/isolated install,
Python smoke and all six decoder regressions passed. Baseline hashes were not edited.
Combined and remote outcomes are tracked in the final engineering report.
