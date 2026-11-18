# Dependency advisory review

Review date: 2026-11-11. Updated: 2026-11-12 (minimp3 removed). Baseline: `c1898635c49534e51437ad853951477094ed0665`.
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

## Resolved transitive advisory

`sound_rs -> minimp3 0.6.1 -> slice-ring-buffer 0.3.4` used to be in the affected range of
[RUSTSEC-2025-0044](https://rustsec.org/advisories/RUSTSEC-2025-0044.html), describing
memory-safety flaws in safe APIs. There is still no patched `slice-ring-buffer` release.

This was resolved by removing minimp3 rather than accepting the advisory: full-file MP3 now
uses Symphonia, like bytes, windows and streaming. The lock no longer contains minimp3,
minimp3-sys or slice-ring-buffer, and `cargo audit` runs without ignores.

Independent verification before removal: a 60-second LAME/Xing stereo file decoded exactly
2,646,000 frames through both paths. Symphonia matched ffmpeg at lag 0 (max 1.6·10⁻⁷, 127.6 dB
SNR); minimp3 had a systematic +47-frame offset from decoding the Xing header as audio.
Symphonia was 31.5 ms versus 25.1 ms median (1.26× slower, roughly 2,000× realtime). The
committed `tone.mp3` fixture now aligns with ffmpeg at lag 0 and has a sample hash instead of
an RMS tolerance.

This is not a complete call-graph or exploitability proof for arbitrary hostile media; it
removes the known advisory from the shipped dependency tree.

## Method and validation

`cargo-audit` was unavailable. Manual version-range triage parsed all 1,426 crate advisories
against 202 baseline lock entries (199 after migration). Python packages, system libraries,
unpublished vulnerabilities and OS audio backends were outside this check.

Isolated migration branch: formatting, Clippy with warnings denied, all-target compilation,
39 Rust tests, build without default features, macOS arm64 cp310-abi3 wheel/isolated install,
Python smoke and all six decoder regressions passed. Baseline hashes were not edited.
Combined and remote outcomes are tracked in the final engineering report.

## CI enforcement

`.github/workflows/ci.yml` now enforces the reviewed state:

- Python 3.13 is added to the `python` matrix on `ubuntu-latest`, alongside 3.10, 3.12 and 3.14.
- The `lint` job runs `cargo test --no-default-features` after installing ALSA headers, reusing
  the existing toolchain to compile and test the crate without the default `numpy` feature.
- A separate `audit` job installs `cargo-audit` 0.23.0 with
  `cargo install cargo-audit --version 0.23.0 --locked` and runs `cargo audit` with no ignores.
  The current lock has 192 packages and reports no advisories. This was verified locally; the
  remote job is configured but not claimed as passed here.

These are configured checks, not observed results. They had not been run on GitHub-hosted
runners at the time of writing, so this report makes no claim that remote CI has passed.
