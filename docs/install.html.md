# Installation

One command. Wheels carry the decoders, so no ffmpeg or development headers are needed.

```bash
pip install sound_rs
```

## Requirements

- Python 3.10 or newer, CPython.

- Linux x86_64 or aarch64, Windows amd64, macOS arm64 or Intel — all ship wheels.

- NumPy for array playback, loading, writing and recording arrays: install with `pip install 'sound_rs[numpy]'`.

| Platform | Wheel tag | Python |
|---|---|---|
| Linux x86_64 | cp310-abi3-manylinux_2_17_x86_64 | 3.10+ |
| Linux aarch64 | cp310-abi3-manylinux_2_17_aarch64 | 3.10+ |
| Windows amd64 | cp310-abi3-win_amd64 | 3.10+ |
| macOS arm64 | cp310-abi3-macosx_11_0_arm64 | 3.10+ |
| macOS Intel | cp310-abi3-macosx_10_12_x86_64 | 3.10+ |

The wheels use the stable ABI, so one file covers every Python from 3.10 up — the same wheel installs on 3.10 through 3.14. Anything else (musl, PyPy, other architectures) falls back to the source distribution, which needs a Rust toolchain.

## From source

```bash
git clone https://github.com/Neksand/sound_rs
cd sound_rs
pip install maturin
maturin develop --release
```

## Linux audio

Playback uses ALSA through `cpal`. Most desktops already have the runtime library; a minimal container or server needs it explicitly:

```bash
sudo apt install libasound2          # Debian / Ubuntu (libasound2t64 on 24.04+)
sudo dnf install alsa-lib            # Fedora
```

## Check it works

```
import sound_rs

print(sound_rs.__version__)
print(sound_rs.list_devices())        # [(0, 'MacBook Pro Speakers', True), ...]
print(sound_rs.get_audio_info("song.mp3"))
```

From a terminal, `soundrs --help` prints the same check as a command, with the documentation link. See [Command line](cli.html).

No sound, or an empty device list? See [Troubleshooting](faq.html).
