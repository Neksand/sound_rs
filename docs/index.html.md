# Audio for Python, without ffmpeg.

sound_rs plays and decodes the formats people actually have — MP3, FLAC, WAV, OGG, M4A — from a single wheel. The decoders are compiled in, so there is nothing to install besides the package itself.

```bash
pip install sound_rs
```

```
import sound_rs

with sound_rs.Player() as player:
    player.play_file("music.flac")   # streams from disk, returns immediately
    player.seek(42.0)
    print(player.position, player.duration)
    player.wait()
```

## What it does

- **Playback** of files and NumPy arrays, blocking or through a persistent player with pause, seek, volume and position.

- **Streaming decode**: a two-hour file never sits in RAM, and the first sound arrives in milliseconds instead of after a full decode.

- **Decoding to arrays** for analysis, one file at a time or in parallel.

- **Writing** WAV in PCM16, PCM24 or float32; record straight to a file.

- **Metadata** straight from the container header, without decoding the audio.

- **Typed**: ships `sound_rs.pyi` and `py.typed`, so editors autocomplete every call.

## What it does not do

No effects, no mixing, and no plugin system. If you need a DAW-style graph, this is the wrong tool; if you need to play or load audio reliably, it is the right one. The [benchmark page](benchmarks.html) also says where it is slower than the alternatives.

## Where to go next

- [Installation](install.html) — platforms, wheels, Linux audio libraries.

- [Quickstart](quickstart.html) — the five things most code does.

- [Command line](cli.html) — `soundrs --help` without writing Python.

- [The player](player.html) — controls, seeking, lifecycle.

- [API reference](api.html) — every function and method.
