# Audio for Python, without ffmpeg.

sound_rs plays and decodes the formats people actually have — MP3, FLAC, WAV, OGG, M4A — from a single wheel. The decoders are compiled in; no ffmpeg binary is required. Linux needs the ALSA runtime, and array APIs need NumPy.

Timing and memory figures above are historical 0.3.0 measurements on one Mac, not device guarantees. See the [benchmark methodology and limits](benchmarks.html).

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

- **Streaming decode**: file playback fills a bounded ring incrementally instead of retaining the whole decoded track.

- **Decoding to arrays** for analysis, one file at a time or in parallel.

- **Recording** from an input device to float32 arrays.

- **Writing** WAV in PCM16, PCM24 or float32; save a recorded take in one call.

- **Metadata** from the container header when complete, with decoding as a fallback.

- **Typed**: ships `sound_rs.pyi` and `py.typed`, so editors autocomplete every call.

## What it does not do

No effects, mixing graph or plugin system. If you need a DAW-style graph, this is the wrong tool; if you need to play or load audio reliably, it is the right one. The [benchmark page](benchmarks.html) also says where it is slower than the alternatives.

## Where to go next

- [Installation](install.html) — platforms, wheels, Linux audio libraries.

- [Quickstart](quickstart.html) — the five things most code does.

- [Command line](cli.html) — `soundrs --help` without writing Python.

- [The player](player.html) — controls, seeking, lifecycle.

- [Recording](recording.html) and [writing WAV files](writing.html).

- [API reference](api.html) — every function and method.
