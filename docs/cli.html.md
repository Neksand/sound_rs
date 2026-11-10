# Command line

Installing the package also installs `soundrs`. It exists so that a minute after `pip install` you can check that audio works and find the docs, without writing Python.

```bash
soundrs --help
```

```text
sound_rs — audio for Python

Usage
  soundrs <command> [options]
  soundrs play <file> [--device <name>] [--volume <0-4>]
  soundrs info <file>
  soundrs devices
  soundrs docs
  soundrs --version
  soundrs --help

Commands
  play     Play a file and wait until it ends.
           --device <name>   output device, see `soundrs devices`
           --volume <number> 0 to 4, default 1
  info     Sample rate, channels and duration, read from the header only.
  devices  Output devices, one per line.
  docs     Print the documentation link.

Links
  Documentation  https://neksand.github.io/sound_rs/
  Source         https://github.com/Neksand/sound_rs
```

## Commands

| Command | What it does |
|---|---|
| soundrs play FILE | Plays a file and waits.  --device  picks an output device,  --volume  takes 0…4. |
| soundrs info FILE | Sample rate, channel count and duration from the container header — no decoding. |
| soundrs devices | Output devices, the default marked with  * . |
| soundrs docs | Prints the documentation URL, for scripts and  open $(soundrs docs) . |
| soundrs --version | Version on one line, safe to parse. |

`sound-rs` is installed as an alias of the same command.

## Examples

```bash
# does audio work at all?
soundrs play ~/Music/test.flac

# which device names can I pass to Player(device=...)?
soundrs devices

# what is this file before writing any code?
soundrs info podcast.m4a
```

```text
$ soundrs info podcast.m4a
file         podcast.m4a
sample rate  44100 Hz
channels     2
duration     42:13.6 (2533.612 s)
```

## Checking an install

After `pip install sound_rs`:

```bash
soundrs --version      # soundrs 0.2.2
soundrs devices        # a device list means the audio backend loaded
soundrs play test.wav  # actual output
```

If any of these fail with an import or library error, the audio backend did not load — see [Troubleshooting](faq.html). Decoding and `soundrs info` do not need an output device at all.
