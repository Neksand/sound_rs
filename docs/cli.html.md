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
  soundrs info <file> [--json]
  soundrs convert <input> <output.wav> [--subtype pcm16|pcm24|float32]
  soundrs devices
  soundrs docs
  soundrs mcp
  soundrs --version
  soundrs --help

Commands
  play     Play a file and wait until it ends.
           --device <name>   output device, see `soundrs devices`
           --volume <number> 0 to 4, default 1
  info     Sample rate, channels and duration.
           Reads the header when complete; decodes only when duration/format info is missing.
           --json            one line of JSON instead of the human output
  convert  Decode any supported file and write it as WAV.
           --subtype pcm16|pcm24|float32   sample format, default pcm16
  devices  Output devices, one per line.
  docs     Print the documentation link.
  mcp      Run the Model Context Protocol stdio server (audio_info, convert).

Links
  Documentation  https://neksand.github.io/sound_rs/
  Source         https://github.com/Neksand/sound_rs
```

## Commands

| Command | What it does |
|---|---|
| soundrs play FILE | Plays a file and waits.  --device  picks an output device,  --volume  takes 0…4. |
| soundrs info FILE [--json] | Sample rate, channel count and duration; reads the header when complete and decodes only when duration or format information is missing.  --json  prints one machine-readable line. |
| soundrs convert IN OUT.wav | Decodes any supported input and writes PCM16, PCM24 or float32 WAV via  --subtype . |
| soundrs devices | Output devices, the default marked with  * . |
| soundrs mcp | Starts the MCP stdio server with  audio_info  and  convert  tools. |
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

# machine-readable metadata for scripts and agents
soundrs info podcast.m4a --json

# convert any supported input to WAV
soundrs convert voice.mp3 take.wav --subtype pcm24
```

```text
$ soundrs info podcast.m4a
file         podcast.m4a
sample rate  44100 Hz
channels     2
duration     42:13.6 (2533.612 s)

$ soundrs info podcast.m4a --json
{"file":"podcast.m4a","sample_rate":44100,"channels":2,"duration_seconds":2533.612}
```

## MCP server for AI agents

`soundrs mcp` speaks the Model Context Protocol over stdio. It does not open an audio device. Tools:

- `audio_info(path)` — sample rate, channels and duration as JSON text.

- `convert(input, output, subtype=)` — decode any supported file to WAV.

```json
{
  "mcpServers": {
    "sound_rs": {
      "command": "soundrs",
      "args": ["mcp"]
    }
  }
}
```

This works without an audio device, so it is suitable for containers and coding agents. The MCP client launches the server; paths are local to that process.

## Checking an install

After `pip install sound_rs`:

```bash
soundrs --version      # soundrs 0.2.2
soundrs devices        # a device list means the audio backend loaded
soundrs play test.wav  # actual output
```

If any of these fail with an import or library error, the audio backend did not load — see [Troubleshooting](faq.html). Decoding and `soundrs info` do not need an output device at all.
