# Streaming and memory

`play_file` never loads the track. A decoder thread fills a one-second ring buffer, and the audio callback drains it. Memory stays flat no matter how long the file is.

## How it works

When the sample rate changes, the resampler removes its filter delay and drains the filter tail at EOF. The output contains exactly `ceil(input_frames * output_rate / input_rate)` frames, including tracks shorter than a processing block. Seeking resets the filter and frame counters. This does not make independently resampled `feed()` blocks continuous; use the device rate for those blocks.

- The decoder thread reads a packet, decodes it to float32, converts the channel count and resamples to the device rate.

- The result goes into a ring buffer sized for one second of device audio.

- The audio callback pops samples from that buffer. When it is empty, the callback writes silence and keeps the stream alive.

- Seeking replaces the ring buffer and repositions the decoder; the old audio that was still queued is discarded and does not count towards the new position.

Because the decoder runs ahead by at most a second, the first sound arrives as soon as the first packet is decoded, not when the file ends.

## Numbers

Ten-minute stereo FLAC, 44.1 kHz, measured with `bench/run.py` on an Apple M5 Pro:

|  | sound_rs | soundfile + sounddevice |
|---|---|---|
| Time to first sound | 11.3 ms | 302 ms |
| Peak RSS | 43 MB | 256 MB |

The competitor number is not a flaw in those libraries: their API takes an array, so the file has to be decoded before playback can start. Streaming is a different design.

## play_file or load?

| You want | Use | Memory |
|---|---|---|
| Play a file | Player.play_file | constant, ~1 s of audio |
| Analyse the samples | load | whole file as float32 |
| Play an array you already have | play_array | one copy at device format |
| Play a file, need the samples too | load, then play_array | whole file |

A 10-minute stereo file is 211 MB in float32. That is fine for analysis and wasteful for playback.

```
import sound_rs

# streaming: memory stays flat, start is immediate
with sound_rs.Player() as player:
    player.play_file("lecture-2h.m4a")
    player.wait()

# analysis: everything in RAM, decoded in parallel with other files
clips = sound_rs.load_many(["a.flac", "b.flac"])
```

## Caveats

- `duration` for a streamed file comes from the header; a VBR MP3 without a frame count reports `None` until the decoder reaches the end.

- Seeking in OGG and MP4 is exact; seeking in a stream without an index (rare) may take longer because the decoder has to walk forward.

- Only playback streams. `load` and `load_many` always decode fully — that is their job.
