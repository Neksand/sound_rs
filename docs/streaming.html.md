# Streaming and memory

`play_file` decodes incrementally. A decoder thread fills a one-second ring buffer, and the audio callback drains it. The ring size is independent of file length; container, decoder, resampler and process memory are additional.

## How it works

When the sample rate changes, the resampler removes its filter delay and drains the filter tail at EOF. The output contains exactly `ceil(input_frames * output_rate / input_rate)` frames, including tracks shorter than a processing block. Seeking resets the filter and frame counters. This does not make independently resampled `feed()` blocks continuous; use the device rate for those blocks.

- The decoder thread reads a packet, decodes it to float32, converts the channel count and resamples to the device rate.

- The result goes into a ring buffer sized for one second of device audio.

- The audio callback pops samples from that buffer. When it is empty, the callback writes silence and keeps the stream alive.

- Seeking replaces the ring buffer and repositions the decoder; the old audio that was still queued is discarded and does not count towards the new position.

Playback can start before the full file is decoded. Startup still depends on device setup, container probing and any resampler buffering.

## Numbers

Historical 0.3.0 measurements: ten-minute stereo FLAC, 44.1 kHz, measured with `bench/run.py` on an Apple M5 Pro. These are not current hardware guarantees:

|  | sound_rs | soundfile + sounddevice |
|---|---|---|
| Time to first sound | 11.3 ms | 302 ms |
| Peak RSS | 43 MB | 256 MB |

The competitor number is not a flaw in those libraries: their API takes an array, so the file has to be decoded before playback can start. Streaming is a different design.

## play_file or load?

| You want | Use | Memory |
|---|---|---|
| Play a file | Player.play_file | ~1 s ring plus decoder/filter overhead |
| Analyse the samples | load | whole file as float32 |
| Play an array you already have | play_array | copied audio plus conversion temporaries |
| Play a file, need the samples too | load, then play_array | whole file |

Ten minutes of 44.1 kHz stereo float32 is about 212 MB of sample payload. Decoding and resampling can need additional temporary buffers.

```
import sound_rs

# streaming: bounded playback ring, incremental decode
with sound_rs.Player() as player:
    player.play_file("lecture-2h.m4a")
    player.wait()

# analysis: everything in RAM, decoded in parallel with other files
clips = sound_rs.load_many(["a.flac", "b.flac"])
```

## Caveats

- `duration` for a streamed file comes from the header; a VBR MP3 without a frame count reports `None` until the decoder reaches the end.

- Seeking uses container indexes and trims decoded packets to the requested position. Accuracy and cost depend on the codec/container; an unindexed stream may require walking forward.

- `load` retains its decoded result (or requested window), and `load_many` retains every result in the batch.
