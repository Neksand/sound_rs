# NumPy and arrays

Two directions: arrays in for playback, arrays out for analysis. Both use float32 samples in the range −1.0 to 1.0.

## Shapes

| Call | Shape | Meaning |
|---|---|---|
| play_array(mono, rate) | (frames,) | 1D, one channel |
| play_array(stereo, rate) | (2, frames) | 2D, channel first |
| load(path) | (frames, channels) | 2D, frames first |

## Playing arrays

```
import numpy as np
import sound_rs

rate = 48000
seconds = 3
t = np.arange(rate * seconds) / rate

left = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
right = (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)

sound_rs.play_array(np.stack([left, right]), rate)
```

The array is converted to the device layout once, before playback starts: channel folding, then one FFT resample if the rates differ. Playing a 48 kHz array on a 44.1 kHz device costs a few milliseconds and sounds correct — a naive nearest-neighbour resampler would alias.

## Decoding to arrays

```
import sound_rs

audio, rate = sound_rs.load("track.mp3")
print(audio.shape, audio.dtype, rate)        # (9437184, 2) float32 44100
print(audio[:, 0].max())                     # peak of the left channel
```

`load` returns the whole file. Ten minutes of stereo float32 is 211 MB, so for long files prefer streaming playback or decode one channel at a time from your own pipeline.

## Batch decoding

```
import glob
import sound_rs

paths = sorted(glob.glob("dataset/*.flac"))
clips = sound_rs.load_many(paths)             # uses every core
print(len(clips), clips[0][0].shape)
```

## Pitfalls

- **float64 arrays are rejected.** NumPy's `np.sin` returns float64; cast with `.astype(np.float32)` or use `dtype=np.float32`.

- **Integer arrays are rejected.** Convert to float32 and scale to −1…1 yourself; the library will not guess whether your int16 is scaled.

- **Non-contiguous arrays are copied.** A slice like `audio[::2]` still works, but you pay for the copy.

- **`play_array` copies the data.** The stream callback needs an owned buffer; that is one copy of the track, not two.
