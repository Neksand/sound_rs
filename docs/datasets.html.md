# Datasets and ML

Most dataset code wants the same thing: 16 kHz, mono, one array per file, as fast as the machine allows. That is one call.

## One call

```
import glob
import sound_rs

paths = sorted(glob.glob("dataset/**/*.flac", recursive=True))
clips = sound_rs.load_many(paths, sample_rate=16000, mono=True)

for audio, rate in clips[:3]:
    print(audio.shape, rate)        # (frames, 1), 16000
```

`load_many` decodes in parallel with the GIL released. On two FLAC files the harness measures 168 ms against 301 ms for a `soundfile` loop; the gap grows with file count until the cores are saturated.

## One file, converted

```
import sound_rs

audio, rate = sound_rs.load(
    "clip.m4a",
    sample_rate=16000,   # resampled with an anti-aliasing filter
    mono=True,           # channels averaged
)
assert audio.dtype == "float32" and audio.ndim == 2
```

The order is fixed: fold channels first, then resample. Resampling before folding would do the same work twice on stereo.

## A window, not the whole file

```
# first 10 seconds at 16 kHz, without decoding the rest
audio, rate = sound_rs.load("lecture.m4a", duration=10.0, sample_rate=16000)

# 30 s starting at 5 minutes
audio, rate = sound_rs.load("lecture.m4a", offset=300.0, duration=30.0)
```

## Memory

| Task | Sample payload / observation |
|---|---|
| Decode 10 min 44.1 kHz stereo to float32 | About 212 MB of audio |
| Return 10 min as 16 kHz mono | 38.4 MB of returned samples |
| Stream the same file for playback | Bounded one-second ring plus overhead; historical 43 MB RSS |

Returned sample size is not peak RAM: decoding, channel conversion and resampling can temporarily retain larger source and destination buffers together.

Resampling to 16 kHz mono is not just faster downstream — it is 5× less memory than keeping 44.1 kHz stereo.

## Choosing the thread count

```
clips = sound_rs.load_many(paths, threads=4)   # explicit
clips = sound_rs.load_many(paths)              # one worker per core
```

Decoding is CPU-bound, so the ceiling is the core count. Feeding more threads than cores does not help; the benchmark's scaling scenario measures 1, 2, 4 and 8 workers.

## When not to use this

- **Training on the fly** — decode once and store the arrays; repeated decoding of the same file is wasted CPU.

- **Huge corpora** — `load_many` returns everything at once. For a corpus larger than RAM, call `load` per file inside your own batching.

- **Augmentation and feature extraction** — not implemented. This library gets you clean float32 audio; the DSP belongs to your framework.
