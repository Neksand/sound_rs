# 🎵 sound_rs

**sound_rs** is a fast, Rust-powered audio playback library for Python that I built
and maintain — decoding with [symphonia](https://github.com/pdeljanov/Symphonia),
output through [cpal](https://github.com/RustAudio/cpal), so the same code runs on
macOS, Windows and Linux.

---

## 🚀 Features

- **Formats:** WAV, MP3, FLAC, OGG/Vorbis, M4A, AAC — any bit depth, including 24-bit
- **NumPy arrays:** play 1D mono `(samples,)` or 2D `(channels, samples)` float32 arrays
- **Decode to array:** `load(path) -> (ndarray, sample_rate)`, shape `(frames, channels)`
- **Gapless:** encoder delay and padding are trimmed, so decoded length matches the original
- **Header-only metadata:** `get_audio_info` reads the container, it does not decode the file
- **Resampling:** linear interpolation to the device rate, channel folding to the device layout
- **Device-native output:** the stream is opened in whatever sample format the device wants (f32, i16, u16, ...)
- **Non-blocking for others:** playback blocks the calling thread but releases the GIL, so other Python threads keep running
- **Tone generator:** `play_sine_wave(440, 2.0, 44100)`

---

## 🧠 Installation

```bash
pip install sound_rs
```

---

## 💻 Usage

### Playback

```python
import sound_rs

sound_rs.play_audio("sound.wav")
sound_rs.play_audio("sound.mp3")
sound_rs.play_audio("sound.flac")
sound_rs.play_audio("sound.m4a")
```

### Inspect a file (no decoding)

```python
sample_rate, channels, duration = sound_rs.get_audio_info("sound.mp3")
sound_rs.is_supported_format("sound.mp3")  # probes the header, does not decode
```

### Decode to a NumPy array

```python
import sound_rs

audio, rate = sound_rs.load("music.flac")   # float32, shape (frames, channels)
```

### From a NumPy array

```python
import numpy as np
import sound_rs

rate = 44100
t = np.linspace(0, 2.0, rate * 2, endpoint=False, dtype=np.float32)
sound_rs.play_array(np.sin(2 * np.pi * 440 * t), rate)              # mono, 1D
sound_rs.play_array(np.stack([left, right]), rate)                  # 2D, channels first
```

### Background playback

```python
import threading

threading.Thread(target=sound_rs.play_audio, args=("long.mp3",)).start()
```

### Tone generator

```python
sound_rs.play_sine_wave(440.0, 1.0, 44100)  # frequency, duration, sample rate
```

---

## 🧩 Benchmark

`play_audio` is synchronous, so its wall-clock time is dominated by the track's
own duration. The interesting numbers are **metadata**, **decode** and
**overhead** (wall clock − audio duration) — and decode is only worth measuring
if the output is correct, so every run also compares PCM against libsndfile.

```bash
python examples/benchmark.py                        # generated 3s tone: decode + playback
python examples/benchmark.py --compare              # + sounddevice / soundfile
python examples/benchmark.py --files a.mp3 b.flac   # decode table for real files
```

**Hardware and setup I measured on:** Apple M5 Pro, macOS 26.6.2, Python 3.14.7,
release build (`lto = "fat"`, `opt-level = 3`), 60 s of 44.1 kHz stereo,
median and p95 of 7 runs, competitor is libsndfile (`soundfile` 0.14).

| format | metadata | decode (median / p95) | libsndfile | result | PCM vs libsndfile |
|---|---|---|---|---|---|
| WAV (PCM 16) | 0.01 ms | **8.8 / 11.7 ms** | 11.5 ms | 1.3× faster | identical |
| FLAC | 0.02 ms | **16.7 / 17.2 ms** | 41.4 ms | 2.5× faster | identical |
| MP3 | 0.02 ms | 38.0 / 41.6 ms | **24.9 ms** | 1.5× slower | 1.9e-07 |
| M4A (AAC) | 0.14 ms | 34.2 / 39.1 ms | unsupported | sound_rs only | — |

Playback of a 3 s tone, overhead over the track duration (median of 5):

| | median | p95 | overhead |
|---|---|---|---|
| sound_rs | 3.103 s | 3.128 s | +0.103 s |
| sounddevice | 3.128 s | 3.137 s | +0.128 s |

What the numbers say, including the unflattering parts:

- **Metadata no longer decodes the file.** Reading a 2-hour file costs the same
  as reading a 3-second one, because it comes from the container header.
- **PCM output matches libsndfile exactly** for WAV and FLAC, and to 1.9e-07
  for MP3 — both trim the encoder delay and padding, so sample N means the
  same thing in both libraries.
- **MP3 decoding is ~1.5× behind libmpg123.** That is the decoder itself, not
  the plumbing; fixing it means a different MP3 decoder.
- **Playback overhead is stream setup and teardown**, ~0.1 s per call. Reducing
  it to single-digit milliseconds needs a persistent stream, which is the next
  item on the roadmap.


---

## 🛠️ Roadmap

| Step | Feature | Description |
|------|---------|-------------|
| ✅ 1 | WAV playback | Core playback |
| ✅ 2 | MP3, FLAC, M4A, AAC, 24-bit | Full symphonia decode path |
| ✅ 3 | NumPy integration | 1D/2D float32 arrays |
| ✅ 4 | Decode to array | `load(path) -> (ndarray, sample_rate)` without playing |
| 🔜 4 | `Player` object | Persistent stream: pause, resume, seek, volume, position |
| 🔜 5 | Device selection | `list_devices()`, choose input/output |
| 🔜 6 | Recording | Microphone → array or WAV |

---

## 🧪 Development

```bash
cargo test                      # resampler and channel conversion
maturin develop --release       # build the extension into your venv
python tests/smoke.py           # end-to-end decode checks
SOUND_RS_PLAY=1 python tests/smoke.py   # also plays (needs an audio device)
python examples/benchmark.py
```

---

## ⚙️ Built With

- 🦀 **Rust** — performance and safety
- 🔗 **PyO3** — Python bridge
- 🧰 **Maturin** — building and publishing
- 🔊 **cpal** + **symphonia** — audio backend and decoders

---

## 👨‍💻 Author

**Nurassyl Mussainov**
Kazakhstan 🇰🇿
Python / Rust Developer

🔗 GitHub: [Neksand](https://github.com/Neksand)
📦 PyPI: [sound_rs](https://pypi.org/project/sound_rs)

---

## 📜 License

Licensed under the **MIT License** — free for personal and commercial use.
