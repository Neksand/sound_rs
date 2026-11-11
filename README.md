# 🎵 sound_rs

[![PyPI](https://img.shields.io/pypi/v/sound_rs)](https://pypi.org/project/sound-rs/)
[![Downloads](https://img.shields.io/pypi/dm/sound_rs)](https://pypi.org/project/sound-rs/)
[![Python](https://img.shields.io/pypi/pyversions/sound_rs)](https://pypi.org/project/sound-rs/)
[![Platforms](https://img.shields.io/badge/platforms-Linux%20%7C%20Windows%20%7C%20macOS-blue)](#installation)
[![Typed](https://img.shields.io/badge/typing-typed-blue)](sound_rs.pyi)
[![Docs](https://img.shields.io/badge/docs-neksand.github.io%2Fsound__rs-ff5a36)](https://neksand.github.io/sound_rs/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**sound_rs** is a fast, Rust-powered audio library for Python that I built and
maintain — decoding with [symphonia](https://github.com/pdeljanov/Symphonia) (plus
minimp3 for MP3), resampling with [rubato](https://github.com/HEnquist/rubato),
output through [cpal](https://github.com/RustAudio/cpal).

**Documentation:** [neksand.github.io/sound_rs](https://neksand.github.io/sound_rs/) — guides,
API reference, the `soundrs` command and the benchmark methodology.

**Latest release:** [0.4.0](https://github.com/Neksand/sound_rs/releases/tag/v0.4.0) — the
recording era begins: write WAV, record straight to a file. Guides for coding agents:
[llms.txt](https://neksand.github.io/sound_rs/llms.txt) and [AGENTS.md](AGENTS.md).

---

## 🚀 Features

- **Formats:** WAV (including tested 24-bit PCM), MP3, FLAC, OGG/Vorbis and AAC in M4A/MP4;
  support depends on the codec and container, not just the file extension
- **Streaming playback:** incremental decoding into a bounded one-second playback ring;
  the full file is not retained in memory, while decoder/filter/process overhead is additional
- **`Player`:** one output stream stays open — `play_file`, `play_array`, `pause`, `resume`, `seek`,
  `stop`, `volume`, `position`, `duration`
- **Decode to NumPy:** `load(path, sample_rate=, mono=, offset=, duration=)` — resample to
  16 kHz mono in one call, or decode only a window
- **Parallel decode:** `load_many(paths, sample_rate=, mono=)` uses every core, GIL released
- **Recording:** `Recorder()` captures from the microphone; `record_to(path, seconds)`
  saves a take in one call. A bounded callback ring feeds a collector thread; overflow
  and device faults raise an error instead of returning an incomplete successful take
- **Writing:** `write(path, audio, rate, subtype=)` saves WAV as PCM16, PCM24 or float32
  with validated header sizes and finite samples; encoding and writing release the GIL
- **Queues and live input:** `Player.enqueue(path)` chains tracks; `begin_feed/feed` push
  samples from Python with backpressure. Queueing requires a file source; feed timeouts
  measure lack of progress, and `wait()` includes the final audio drain
- **Gapless metadata:** encoder delay/padding are trimmed where available. MP3 decoder paths
  retain a known alignment difference; AAC can retain less than one codec frame of tail padding
- **Metadata:** `get_audio_info` uses the container header when complete; missing information
  can require packet or full-file decoding
- **Resampling:** rubato's FFT resampler with a proper anti-aliasing filter
  and streaming EOF drain that preserves the converted frame count and filter tail
- **Device-native output:** the stream opens in whatever sample format the device
  wants (f32, f64, i16, i32, u8, u16)
- **Errors:** library failures raise `sound_rs.SoundRsError` with a stable `.code`
  (`io`, `format`, `device`, `stalled`, `unsupported`, `usage`)
- **Tested behavior:** malformed-input mutation tests, quantized fixture hashes and MP3
  length/level checks, ring-starvation reporting. The full list is on the docs site.
- **No external decoder:** MP3, FLAC, OGG, AAC and WAV decoders are inside the wheel.
  Linux wheels still need the ALSA runtime; NumPy is required for array APIs
- **Typed:** ships `sound_rs.pyi` and `py.typed`, so editors autocomplete everything

---

## 🧠 Installation

```bash
pip install sound_rs
```

Wheels are built with the stable ABI (`abi3-py310`): one per platform — Linux
x86_64 and aarch64, Windows amd64, macOS arm64 and Intel — each covering Python
3.10 and newer. Other platforms build from source.

For loading, writing, array playback and recording arrays, use `pip install 'sound_rs[numpy]'`.
Python argument conversion may raise `TypeError` or `OverflowError`; float64 arrays are rejected.

---

## ⌨️ Command line

Installing the package also installs `soundrs`, so you can check the setup and read the docs
without opening Python:

```bash
soundrs --help       # commands and the documentation link
soundrs devices      # output devices, default marked with *
soundrs info a.flac  # rate, channels, duration; decodes if the header is incomplete
soundrs play a.flac --volume 0.5
soundrs docs         # prints https://neksand.github.io/sound_rs/
```

Exit codes: `0` success, `1` runtime failure, `2` bad usage.

## 💻 Usage

### One-shot playback

```python
import sound_rs

sound_rs.play_audio("sound.wav")          # blocks, GIL released
sound_rs.play_audio("sound.mp3", device="MacBook Pro Speakers")
```

### The player

```python
import sound_rs

with sound_rs.Player() as player:
    player.play_file("music.flac")   # returns immediately, streams from disk
    player.volume = 0.7
    player.seek(42.0)
    player.pause()
    player.resume()
    print(player.position, player.duration)
    player.wait()                    # block until it ends

sound_rs.list_devices()              # [(0, "MacBook Pro Speakers", True), ...]
```

Create, use and destroy each `Player` or `Recorder` on the same thread. Separate threads
can own separate instances. Playback waits release the GIL; preparation and teardown can
still run while holding it.

### Record

```python
import sound_rs

with sound_rs.Recorder() as recorder:
    audio, rate = recorder.record(5.0)     # float32, (frames, channels)
```

### Write a file

```python
import sound_rs

audio, rate = sound_rs.load("take.flac")          # (frames, channels)
sound_rs.write("take.wav", audio, rate, subtype="pcm24")

with sound_rs.Recorder() as recorder:
    recorder.record_to("take.wav", 5.0)           # record and save
```

Arrays use `(frames, channels)` — the layout `load` returns — so a load/write round-trip
needs no transpose.

### Queue and live input

```python
import numpy as np
import sound_rs

with sound_rs.Player() as player:
    player.play_file("a.flac")
    player.enqueue("b.mp3")                # plays next, own rate and layout
    player.wait()

player = sound_rs.Player()
player.begin_feed(48000)
player.feed(np.zeros(4800, dtype=np.float32))   # blocks when the buffer is full
player.end_feed()
player.wait()
```

### Inspect and decode

```python
sample_rate, channels, duration = sound_rs.get_audio_info("music.mp3")  # header when sufficient
audio, rate = sound_rs.load("music.flac")                               # (frames, channels) float32
audio, rate = sound_rs.load("music.flac", sample_rate=16000, mono=True) # dataset shape

pairs = sound_rs.load_many(["a.flac", "b.flac", "c.flac"])              # parallel
```

### From a NumPy array

```python
import numpy as np

rate = 44100
t = np.linspace(0, 2.0, rate * 2, endpoint=False, dtype=np.float32)
sound_rs.play_array(np.sin(2 * np.pi * 440 * t), rate)   # mono, 1D
sound_rs.play_array(np.stack([left, right]), rate)       # 2D, channels first
sound_rs.play_sine_wave(440.0, 1.0, 44100)
```

---

## 🧩 Benchmark

Every measurement runs in its own process: peak RSS is process-wide, so timing
and memory cannot share one. Median and p95 over repeated runs, competitors run
side by side on the same files.

```bash
python bench/run.py                 # sound_rs only
python bench/run.py --compare       # + soundfile / sounddevice
python bench/run.py --reps 9
```

**Historical 0.3.0 measurements:** Apple M5 Pro, macOS 26.6.2, Python 3.14.7, release
build (`lto = "fat"`, `opt-level = 3`). Competitors: libsndfile 1.2
(`soundfile` 0.14), PortAudio (`sounddevice`). Corpus: 3 s and 10 min stereo
44.1 kHz, generated by the harness.

The tables below are historical measurements, not performance guarantees for a current device.
The [0.3.0 report](docs/reports/0.3.0-report.md) records the original run; the
[current engineering report](docs/reports/FINAL_ENGINEERING_REPORT.md) separates newly executed
headless measurements from device tests that were not rerun.

**Historical scoreboard:**

| axis | sound_rs | best competitor | ratio |
|---|---|---|---|
| import | 1.1 ms | 31.8 ms (soundfile) | 29× |
| metadata (60 s file) | 1.2 ms | 31.7 ms (soundfile) | 26× |
| decode WAV | 130 ms | 157 ms | 1.2× |
| decode FLAC | 265 ms | 455 ms | 1.7× |
| decode MP3 | 241 ms | 268 ms | 1.1× |
| decode AAC | 349 ms | not supported | — |
| time to first sound (10 min) | 11 ms | 302 ms | 27× |
| peak RAM (10 min file) | 43 MB | 255 MB | 6× |
| playback overhead | 30 ms | 241 ms | 8× |
| batch decode (2 files) | 177 ms | 299 ms | 1.7× |

Detailed original timings and validation are retained in the linked historical report.

What the numbers do not hide:

- **MP3 output is offset by 47 samples (~1 ms) versus libsndfile and ffmpeg.**
  Sample values agree to 1.6e-05 once aligned; the offset comes from how the
  decoder delay is trimmed, and the length matches exactly.
- **WAV decode is a memcpy race** with libsndfile; the gap there is small by
  nature, the real wins are FLAC, metadata, memory and latency.
- **AAC decoding is slower than everything else** — it is a newer, less tuned
  decoder — but libsndfile cannot read AAC at all.

---

## 🛠️ Roadmap

| Step | Feature | Description |
|------|---------|-------------|
| ✅ 1 | Playback + formats | WAV, MP3, FLAC, OGG, M4A, 24-bit |
| ✅ 2 | NumPy integration | 1D/2D arrays, `load`, `load_many` |
| ✅ 3 | `Player` | Persistent stream: pause, seek, volume, position |
| ✅ 4 | Streaming decode | Bounded playback ring, incremental file decoding |
| ✅ 5 | Device selection | `list_devices`, `list_input_devices`, `device=` |
| ✅ 6 | Recording | `Recorder`: microphone to float32 array |
| ✅ 7 | Queues and live input | `enqueue`, `begin_feed`/`feed` |
| ✅ 8 | Datasets | `load(..., sample_rate=16000, mono=True)`, parallel `load_many` |
| ✅ 9 | Writing files | `write`, `Recorder.record_to`, PCM16/PCM24/float32 WAV |
| Under investigation | Hot-swap on disconnect | Requires device-backed recovery and cancellation tests on each OS |
| Planned | Editing and level analysis | Frame-first slice/join/gain/fade/normalize/RMS/peak with numerical semantics |
| Planned | Continuous fed resampling | Retain filter state across blocks; validate long-run frame counts |
| Planned | Incremental disk recording | Bounded memory and correct WAV finalization on errors/cancel |
| Under investigation | FFT/STFT, filters and synthesis | Independent numerical references and documented axes/latency |

The [implementation plan](docs/IMPLEMENTATION_PLAN.md) gives acceptance criteria. The
[engineering audit](docs/reports/ENGINEERING_AUDIT.md) and [PR roadmap](docs/PR_ROADMAP.md)
distinguish current capabilities, proposed fixes and deferred work.

---

## 🧪 Development

```bash
cargo test                      # decode, gapless, seek, fuzzing, properties, player
cargo clippy --all-targets -- -D warnings
maturin develop --release       # build into your venv
python tests/smoke.py           # end-to-end decode + API checks
python tests/test_validation.py # invalid inputs, WAV boundaries, NumPy layouts, GIL
SOUND_RS_PLAY=1 python tests/smoke.py   # also plays (needs an audio device)
python tests/decode_regression.py  # PCM hashes; must not change
python docs/check_api.py        # every public name is documented
python docs/check_site.py       # links, anchors, search, sitemap, Markdown and llms
python -m unittest discover -s tests -p 'test_*.py'
python3 docs/_build.py          # docs site, llms.txt and .md pages
python bench/run.py --compare --report bench/REPORT.md
python bench/run.py --compare --no-audio  # reproducible device-free subset
```

---

## ⚙️ Built With

- 🦀 **Rust** — performance and safety
- 🔗 **PyO3** — Python bridge
- 🧰 **Maturin** — building and publishing
- 🔊 **cpal**, **symphonia**, **minimp3**, **rubato** — output, decoding, resampling

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
