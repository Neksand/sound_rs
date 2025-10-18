# 🎵 sound_rs

**sound_rs** is a high-performance Python audio library powered by **Rust** —  
offering **instant playback**, **low latency**, and **seamless NumPy integration**.  

Built for developers who want *speed*, *stability*, and *simplicity* in one package.

---

## 🚀 Features

✅ Play `.wav` files instantly  
✅ Generate and play sounds directly from **NumPy arrays**  
✅ Near-zero latency and minimal CPU usage thanks to **Rust**  
✅ Safe, stable, and cross-platform  
✅ Async-ready design for future streaming  

---

## 🧠 Installation

```bash
pip install sound_rs
```

---

## 💻 Usage Examples

### 1️⃣ Basic Playback
```python
import sound_rs

# Play a WAV file
sound_rs.play_audio("sound.wav")
```

### 2️⃣ From NumPy Array 
```python
import sound_rs
import numpy as np

# Generate 440 Hz sine wave (A4 note)
sample_rate = 44100
duration = 2.0
t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
sine_wave = np.sin(2 * np.pi * 440 * t)

# Play directly from array (mono)
sound_rs.play_array(sine_wave, sample_rate)
```

### 3️⃣ Stream Large Files 
```python
# Play large audio file without full memory load
sound_rs.play_audio_streamed("large_file.wav")
```

### 4️⃣ Async Playback
```python
# Play in background while doing other work
sound_rs.play_audio_async("background.wav")
```

### 5️⃣ Tone Generator
```python
# Instantly generate and play a 440 Hz sine wave
sound_rs.play_sine_wave(440.0, 1.0, 44100)
```

---

## 🛠️ Roadmap

| Step | Feature | Description |
|------|----------|-------------|
| ✅ 1 | **WAV playback** | Core playback for `.wav` files |
| 🔜 2 | **Support for MP3, OGG, FLAC** | Add support for popular formats |
| 🔜 3 | **Real-time audio recording** | Record microphone input and save/play |
| 🔜 4 | **Volume & playback control** | Pause, resume, seek, adjust volume |
| 🔜 5 | **Asynchronous streaming API** | Non-blocking playback for large or streamed audio |
| 🔜 6 | **Audio synthesis engine** | Generate tones, envelopes, and waveforms |
| 🔜 7 | **Audio visualization tools** | Waveform and frequency spectrum visualization |
| 🔜 8 | **AI-powered voice filtering** | Integration with deep learning filters (noise removal, EQ) |
| 🔜 9 | **FFI plugin system** | Allow C/C++/Rust extensions for custom audio effects |

---

## ⚙️ Built With

- 🦀 **Rust** – performance and safety  
- 🔗 **PyO3** – bridge between Python and Rust  
- 🧰 **Maturin** – build and publish Rust-based Python packages  
- 🔊 **CPAL / Rodio** – for cross-platform audio backend  

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
