# Troubleshooting

What to check when there is no sound, no device, or a call that blocks longer than expected.

## No sound

- `sound_rs.list_devices()` — if the list is empty, the OS sees no output device.

- `sound_rs.play_sine_wave(440.0, 1.0, 44100)` — if this is silent, the problem is the device, not the file.

- Check the system volume and the default output device; the player follows the OS default.

- On macOS, the terminal or IDE may need microphone/audio permission for output routing in some sandboxed setups.

## The device is busy

`SoundRsError: Cannot open output stream` usually means another application holds the device in exclusive mode — a DAW, a call, or another Python process with an open player. Close one player before opening another, or use a different device:

```
import sound_rs

print(sound_rs.list_devices())
sound_rs.play_audio("track.mp3", device="Built-in Audio Analog Stereo")
```

## Linux: ALSA and PipeWire

Playback goes through ALSA. On PipeWire systems install `pipewire-alsa`; on PulseAudio systems install `pulseaudio-alsa`, so ALSA has a bridge to the sound server.

In a container there is no sound server and no device: that is expected. Use `load` for decoding — it needs no audio device.

## Why the first play is slow

`play_audio` and `play_array` open the output device on every call: roughly 0.1–0.2 s before the first sample. `Player` opens it once, which brings the overhead down to about 30 ms.

```
with sound_rs.Player() as player:
    player.play_file("a.flac"); player.wait()
    player.play_file("b.flac"); player.wait()   # no device setup
```

## The call blocks forever

It should not. Playback without progress for two seconds raises `SoundRsError: Playback stalled` instead of hanging. If you see it, the device probably disappeared — reopen the player and retry.

If you need playback in the background, use a thread or `Player.play_file`, which already returns immediately.

## Jupyter and notebooks

Blocking calls work, but a long track freezes the cell. Use the player:

```
player = sound_rs.Player()
player.play_file("lecture.mp3")
# the cell returns; more code runs while it plays
player.position
```

## asyncio

`play_audio` and `wait` release the GIL but they are still blocking calls. Off-load them:

```
import asyncio
import sound_rs

await asyncio.to_thread(sound_rs.play_audio, "track.flac")
```

## MP3 samples do not line up with another library

Different decoders trim the encoder delay slightly differently. sound_rs starts 47 samples (~1 ms) earlier than libsndfile and ffmpeg on LAME files; lengths match exactly. If you compare waveforms, align first. See [Formats and gapless](formats.html#gapless).

## "Input must be a 1D or 2D numpy array of float32"

NumPy's maths returns float64 by default:

```
samples = (0.2 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000))
samples = samples.astype(np.float32)          # required
sound_rs.play_array(samples, 48000)
```

## Intel Macs and ARM Linux

Only macOS arm64, Windows amd64 and Linux x86_64 have wheels. Elsewhere pip builds from source, which needs Rust. `pip install sound_rs --no-binary sound_rs` forces that path if a stale wheel is being picked up.
