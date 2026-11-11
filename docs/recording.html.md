# Recording

The mirror of playback: one input stream, float32 samples, and the same error contract.

```
import sound_rs

print(sound_rs.list_input_devices())     # [(0, 'MacBook Pro Microphone', True)]

with sound_rs.Recorder() as recorder:
    audio, rate = recorder.record(5.0)   # five seconds
print(audio.shape, rate)                 # (240000, 1) 48000
```

## Devices

`list_input_devices()` returns `(index, name, is_default)`. Pass the name to the constructor:

```
recorder = sound_rs.Recorder(device="MacBook Pro Microphone")
print(recorder.device, recorder.sample_rate, recorder.channels)
```

The rate and channel count are whatever the device reports; the library does not resample on input. If you need 16 kHz mono, ask for it after the fact:

```
import sound_rs

mono = audio.mean(axis=1, keepdims=True)     # or use numpy's resampler of choice
```

## start and stop

`record(seconds)` is the convenient form. For an open-ended take, use `start()` and `stop()`:

```
recorder = sound_rs.Recorder()
recorder.start()
input("press enter to stop")
audio, rate = recorder.stop()
```

## Writing a WAV

The arrays are plain float32, so the standard library writes them:

```
import wave
import numpy as np
import sound_rs

audio, rate = sound_rs.Recorder().record(3.0)
pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")

with wave.open("take.wav", "w") as handle:
    handle.setnchannels(audio.shape[1])
    handle.setsampwidth(2)
    handle.setframerate(rate)
    handle.writeframes(pcm.tobytes())
```

## Saving a take

```
import sound_rs

with sound_rs.Recorder() as recorder:
    recorder.record_to("take.wav", 5.0, subtype="pcm24")
```

For anything else — resampling, trimming, writing a different name — record to an array and use `sound_rs.write`, which writes the same `(frames, channels)` layout that `load` returns.

```
import sound_rs

audio, rate = sound_rs.Recorder().record(5.0)
sound_rs.write("take.wav", audio, rate, subtype="float32")
```

## Limits

- The capture starts when the device delivers its first buffer. On Bluetooth headsets that is 100–150 ms, so a short take can be that much shorter than requested — the recorder never adds frames to hide it.

- The callback writes to a bounded input ring (normally one second, at most 4 MiB). A collector thread stores the take outside the audio callback. The recording limit remains 230 M float32 samples (920 MB of payload, roughly 40 minutes at 48 kHz stereo). This is not a cap on total process memory.

- Input overflow invalidates the take with `code="stalled"`, the storage limit or allocation failure uses `unsupported`, and a backend failure uses `device`. The library does not return a successful partial take after these faults. `stop()` drains the final callback and joins the collector with the GIL released.

- `record_to()` records to memory before writing the WAV; it is not an incremental disk recorder.

- No resampling, mixing or effects on input — capture is raw device audio.

- Duplex (playing while recording) works because they are separate streams, but they are independent objects with independent device handles.
