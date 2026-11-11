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

The rate and channel count are whatever the device reports; the library does not resample on input. Folding to mono does not change the rate:

```
import sound_rs

mono = audio.mean(axis=1, keepdims=True)     # still at the captured sample rate
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

Use the same frames-first float32 layout as `load`:

```
import sound_rs

with sound_rs.Recorder() as recorder:
    audio, rate = recorder.record(3.0)
sound_rs.write("take.wav", audio, rate, subtype="pcm16")
```

## Saving a take

```
import sound_rs

with sound_rs.Recorder() as recorder:
    recorder.record_to("take.wav", 5.0, subtype="pcm24")
```

`record_to` captures the take in memory, then writes it; it is not incremental disk recording. For preprocessing or another destination, record to an array and use `sound_rs.write`, which writes the same `(frames, channels)` layout that `load` returns.

```
import sound_rs

audio, rate = sound_rs.Recorder().record(5.0)
sound_rs.write("take.wav", audio, rate, subtype="float32")
```

## Limits

- Create, use and destroy the recorder on its original thread.

- Capture starts when the device delivers its first buffer. Startup latency can make a timed take shorter than requested; the recorder does not add frames to hide it.

- The callback writes to a bounded input ring (normally one second, at most 4 MiB). A collector thread stores the take outside the audio callback. The recording limit remains 230 M float32 samples (920 MB of payload, roughly 40 minutes at 48 kHz stereo). This is not a cap on total process memory or a hard real-time guarantee.

- Input overflow invalidates the take with `code="stalled"`, the storage limit or allocation failure uses `unsupported`, and a backend failure uses `device`. The library does not return a successful partial take after these faults. `stop()` drains the final callback and joins the collector with the GIL released.

- `record_to()` records to memory before writing the WAV; it is not an incremental disk recorder.

- No resampling, mixing or effects on input — capture is raw device audio.

- Playback and recording use independent streams. Simultaneous access depends on backend and device support; there is no shared clock or synchronization guarantee.
