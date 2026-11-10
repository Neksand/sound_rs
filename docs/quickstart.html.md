# Quickstart

Five recipes. Each one runs as written.

## Play a file

```
import sound_rs

sound_rs.play_audio("music.mp3")
```

`play_audio` blocks until the track ends and releases the GIL while it does, so other Python threads keep running. It opens the output device for that one call.

## Play with controls

```
import sound_rs

with sound_rs.Player() as player:
    player.play_file("music.flac")   # returns immediately
    player.volume = 0.6
    player.seek(30.0)                # jump to 0:30
    player.pause()
    player.resume()
    player.wait()                    # block until it ends
```

The player keeps the stream open, so starting the next track costs milliseconds instead of reopening the device. Details in [The player](player.html).

## Play a NumPy array

```
import numpy as np
import sound_rs

rate = 44100
t = np.arange(rate * 2) / rate
mono = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

sound_rs.play_array(mono, rate)                       # 1D, mono
sound_rs.play_array(np.stack([mono, mono]), rate)     # 2D, (channels, frames)
```

## Inspect a file

```
sample_rate, channels, duration = sound_rs.get_audio_info("music.m4a")
print(sample_rate, channels, duration)   # 44100 2 213.4
```

This reads the container header — about 1 ms regardless of file length. It decodes one packet only for AAC, whose container does not record the channel count.

## Decode to NumPy

```
import sound_rs

audio, rate = sound_rs.load("music.flac")   # float32, (frames, channels)
print(audio.shape, rate)
```

## Decode many files

```
import sound_rs

clips = sound_rs.load_many(["a.flac", "b.flac", "c.flac"])   # all cores, GIL released
for audio, rate in clips:
    print(audio.shape, rate)
```

## Choose a device

```
import sound_rs

for index, name, is_default in sound_rs.list_devices():
    print(("*" if is_default else " "), index, name)

sound_rs.play_audio("music.mp3", device="MacBook Pro Speakers")
```

Continue with [The player](player.html).
