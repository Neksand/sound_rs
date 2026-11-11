# Coming from another library

Most Python audio code is three lines. Here is what those three lines become.

## From sounddevice

```
# before
import sounddevice as sd
import soundfile as sf

data, rate = sf.read("track.flac")
sd.play(data, rate)
sd.wait()

# after
import sound_rs

sound_rs.play_audio("track.flac")
```

You lose the array, which you did not need for playback, and gain no ffmpeg and no per-call device setup. If you do need the samples, `sound_rs.load` returns them in the same shape as `sf.read`.

For a persistent stream with pause and seek, `sounddevice` needs an `OutputStream` plus a callback queue; `sound_rs.Player` is that in one object.

## From soundfile

```
# before
import soundfile as sf
data, rate = sf.read("track.m4a", dtype="float32")

# after
import sound_rs
data, rate = sound_rs.load("track.m4a")
```

## From pydub

```
# before — needs an ffmpeg binary on PATH
from pydub import AudioSegment
from pydub.playback import play

segment = AudioSegment.from_file("track.mp3")
play(segment)

# after
import sound_rs

sound_rs.play_audio("track.mp3")
```

No subprocess, no temporary WAV, no ffmpeg install step for your users. The trade: pydub also does cutting, concatenation and format conversion, which sound_rs does not.

## From playsound / simpleaudio

```
# before — one shot, no controls, format support varies by platform
from playsound import playsound

playsound("track.mp3")

# after
import sound_rs

sound_rs.play_audio("track.mp3")
```

The wheel includes the same decoder families on each supported platform. The playback wait releases the GIL; device setup and teardown still depend on the backend.

## What you gain, what you give up

|  | sound_rs | soundfile + sounddevice | pydub |
|---|---|---|---|
| Formats | WAV, FLAC, MP3, OGG, M4A | WAV, FLAC, MP3; no AAC | anything ffmpeg reads |
| External tools | none | none | ffmpeg binary |
| Playback without loading | yes | no | no |
| Editing (cut, mix, convert) | no | no | yes |
| Platforms | 3, wheels | many | any with ffmpeg |
