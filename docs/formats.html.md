# Formats and gapless

Five container families, every common bit depth, and gapless trimming for the lossy ones.

## Supported formats

| Format | Extensions | Decoder | Notes |
|---|---|---|---|
| WAV / AIFF | .wav | symphonia | PCM 8/16/24/32-bit float and integer |
| FLAC | .flac | symphonia | 16/24-bit, any sample rate |
| MP3 | .mp3 | minimp3 | ~45% faster than the symphonia decoder |
| OGG Vorbis | .ogg | symphonia | — |
| AAC in MP4 | .m4a, .mp4 | symphonia | priming read from the edit list |

There is no extension allowlist: whatever the probe understands is accepted. A file with the wrong extension but valid contents still opens.

## Gapless

Lossy encoders add silence at both ends — a few thousand samples that are not part of the recording. Left in, every track gains a click of silence and offsets by a millisecond or two. sound_rs trims them:

- **MP3 and OGG** declare the delay and padding in the container.

- **AAC in MP4** records the priming in an edit list that symphonia parses but does not expose, so the library reads the `elst` box itself.

The result matches other decoders: a 60-second AAC file decodes to 2 646 016 frames and agrees with ffmpeg to 8·10⁻⁸ across the whole file.

```
import sound_rs, soundfile

ours, _ = sound_rs.load("track.flac")
theirs, _ = soundfile.read("track.flac", dtype="float32")
print(ours.shape == theirs.shape, abs(ours - theirs).max())   # (True, 0.0)
```

## Bit depth

Everything is converted to float32 on decode. 24-bit files keep their resolution; there is no silent truncation to 16-bit anywhere in the path. On output the device decides the sample format — f32, f64, i16, i32, u8 or u16 — and conversion happens in the callback.

## When a file will not open

All failures raise `sound_rs.SoundRsError`, a `RuntimeError` subclass.

```
import sound_rs

try:
    sound_rs.play_audio("broken.mp3")
except sound_rs.SoundRsError as error:
    print(error)   # Unsupported or corrupt audio file 'broken.mp3': ...
```

Decoding stops at the first corrupt packet rather than returning truncated audio, except for a file that ends early: a truncated tail is played up to the last good packet.

## Checking without decoding

```
import sound_rs

sound_rs.is_supported_format("maybe.mp3")   # True / False, header only
sound_rs.get_audio_info("maybe.mp3")        # (rate, channels, seconds)
```

Neither call decodes the audio, so both stay around a millisecond on a two-hour file.
