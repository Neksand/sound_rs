# Formats and gapless

Built-in WAV, FLAC, MP3, Ogg/Vorbis and AAC-in-MP4 decoding, with encoder-delay trimming where metadata is available.

## Supported formats

| Format | Extensions | Decoder | Notes |
|---|---|---|---|
| WAV | .wav | symphonia | PCM, including tested 24-bit and multichannel files |
| FLAC | .flac | symphonia | Lossless decode; supported codec layouts/rates |
| MP3 | .mp3 | minimp3 / symphonia | Full-file load uses minimp3; bytes/windows/streaming use Symphonia |
| OGG Vorbis | .ogg | symphonia | — |
| AAC in MP4 | .m4a, .mp4 | symphonia | priming read from the edit list |

There is no extension allowlist. Probing recognizes containers; a successful probe does not prove that every codec or packet inside is supported. Raw AAC and other containers need dedicated validation before relying on them.

## Gapless

Lossy encoders add silence at both ends — a few thousand samples that are not part of the recording. Left in, every track gains a click of silence and offsets by a millisecond or two. sound_rs trims them:

- **MP3 and OGG** declare the delay and padding in the container.

- **AAC in MP4** records the priming in an edit list that symphonia parses but does not expose, so the library reads the `elst` box itself.

Historical comparison: a 60-second AAC file decoded to 2 646 016 frames and agreed with ffmpeg to 8·10⁻⁸. AAC can retain less than one codec frame of tail padding; these measurements do not guarantee exact alignment for every file.

```
import sound_rs, soundfile

ours, _ = sound_rs.load("track.flac")
theirs, _ = soundfile.read("track.flac", dtype="float32")
print(ours.shape == theirs.shape, abs(ours - theirs).max())   # (True, 0.0)
```

## Bit depth

Everything is converted to float32 on decode. 24-bit files keep their resolution; there is no silent truncation to 16-bit anywhere in the path. On output the device decides the sample format — f32, f64, i16, i32, u8 or u16 — and conversion happens in the callback.

## When a file will not open

Library failures raise `sound_rs.SoundRsError`, a `RuntimeError` subclass. Python argument conversion can instead raise `TypeError` or `OverflowError`.

```
import sound_rs

try:
    sound_rs.play_audio("broken.mp3")
except sound_rs.SoundRsError as error:
    print(error)   # Unsupported or corrupt audio file 'broken.mp3': ...
```

Decoder errors normally fail the call. Unexpected end-of-file can return a decodable prefix, so a successful decode is not strict proof that the source file was complete.

## Checking without decoding

```
import sound_rs

sound_rs.is_supported_format("maybe.mp3")   # True / False, header only
sound_rs.get_audio_info("maybe.mp3")        # (rate, channels, seconds)
```

`is_supported_format` probes without decoding. `get_audio_info` uses complete header metadata when available, decodes a packet to discover missing format details, or decodes the file to determine an unknown duration. Its cost can grow with file length.
