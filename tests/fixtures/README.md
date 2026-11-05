# Decode fixtures

Generated once from a 0.25 s 440 Hz tone, committed so the decode regression
test does not depend on ffmpeg being installed.

```
python - <<'PY'
import math, struct, wave
rate, frames = 44100, int(0.25 * 44100)
samples = [int(12000 * math.sin(2*math.pi*440*i/rate)) for i in range(frames)]
for path, ch, width in (("tone.wav", 2, 2), ("tone24.wav", 2, 3), ("six.wav", 6, 2)):
    with wave.open(path, "w") as w:
        w.setnchannels(ch); w.setsampwidth(width); w.setframerate(rate)
        body = bytearray()
        for s in samples:
            for c in range(ch):
                v = int(s * (1.0 - 0.1 * c))
                body += struct.pack("<h", v) if width == 2 else (v * 256).to_bytes(3, "little", signed=True)
        w.writeframes(bytes(body))
PY

ffmpeg -i tone.wav tone.flac
ffmpeg -i tone.wav -c:a libmp3lame -b:a 128k tone.mp3
ffmpeg -i tone.wav -c:a aac tone.m4a
```

Notes:
- `tone24.wav` holds the same signal scaled by 256, so its float32 output is
  intentionally identical to `tone.wav` — the check that 24-bit is not truncated.
- `tone.m4a` decodes to 11264 frames, not 11025: AAC pads the tail to a frame
  boundary and the container carries no padding value. Priming is trimmed from
  the edit list; the residual is under one frame.
