"""Run with: python tests/smoke.py  (set SOUND_RS_PLAY=1 to also play audio)

Needs the extension on sys.path: build it with `maturin develop` or point
PYTHONPATH at a directory containing sound_rs.so / sound_rs*.so.
"""
import math
import os
import struct
import sys
import tempfile
import wave

import sound_rs

RATE = 44100
AMPLITUDE = 12000


def write_wav(path, seconds, rate=RATE, channels=1, width=2):
    frames = int(seconds * rate)
    body = bytearray()
    for i in range(frames):
        value = int(AMPLITUDE * math.sin(2 * math.pi * 440 * i / rate))
        for _ in range(channels):
            body += struct.pack("<h", value)
    with wave.open(path, "w") as w:
        w.setnchannels(channels)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(bytes(body))


def main():
    tmp = tempfile.mkdtemp()
    mono = os.path.join(tmp, "mono.wav")
    stereo = os.path.join(tmp, "stereo.wav")
    write_wav(mono, 0.1)
    write_wav(stereo, 0.1, channels=2)

    assert sound_rs.is_supported_format(mono), "wav should be supported"
    assert not sound_rs.is_supported_format(__file__), "a .py file is not audio"

    rate, channels, duration = sound_rs.get_audio_info(mono)
    assert (rate, channels) == (RATE, 1), (rate, channels)
    assert abs(duration - 0.1) < 0.01, duration

    rate, channels, duration = sound_rs.get_audio_info(stereo)
    assert channels == 2, channels
    assert abs(duration - 0.1) < 0.01, duration  # duration must not double with 2 channels

    try:
        sound_rs.get_audio_info(__file__)
    except RuntimeError:
        pass
    else:
        raise AssertionError("garbage input must raise RuntimeError")

    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None:
        # Decoded PCM must match what was written, not merely have the right shape.
        data, sr = sound_rs.load(stereo)
        assert sr == RATE, sr
        assert data.shape == (4410, 2), data.shape
        expected = (AMPLITUDE / 32768.0) * np.sin(
            2 * np.pi * 440 * np.arange(data.shape[0]) / RATE
        )
        error = float(np.abs(data[:, 0] - expected).max())
        assert error < 1e-3, f"decoded samples are wrong (max error {error})"
        assert np.abs(data[:, 0] - data[:, 1]).max() == 0, "channels were mixed up"

    if os.environ.get("SOUND_RS_PLAY") == "1":
        import time

        assert np is not None, "playback checks need numpy"

        sound_rs.play_audio(mono)

        # A mono array must play for its real duration, not half of it.
        samples = np.sin(2 * np.pi * 440 * np.arange(int(0.2 * RATE)) / RATE).astype(np.float32)
        start = time.perf_counter()
        sound_rs.play_array(samples, RATE)
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.2, f"playback returned early ({elapsed:.2f}s) — the tail was truncated"

        sound_rs.play_array(np.stack([samples, samples]), RATE)
        sound_rs.play_sine_wave(440.0, 0.1, RATE)

    print("smoke ok")


if __name__ == "__main__":
    sys.exit(main())
