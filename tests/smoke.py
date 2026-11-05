"""Run with: python tests/smoke.py  (set SOUND_RS_PLAY=1 to also play audio)

Needs the extension on sys.path: build it with `maturin develop` or point
PYTHONPATH at a directory containing sound_rs.so / sound_rs*.so.
"""
import math
import os
import shutil
import struct
import subprocess
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

    if np is not None:
        # Batch decode must return the same arrays in the same order.
        batch = sound_rs.load_many([mono, stereo, mono])
        assert len(batch) == 3, len(batch)
        assert batch[0][0].shape == (4410, 1), batch[0][0].shape
        assert batch[1][0].shape == (4410, 2), batch[1][0].shape

        devices = sound_rs.list_devices()
        assert isinstance(devices, list), devices
        for index, name, is_default in devices:
            assert isinstance(index, int) and isinstance(name, str) and isinstance(is_default, bool)

    # The console script must exist and find the documentation.
    script = shutil.which("soundrs") or os.path.join(
        os.path.dirname(sys.executable), "soundrs"
    )
    if os.path.exists(script):
        help_text = subprocess.run(
            [script, "--help"], capture_output=True, text=True, check=True
        ).stdout
        assert "https://neksand.github.io/sound_rs/" in help_text, help_text
        assert "play" in help_text and "info" in help_text, help_text

        version = subprocess.run(
            [script, "--version"], capture_output=True, text=True, check=True
        ).stdout.strip()
        assert version.startswith("soundrs "), version

        info = subprocess.run(
            [script, "info", mono], capture_output=True, text=True, check=True
        ).stdout
        assert "44100 Hz" in info and "0:00.1" in info, info

        bad = subprocess.run([script, "nope"], capture_output=True, text=True)
        assert bad.returncode == 2, bad.returncode

    if os.environ.get("SOUND_RS_PLAY") == "1":
        import time

        assert np is not None, "playback checks need numpy"

        sound_rs.play_audio(mono)

        # The persistent player: duration/position/seek/pause without reopening
        # the device, and playback that lines up with the file length.
        with sound_rs.Player() as player:
            player.play_file(mono)
            assert abs(player.duration - 0.1) < 0.01, player.duration
            player.wait()
            assert abs(player.position - 0.1) < 0.02, player.position

            player.play_file(stereo)
            time.sleep(0.02)
            player.pause()
            frozen = player.position
            time.sleep(0.05)
            assert abs(player.position - frozen) < 1e-9, "pause must freeze the position"
            player.resume()
            player.seek(0.02)
            player.wait()

            player.volume = 0.5
            assert player.volume == 0.5
            player.play_array(np.zeros(4800, dtype=np.float32), 48000)
            player.wait()
            assert player.is_playing is False

        # A mono array must play for its real duration, not half of it.
        samples = np.sin(2 * np.pi * 440 * np.arange(int(0.2 * RATE)) / RATE).astype(np.float32)
        start = time.perf_counter()
        sound_rs.play_array(samples, RATE)
        elapsed = time.perf_counter() - start
        # Playback must not return before the audio ends; the upper bound only
        # catches a runaway wait (an 0.2 s clip cannot take a second).
        assert 0.2 <= elapsed < 1.0, f"playback took {elapsed:.2f}s for a 0.2s clip"

        sound_rs.play_array(np.stack([samples, samples]), RATE)
        sound_rs.play_sine_wave(440.0, 0.1, RATE)

    print("smoke ok")


if __name__ == "__main__":
    sys.exit(main())
