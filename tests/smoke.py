"""Run with: python tests/smoke.py  (set SOUND_RS_PLAY=1 to also play audio)

Needs the extension on sys.path: build it with `maturin develop` or point
PYTHONPATH at a directory containing sound_rs.so / sound_rs*.so.
"""
import inspect
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
    # The Rust binding must match the documented and typed recording default.
    assert inspect.signature(sound_rs.Recorder.record_to).parameters["subtype"].default == "pcm16"

    tmp = tempfile.mkdtemp()
    FIX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
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
    except sound_rs.SoundRsError as error:
        assert error.code == "format", error.code
    else:
        raise AssertionError("garbage input must raise SoundRsError")

    try:
        sound_rs.get_audio_info(os.path.join(tmp, "missing.wav"))
    except sound_rs.SoundRsError as error:
        assert error.code == "io", error.code
    else:
        raise AssertionError("a missing file must raise SoundRsError")

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
        # Bytes decode must match the file decode exactly.
        with open(mono, "rb") as handle:
            raw = handle.read()
        from_bytes, rate_bytes = sound_rs.load_bytes(raw)
        from_file, rate_file = sound_rs.load(mono)
        assert rate_bytes == rate_file, (rate_bytes, rate_file)
        assert from_bytes.shape == from_file.shape, (from_bytes.shape, from_file.shape)
        assert (from_bytes == from_file).all(), "bytes decode differs from file decode"
        # A truncated file still yields its decodable prefix (by design); only
        # data with no audio at all must raise.
        for bad in (b"", b"not audio at all" * 20, raw[:44]):
            try:
                sound_rs.load_bytes(bad)
            except sound_rs.SoundRsError as error:
                assert error.code == "format", error.code
            else:
                raise AssertionError("bad bytes must raise SoundRsError")

        # load() options: rate, mono, window.
        assert sound_rs.load(stereo, mono=True)[0].shape == (4410, 1)
        resampled, rate = sound_rs.load(stereo, sample_rate=22050)
        assert (rate, resampled.shape) == (22050, (2205, 2)), (rate, resampled.shape)
        assert sound_rs.load(stereo, offset=0.05)[0].shape == (2205, 2)
        window, _ = sound_rs.load(stereo, offset=0.05, duration=0.02)
        assert window.shape == (882, 2), window.shape
        folded, rate = sound_rs.load(stereo, mono=True, sample_rate=16000)
        assert (rate, folded.shape) == (16000, (1600, 1)), (rate, folded.shape)

        # A window must carry the same samples as the head of a full decode,
        # including the encoder delay trim for MP3/AAC.
        for name in ("tone.wav", "tone.flac", "tone.m4a"):
            path = os.path.join(FIX, name)
            full, _ = sound_rs.load(path)
            head, _ = sound_rs.load(path, duration=0.1)
            n = min(len(head), len(full), 4000)
            assert n > 1000, (name, len(head))
            diff = float(np.abs(head[:n] - full[:n]).max())
            assert diff < 1e-3, f"{name} window is misaligned by {diff}"

        # MP3 windows go through symfonia's reader while a full load uses
        # minimp3, so compare the length, not the samples.
        mp3_head, _ = sound_rs.load(os.path.join(FIX, "tone.mp3"), duration=0.1)
        assert abs(len(mp3_head) - 4410) <= 1, len(mp3_head)

        # Zero-length windows are empty arrays, not errors.
        empty, _ = sound_rs.load(stereo, duration=0.0)
        assert empty.shape == (0, 2), empty.shape

        # Bytes decode must keep the gapless trim, AAC included.
        m4a = os.path.join(FIX, "tone.m4a")
        file_audio, _ = sound_rs.load(m4a)
        bytes_audio, _ = sound_rs.load_bytes(open(m4a, "rb").read())
        assert file_audio.shape == bytes_audio.shape, (
            file_audio.shape,
            bytes_audio.shape,
        )

        # Writing: (frames, channels) round-trips through load().
        out = os.path.join(tmp, "written.wav")
        original, original_rate = sound_rs.load(stereo)
        sound_rs.write(out, original, original_rate, subtype="pcm24")
        written, written_rate = sound_rs.load(out)
        assert written_rate == original_rate, (written_rate, original_rate)
        assert written.shape == original.shape, (written.shape, original.shape)
        assert np.abs(written - original).max() < 1e-5, "write/load round-trip drifted"
        sound_rs.write(out, original[:, 0], original_rate)      # 1D mono
        assert sound_rs.get_audio_info(out)[1] == 1
        try:
            sound_rs.write(os.path.join(tmp, "x.flac"), original, original_rate)
        except sound_rs.SoundRsError as error:
            assert error.code == "usage", error.code
        else:
            raise AssertionError("writing FLAC must fail with usage")
        try:
            sound_rs.write(out, original.astype("float64"), original_rate)
        except TypeError:
            pass
        else:
            raise AssertionError("float64 must be rejected")

        # mono folds every channel: 6ch mean equals the mono decode.
        six = os.path.join(FIX, "six.wav")
        folded, _ = sound_rs.load(six, mono=True)
        full, _ = sound_rs.load(six)
        assert folded.shape[1] == 1
        assert np.abs(folded[:, 0] - full.mean(axis=1)).max() < 1e-6

        # Batch decode must return the same arrays in the same order.
        batch = sound_rs.load_many([mono, stereo, mono])
        assert len(batch) == 3, len(batch)
        assert batch[0][0].shape == (4410, 1), batch[0][0].shape
        assert batch[1][0].shape == (4410, 2), batch[1][0].shape

        devices = sound_rs.list_devices()
        assert isinstance(devices, list), devices
        assert isinstance(sound_rs.list_input_devices(), list)
        if sound_rs.list_input_devices():
            recorder = sound_rs.Recorder()
            assert recorder.sample_rate > 0 and recorder.channels > 0
            del recorder
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

        # Queue: the handover must not cut the tail off the first track.
        player = sound_rs.Player()
        player.play_file(stereo)
        player.enqueue(mono)
        started = time.perf_counter()
        player.wait()
        elapsed = time.perf_counter() - started
        assert player.queue == [], player.queue
        assert elapsed > 0.18, f"queue handover cut the first track: {elapsed:.3f}s"

        # Enqueuing after the current track already finished still plays it.
        player.play_file(mono)
        player.wait()
        player.enqueue(stereo)
        player.wait()
        assert player.position > 0.05, player.position

        # Fed stream: five 0.1 s blocks then end_feed.
        player.begin_feed(RATE)
        block = np.zeros(4410, dtype=np.float32)
        for _ in range(5):
            player.feed(block)
        player.end_feed()
        player.wait()
        assert player.underruns == 0, f"a healthy fed stream reported {player.underruns} underruns"

        # Feeding after stop must fail fast instead of hanging.
        player.begin_feed(RATE)
        player.stop()
        try:
            player.feed(np.zeros(48000 * 2, dtype=np.float32))  # 2 s, ring is 1 s
        except sound_rs.SoundRsError as error:
            assert error.code == "usage", error.code
        else:
            raise AssertionError("feed() after stop() must raise")

        # Recording returns exactly the frames that were asked for.
        try:
            recorder = sound_rs.Recorder()
            take, take_rate = recorder.record(1.0)
            wanted = round(1.0 * take_rate)
            # Never more than asked; a slow device (Bluetooth) may start late.
            assert take.shape[0] <= wanted + 1, (take.shape, wanted)
            assert take.shape[0] > wanted * 0.7, (take.shape, wanted)
        except sound_rs.SoundRsError:
            pass  # no microphone in this environment

        # Soak: 100 cycles must not leak, and a healthy device must not report
        # underruns after the first start-ups.
        import resource

        def rss_mb():
            value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024

        player = sound_rs.Player()
        player.play_array(np.zeros(4800, dtype=np.float32), 48000)
        player.wait()
        baseline_rss = rss_mb()
        for _ in range(100):
            player.play_array(np.zeros(4800, dtype=np.float32), 48000)
            player.wait()
        grown = rss_mb() - baseline_rss
        assert grown < 20, f"RSS grew {grown:.1f} MB over 100 cycles"
        assert player.underruns < 5, f"{player.underruns} underruns on a 0.1 s buffer"
        print(f"soak ok (rss +{grown:.1f} MB, underruns {player.underruns})")

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
