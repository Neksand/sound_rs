# The player

`Player` owns one output stream for its whole lifetime. Every method below acts on that stream, so starting a track never reopens the device.

```
import sound_rs

player = sound_rs.Player()                 # opens the default device
player.play_file("first.flac")             # returns immediately
player.wait()

player.play_file("second.mp3")             # no device setup, starts at once
player.stop()
del player                                 # closes the stream
```

## Lifecycle

| State | What it means | Leaves it |
|---|---|---|
| idle | Stream open, nothing loaded | play_file ,  play_array |
| playing | Callback writes audio | pause ,  stop , end of source |
| paused | Callback writes silence, position frozen | resume ,  stop |
| finished | Source exhausted, stream still open | play_* ,  resume  (replays) |

## Controls

| Call | Effect |
|---|---|
| play_file(path) | Streams a file from disk. Returns immediately. |
| play_array(arr, rate) | Plays an in-memory array; resampled to the device once, up front. |
| pause() | Stops producing sound and freezes  position . |
| resume() | Continues from the paused position, or replays from the start if finished. |
| stop() | Clears the source and queue, rewinds to zero, keeps the stream open. Start a new source before resume/seek. |
| seek(seconds) | Jumps; works while paused. |
| wait() | Waits through the final audio drain (GIL released); returns while paused with the queue intact. |
| del player | Stops the decoder and closes the stream (or use  with ). |

## Seeking

Seeking a streamed file is accurate to the frame. Containers can only seek to a packet boundary, so the player reads the packet's real timestamp and drops the difference. A seek to `0.5` in a 16 kHz file lands on frame 8000, not on the nearest packet start.

Seeking preserves pause and queued files. Non-finite positions raise `SoundRsError(code="usage")`; a fed stream or stopped player cannot seek. After natural file completion, seek or resume restarts the current file, including the last file reached through a queue.

```
player.play_file("track.flac")
player.seek(90.5)          # 1:30.5
player.wait()
print(player.position)     # ~ track length, not the seek offset
```

## Volume

Applied in the callback, so it changes instantly and does not touch the decoded data. Range is 0 to 4; values outside are clamped. Volume 1.0 is unity.

```
player.volume = 0.25       # quiet
player.volume = 1.0        # unity
```

## Position and duration

Both are seconds, read from the audio clock, not from wall time. For files the duration comes from the container header and is available right after `play_file`. For arrays it is the array length divided by the sample rate.

```
player.play_file("track.mp3")
print(player.duration)     # 213.4
print(player.position)     # 0.0
print(player.is_playing)   # True
```

## Queue

`enqueue` lines up the next file; when the current one ends the decoder opens it and continues on the same stream. Rate and channel count may differ between tracks.

A queue requires a file source started with `play_file()`. Enqueuing on an idle, array or fed player raises `usage` instead of creating a queue that cannot advance.

```
with sound_rs.Player() as player:
    player.play_file("first.flac")
    player.enqueue("second.mp3")
    player.enqueue("third.wav")
    print(player.queue)        # ['second.mp3', 'third.wav']
    player.wait()              # plays all three
```

## Pushing audio from Python

For synthesis or a network stream, feed samples instead of a file. `feed` blocks while the buffer is full, so a fast producer cannot exhaust memory. If the buffer cannot drain for five seconds — a paused player, for example — it raises `SoundRsError` with `code="stalled"` instead of hanging.

The timeout measures consecutive time without ring progress, not the duration of a `feed()` call. Long healthy feeds are supported. Closing an empty feed completes cleanly; `wait()` includes the final output drain. `end_feed()` requires a fed source; after a finished feed, call `begin_feed()` to start another.

```
import numpy as np
import sound_rs

player = sound_rs.Player()
player.begin_feed(48000)
for _ in range(100):
    block = (0.2 * np.sin(np.arange(4800) * 0.057)).astype(np.float32)
    player.feed(block)         # returns when the buffer has room
player.end_feed()
player.wait()
```

## Threading and the GIL

Playback and decoding run on Rust threads; decoding, feeding and `wait()` release the GIL. Control methods can briefly join a decoder or query metadata while holding it. `player.wait()` in the main thread does not block an `asyncio` loop running in another thread, and `threading.Thread(target=sound_rs.play_audio, ...)` works as expected.

One player is one stream. Do not call methods on the same player from several threads at once; create one player per thread instead.

## Device selection

```
import sound_rs

print(sound_rs.list_devices())
player = sound_rs.Player(device="MacBook Pro Speakers")
print(player.device, player.sample_rate, player.channels)
```

The device is opened once, at construction. A backend failure is signaled atomically and reported as `SoundRsError(code="device")`; create a new Player to reopen it. The no-progress watchdog separately reports `stalled`. This is fault reporting, not automatic reconnection. The output callback never waits for its state mutex: contention produces silence for that callback, so hard real-time or zero-underrun behavior is not promised.
