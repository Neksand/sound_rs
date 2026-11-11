# The player

`Player` owns one output stream for its whole lifetime. Every method below acts on that stream, so starting a track never reopens the device.

```
import sound_rs

player = sound_rs.Player()                 # opens the default device
player.play_file("first.flac")             # returns immediately
player.wait()

player.play_file("second.mp3")             # reuses the open device
player.stop()
del player                                 # closes the stream
```

## Lifecycle

| State | What it means | Leaves it |
|---|---|---|
| idle | Stream open, nothing loaded | play_file ,  play_array |
| playing | Callback writes audio | pause ,  stop , end of source |
| paused | Callback writes silence, position frozen | resume ,  stop |
| finished | Source exhausted, stream still open | Start another source; a retained file/array can resume, a finished feed must restart |

## Controls

| Call | Effect |
|---|---|
| play_file(path) | Streams a file from disk. Returns immediately. |
| play_array(arr, rate) | Plays an in-memory array; resampled to the device once, up front. |
| pause() | Stops producing sound and freezes  position . |
| resume() | Continues an active source or replays a retained finished file/array. A stopped source or finished feed must restart. |
| stop() | Clears the source and queue, resets position, keeps the device stream open. |
| seek(seconds) | Jumps; works while paused. |
| wait() | Waits for completion/drain, or returns when paused/stopped (GIL released). |
| del player | Releases this reference; destroying the object stops the decoder and closes the stream. |

## Seeking

Seeking uses packet timestamps and drops decoded frames before the requested position. The source must be a file or an array, and the time finite; paused seeks preserve pause intent. Codec/container support determines positioning accuracy and seek cost.

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

`enqueue` requires an active file source started with `play_file`; array/feed modes reject it. When the current file ends, the decoder opens the next and continues on the same stream. Rate and channel count may differ between tracks. Pausing preserves the queue.

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

File decoding runs on a Rust worker. Playback waits and feed backpressure release the GIL, so other Python threads can run. Array conversion/resampling, metadata queries, player shutdown joins and device setup/teardown can still hold it. Calling `wait()` on an event-loop thread still blocks that loop until playback completes.

Create, use and destroy each `Player` or `Recorder` on the same thread. Even sequential use from a different thread is invalid. Separate threads may own separate instances; a worker may call `sound_rs.play_audio` because that function creates and disposes its own player there.

## Device selection

```
import sound_rs

print(sound_rs.list_devices())
player = sound_rs.Player(device="MacBook Pro Speakers")
print(player.device, player.sample_rate, player.channels)
```

The device is opened once, at construction. A backend failure is signaled atomically and reported as `SoundRsError(code="device")`; create a new Player to reopen it. The two-second no-progress watchdog separately reports `stalled`. This is fault reporting, not automatic reconnection. The output callback never waits for its state mutex: contention produces silence for that callback, so hard real-time or zero-underrun behavior is not promised. Physical disconnect behavior still needs device testing on each platform.
