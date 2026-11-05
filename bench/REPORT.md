# sound_rs benchmark

arm, Darwin 25.6.0, Python 3.14.7, reps=10, 2026-11-05T15:39:32

## batch

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 173.4 ms | 197.4 ms | 199.5 ms | 0.18s | 239MB |
| soundfile | 306.0 ms | 307.7 ms | 307.9 ms | 0.31s | 238MB |

## decode:long

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 85.1 ms | 94.7 ms | 97.9 ms | 0.08s | 237MB |
| soundfile | 103.7 ms | 134.2 ms | 135.0 ms | 0.10s | 238MB |

## decode:long.flac

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 170.7 ms | 179.2 ms | 182.9 ms | 0.16s | 237MB |
| soundfile | 307.5 ms | 328.8 ms | 334.8 ms | 0.31s | 238MB |

## decode:long.m4a

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 238.4 ms | 243.1 ms | 245.1 ms | 0.24s | 239MB |

## decode:long.mp3

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 163.7 ms | 171.2 ms | 174.8 ms | 0.17s | 237MB |
| soundfile | 190.8 ms | 270.8 ms | 273.9 ms | 0.20s | 238MB |

## decode:short

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 21.5 ms | 21.7 ms | 21.7 ms | 0.03s | 36MB |
| soundfile | 22.3 ms | 24.0 ms | 24.6 ms | 0.02s | 36MB |

## decode:short.flac

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 20.8 ms | 21.0 ms | 21.2 ms | 0.02s | 36MB |
| soundfile | 23.2 ms | 23.4 ms | 23.5 ms | 0.02s | 37MB |

## decode:short.m4a

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 21.5 ms | 23.4 ms | 24.4 ms | 0.02s | 37MB |

## decode:short.mp3

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 20.7 ms | 22.5 ms | 23.1 ms | 0.02s | 36MB |
| soundfile | 23.0 ms | 32.2 ms | 35.0 ms | 0.02s | 37MB |

## first_sound

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 76.1 ms | 76.6 ms | 76.7 ms | 0.02s | 43MB |
| soundfile+sounddevice | 494.1 ms | 494.3 ms | 494.3 ms | 0.33s | 256MB |

## import

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 775 µs | 854 µs | 864 µs | 0.00s | 26MB |
| sounddevice | 50.5 ms | 53.1 ms | 54.6 ms | 0.02s | 33MB |
| soundfile | 20.4 ms | 21.1 ms | 21.2 ms | 0.01s | 35MB |

## metadata:long

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 792 µs | 902 µs | 912 µs | 0.00s | 26MB |
| soundfile | 20.8 ms | 21.6 ms | 22.0 ms | 0.01s | 36MB |

## metadata:long.flac

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 816 µs | 900 µs | 921 µs | 0.00s | 26MB |
| soundfile | 20.7 ms | 23.1 ms | 23.6 ms | 0.01s | 36MB |

## metadata:long.m4a

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 1164 µs | 1299 µs | 1337 µs | 0.00s | 26MB |

## metadata:long.mp3

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 796 µs | 811 µs | 813 µs | 0.00s | 26MB |
| soundfile | 21.0 ms | 21.9 ms | 22.0 ms | 0.02s | 36MB |

## metadata:short

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 863 µs | 956 µs | 971 µs | 0.00s | 26MB |
| soundfile | 23.1 ms | 29.1 ms | 29.6 ms | 0.03s | 36MB |

## metadata:short.flac

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 868 µs | 1001 µs | 1061 µs | 0.00s | 26MB |
| soundfile | 22.1 ms | 23.2 ms | 23.2 ms | 0.03s | 36MB |

## metadata:short.m4a

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 1075 µs | 1166 µs | 1187 µs | 0.00s | 26MB |

## metadata:short.mp3

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 899 µs | 991 µs | 996 µs | 0.00s | 26MB |
| soundfile | 22.6 ms | 26.1 ms | 27.0 ms | 0.03s | 36MB |

## playback

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| one-shot | 3.170 s | 3.171 s | 3.171 s | 0.06s | 56MB |
| player | 3.220 s | 3.221 s | 3.221 s | 0.06s | 56MB |
| sounddevice | 3.230 s | 3.238 s | 3.239 s | 0.11s | 55MB |

## soak

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 115.246 s | 115.246 s | 115.246 s | 1.54s | 53MB |

## stream

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 5.078 s | 5.079 s | 5.079 s | 0.07s | 43MB |

