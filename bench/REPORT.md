# sound_rs benchmark

arm, Darwin 25.6.0, Python 3.14.7, reps=7, 2026-11-05T15:53:33

## batch

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 177.0 ms | 181.2 ms | 181.6 ms | 0.17s | 238MB |
| soundfile | 298.8 ms | 309.8 ms | 310.8 ms | 0.30s | 238MB |

## decode:long

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 123.3 ms | 134.5 ms | 136.0 ms | 0.10s | 237MB |
| soundfile | 104.3 ms | 106.1 ms | 106.2 ms | 0.10s | 237MB |

## decode:long.flac

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 169.0 ms | 169.8 ms | 169.8 ms | 0.16s | 237MB |
| soundfile | 303.3 ms | 308.2 ms | 309.2 ms | 0.30s | 238MB |

## decode:long.m4a

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 237.7 ms | 238.6 ms | 238.6 ms | 0.24s | 239MB |

## decode:long.mp3

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 162.1 ms | 162.8 ms | 162.8 ms | 0.16s | 237MB |
| soundfile | 177.2 ms | 178.9 ms | 179.3 ms | 0.18s | 238MB |

## decode:short

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 19.0 ms | 19.8 ms | 20.0 ms | 0.01s | 36MB |
| soundfile | 20.9 ms | 21.5 ms | 21.6 ms | 0.01s | 36MB |

## decode:short.flac

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 19.5 ms | 19.8 ms | 19.9 ms | 0.01s | 36MB |
| soundfile | 22.2 ms | 22.6 ms | 22.7 ms | 0.02s | 37MB |

## decode:short.m4a

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 20.4 ms | 20.9 ms | 21.0 ms | 0.01s | 36MB |

## decode:short.mp3

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 19.6 ms | 19.8 ms | 19.9 ms | 0.01s | 36MB |
| soundfile | 21.3 ms | 22.4 ms | 22.6 ms | 0.02s | 37MB |

## first_sound

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 73.1 ms | 73.2 ms | 73.2 ms | 0.02s | 43MB |
| soundfile+sounddevice | 484.1 ms | 485.3 ms | 485.4 ms | 0.33s | 256MB |

## import

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 795 µs | 890 µs | 914 µs | 0.00s | 26MB |
| sounddevice | 48.8 ms | 52.1 ms | 52.9 ms | 0.02s | 33MB |
| soundfile | 20.1 ms | 21.7 ms | 22.1 ms | 0.01s | 35MB |

## metadata:long

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 793 µs | 3029 µs | 3785 µs | 0.00s | 26MB |
| soundfile | 22.6 ms | 28.5 ms | 29.8 ms | 0.03s | 36MB |

## metadata:long.flac

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 823 µs | 875 µs | 885 µs | 0.00s | 26MB |
| soundfile | 20.7 ms | 20.8 ms | 20.8 ms | 0.01s | 36MB |

## metadata:long.m4a

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 1163 µs | 1196 µs | 1203 µs | 0.00s | 26MB |

## metadata:long.mp3

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 784 µs | 842 µs | 850 µs | 0.00s | 26MB |
| soundfile | 20.4 ms | 20.5 ms | 20.5 ms | 0.01s | 36MB |

## metadata:short

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 792 µs | 860 µs | 869 µs | 0.00s | 26MB |
| soundfile | 20.5 ms | 21.0 ms | 21.1 ms | 0.01s | 36MB |

## metadata:short.flac

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 794 µs | 838 µs | 844 µs | 0.00s | 26MB |
| soundfile | 20.6 ms | 20.7 ms | 20.7 ms | 0.01s | 36MB |

## metadata:short.m4a

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 986 µs | 1063 µs | 1076 µs | 0.00s | 26MB |

## metadata:short.mp3

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 818 µs | 844 µs | 847 µs | 0.00s | 26MB |
| soundfile | 20.7 ms | 21.1 ms | 21.2 ms | 0.01s | 36MB |

## playback

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| one-shot | 3.166 s | 3.168 s | 3.168 s | 0.05s | 56MB |
| player | 3.211 s | 3.211 s | 3.211 s | 0.05s | 56MB |
| sounddevice | 3.241 s | 3.243 s | 3.243 s | 0.11s | 55MB |

## soak

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 114.666 s | 114.666 s | 114.666 s | 1.58s | 53MB |

## stream

| variant | median | p95 | p99 | cpu | rss |
|---|---|---|---|---|---|
| sound_rs | 5.073 s | 5.073 s | 5.074 s | 0.06s | 43MB |

