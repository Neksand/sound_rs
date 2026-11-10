"""Page content. Imported by _build.py, which renders the HTML."""

from _build import code, note, page, warn


page(
    "index.html",
    "Overview",
    "sound_rs plays and decodes MP3, FLAC, WAV, OGG and M4A in Python from one wheel, with no ffmpeg, constant memory and ~11 ms to first sound.",
    """
<p class="kicker">Python · Rust · macOS / Windows / Linux</p>
<h1>Audio for Python, without ffmpeg.</h1>
<p class="lead">sound_rs plays and decodes the formats people actually have — MP3, FLAC, WAV,
OGG, M4A — from a single wheel. The decoders are compiled in, so there is nothing to install
besides the package itself.</p>

<div class="figures">
  <div><span>First sound</span><b>11 ms</b><i>on a 10-minute file</i></div>
  <div><span>Memory</span><b>43 MB</b><i>while streaming that file</i></div>
  <div><span>Import</span><b>1.1 ms</b><i>versus 32 ms for soundfile</i></div>
  <div><span>Platforms</span><b>3</b><i>Linux, Windows, macOS</i></div>
</div>

""" + code("""
pip install sound_rs
""", "bash") + code("""
import sound_rs

with sound_rs.Player() as player:
    player.play_file("music.flac")   # streams from disk, returns immediately
    player.seek(42.0)
    print(player.position, player.duration)
    player.wait()
""") + """
<h2 id="what-it-does">What it does</h2>
<ul>
  <li><strong>Playback</strong> of files and NumPy arrays, blocking or through a persistent
      player with pause, seek, volume and position.</li>
  <li><strong>Streaming decode</strong>: a two-hour file never sits in RAM, and the first
      sound arrives in milliseconds instead of after a full decode.</li>
  <li><strong>Decoding to arrays</strong> for analysis, one file at a time or in parallel.</li>
  <li><strong>Writing</strong> WAV in PCM16, PCM24 or float32; record straight to a file.</li>
  <li><strong>Metadata</strong> straight from the container header, without decoding the audio.</li>
  <li><strong>Typed</strong>: ships <code>sound_rs.pyi</code> and <code>py.typed</code>, so
      editors autocomplete every call.</li>
</ul>

<h2 id="what-it-does-not">What it does not do</h2>
<p>No effects, no mixing, no recording yet, and no plugin system. If you need a DAW-style
graph, this is the wrong tool; if you need to play or load audio reliably, it is the right
one. The <a href="benchmarks.html">benchmark page</a> also says where it is slower than the
alternatives.</p>

<h2 id="next">Where to go next</h2>
<ul>
  <li><a href="install.html">Installation</a> — platforms, wheels, Linux audio libraries.</li>
  <li><a href="quickstart.html">Quickstart</a> — the five things most code does.</li>
  <li><a href="cli.html">Command line</a> — <code>soundrs --help</code> without writing Python.</li>
  <li><a href="player.html">The player</a> — controls, seeking, lifecycle.</li>
  <li><a href="api.html">API reference</a> — every function and method.</li>
</ul>
""",
)

page(
    "install.html",
    "Installation",
    "How to install sound_rs: wheels for Linux, Windows and macOS, Python versions, from-source builds, and the Linux audio library it needs.",
    """
<p class="kicker">01 · Start</p>
<h1>Installation</h1>
<p class="lead">One command. Wheels carry the decoders, so no ffmpeg or development headers
are needed.</p>

""" + code("""
pip install sound_rs
""", "bash") + """
<h2 id="requirements">Requirements</h2>
<ul>
  <li>Python 3.10 or newer, CPython.</li>
  <li>Linux x86_64 or aarch64, Windows amd64, macOS arm64 or Intel — all ship wheels.</li>
  <li>NumPy only if you use <code>play_array</code>, <code>load</code> or <code>load_many</code>.</li>
</ul>

<table>
  <tr><th>Platform</th><th>Wheel tag</th><th>Python</th></tr>
  <tr><td>Linux x86_64</td><td class="num">cp310-abi3-manylinux_2_17_x86_64</td><td class="num">3.10+</td></tr>
  <tr><td>Linux aarch64</td><td class="num">cp310-abi3-manylinux_2_17_aarch64</td><td class="num">3.10+</td></tr>
  <tr><td>Windows amd64</td><td class="num">cp310-abi3-win_amd64</td><td class="num">3.10+</td></tr>
  <tr><td>macOS arm64</td><td class="num">cp310-abi3-macosx_11_0_arm64</td><td class="num">3.10+</td></tr>
  <tr><td>macOS Intel</td><td class="num">cp310-abi3-macosx_10_12_x86_64</td><td class="num">3.10+</td></tr>
</table>

<p>The wheels use the stable ABI, so one file covers every Python from 3.10 up — the same
wheel installs on 3.10 through 3.14. Anything else (musl, PyPy, other architectures) falls
back to the source distribution, which needs a Rust toolchain.</p>

<h2 id="from-source">From source</h2>
""" + code("""
git clone https://github.com/Neksand/sound_rs
cd sound_rs
pip install maturin
maturin develop --release
""", "bash") + """
<h2 id="linux-audio">Linux audio</h2>
<p>Playback uses ALSA through <code>cpal</code>. Most desktops already have the runtime
library; a minimal container or server needs it explicitly:</p>
""" + code("""
sudo apt install libasound2          # Debian / Ubuntu (libasound2t64 on 24.04+)
sudo dnf install alsa-lib            # Fedora
""", "bash") + note(
        "Decoding does not need ALSA",
        "If the library is missing, <code>import sound_rs</code> fails at load time, not on the "
        "first play. Install the package above, or use a build with playback disabled.",
    ) + """
<h2 id="check">Check it works</h2>
""" + code("""
import sound_rs

print(sound_rs.__version__)
print(sound_rs.list_devices())        # [(0, 'MacBook Pro Speakers', True), ...]
print(sound_rs.get_audio_info("song.mp3"))
""") + """
<p>From a terminal, <code>soundrs --help</code> prints the same check as a command, with the
documentation link. See <a href="cli.html">Command line</a>.</p>
<p>No sound, or an empty device list? See <a href="faq.html">Troubleshooting</a>.</p>
""",
)

page(
    "quickstart.html",
    "Quickstart",
    "The five things most sound_rs code does: play a file, control it, play an array, read metadata, decode to NumPy.",
    """
<p class="kicker">02 · Start</p>
<h1>Quickstart</h1>
<p class="lead">Five recipes. Each one runs as written.</p>

<h2 id="play">Play a file</h2>
""" + code("""
import sound_rs

sound_rs.play_audio("music.mp3")
""") + """
<p><code>play_audio</code> blocks until the track ends and releases the GIL while it does, so
other Python threads keep running. It opens the output device for that one call.</p>

<h2 id="player">Play with controls</h2>
""" + code("""
import sound_rs

with sound_rs.Player() as player:
    player.play_file("music.flac")   # returns immediately
    player.volume = 0.6
    player.seek(30.0)                # jump to 0:30
    player.pause()
    player.resume()
    player.wait()                    # block until it ends
""") + """
<p>The player keeps the stream open, so starting the next track costs milliseconds instead of
reopening the device. Details in <a href="player.html">The player</a>.</p>

<h2 id="arrays">Play a NumPy array</h2>
""" + code("""
import numpy as np
import sound_rs

rate = 44100
t = np.arange(rate * 2) / rate
mono = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

sound_rs.play_array(mono, rate)                       # 1D, mono
sound_rs.play_array(np.stack([mono, mono]), rate)     # 2D, (channels, frames)
""") + warn(
        "2D arrays are channels first",
        "The shape is <code>(channels, frames)</code>, not <code>(frames, channels)</code>. "
        "Passing <code>(frames, 2)</code> silently plays the wrong thing — check the shape.",
    ) + """
<h2 id="inspect">Inspect a file</h2>
""" + code("""
sample_rate, channels, duration = sound_rs.get_audio_info("music.m4a")
print(sample_rate, channels, duration)   # 44100 2 213.4
""") + """
<p>This reads the container header — about 1 ms regardless of file length. It decodes one
packet only for AAC, whose container does not record the channel count.</p>

<h2 id="decode">Decode to NumPy</h2>
""" + code("""
import sound_rs

audio, rate = sound_rs.load("music.flac")   # float32, (frames, channels)
print(audio.shape, rate)
""") + warn(
        "This puts the whole file in RAM",
        "A 10-minute stereo FLAC becomes 211 MB. For playback use "
        "<code>Player.play_file</code>, which streams and stays under a few megabytes.",
    ) + """
<h2 id="batch">Decode many files</h2>
""" + code("""
import sound_rs

clips = sound_rs.load_many(["a.flac", "b.flac", "c.flac"])   # all cores, GIL released
for audio, rate in clips:
    print(audio.shape, rate)
""") + """
<h2 id="devices">Choose a device</h2>
""" + code("""
import sound_rs

for index, name, is_default in sound_rs.list_devices():
    print(("*" if is_default else " "), index, name)

sound_rs.play_audio("music.mp3", device="MacBook Pro Speakers")
""") + """
<p>Continue with <a href="player.html">The player</a>.</p>
""",
)

page(
    "player.html",
    "The player",
    "The sound_rs Player: one stream kept open, play, pause, resume, stop, frame-accurate seek, volume, position and duration.",
    """
<p class="kicker">04 · Guides</p>
<h1>The player</h1>
<p class="lead"><code>Player</code> owns one output stream for its whole lifetime. Every
method below acts on that stream, so starting a track never reopens the device.</p>

""" + code("""
import sound_rs

player = sound_rs.Player()                 # opens the default device
player.play_file("first.flac")             # returns immediately
player.wait()

player.play_file("second.mp3")             # no device setup, starts at once
player.stop()
del player                                 # closes the stream
""") + note(
        "Use it as a context manager",
        "<code>with sound_rs.Player() as player:</code> closes the stream on exit, including "
        "when an exception is raised. Without it, the stream closes when the object is garbage "
        "collected, which you cannot predict.",
    ) + """
<h2 id="lifecycle">Lifecycle</h2>
<table>
  <tr><th>State</th><th>What it means</th><th>Leaves it</th></tr>
  <tr><td class="num">idle</td><td>Stream open, nothing loaded</td><td><code>play_file</code>, <code>play_array</code></td></tr>
  <tr><td class="num">playing</td><td>Callback writes audio</td><td><code>pause</code>, <code>stop</code>, end of source</td></tr>
  <tr><td class="num">paused</td><td>Callback writes silence, position frozen</td><td><code>resume</code>, <code>stop</code></td></tr>
  <tr><td class="num">finished</td><td>Source exhausted, stream still open</td><td><code>play_*</code>, <code>resume</code> (replays)</td></tr>
</table>

<h2 id="controls">Controls</h2>
<table>
  <tr><th>Call</th><th>Effect</th></tr>
  <tr><td class="num">play_file(path)</td><td>Streams a file from disk. Returns immediately.</td></tr>
  <tr><td class="num">play_array(arr, rate)</td><td>Plays an in-memory array; resampled to the device once, up front.</td></tr>
  <tr><td class="num">pause()</td><td>Stops producing sound and freezes <code>position</code>.</td></tr>
  <tr><td class="num">resume()</td><td>Continues from the paused position, or replays from the start if finished.</td></tr>
  <tr><td class="num">stop()</td><td>Rewinds to zero, keeps the stream open.</td></tr>
  <tr><td class="num">seek(seconds)</td><td>Jumps; works while paused.</td></tr>
  <tr><td class="num">wait()</td><td>Blocks until the source ends (GIL released).</td></tr>
  <tr><td class="num">del player</td><td>Stops the decoder and closes the stream (or use <code>with</code>).</td></tr>
</table>

<h2 id="seek">Seeking</h2>
<p>Seeking a streamed file is accurate to the frame. Containers can only seek to a packet
boundary, so the player reads the packet's real timestamp and drops the difference. A seek to
<code>0.5</code> in a 16 kHz file lands on frame 8000, not on the nearest packet start.</p>
""" + code("""
player.play_file("track.flac")
player.seek(90.5)          # 1:30.5
player.wait()
print(player.position)     # ~ track length, not the seek offset
""") + note(
        "Seeking past the end",
        "Requests beyond the track are clamped to the last frame. <code>position</code> keeps "
        "counting from the seek target, so <code>duration</code> and <code>position</code> stay "
        "in the same units.",
    ) + """
<h2 id="volume">Volume</h2>
<p>Applied in the callback, so it changes instantly and does not touch the decoded data.
Range is 0 to 4; values outside are clamped. Volume 1.0 is unity.</p>
""" + code("""
player.volume = 0.25       # quiet
player.volume = 1.0        # unity
""") + """
<h2 id="position">Position and duration</h2>
<p>Both are seconds, read from the audio clock, not from wall time. For files the duration
comes from the container header and is available right after <code>play_file</code>. For
arrays it is the array length divided by the sample rate.</p>
""" + code("""
player.play_file("track.mp3")
print(player.duration)     # 213.4
print(player.position)     # 0.0
print(player.is_playing)   # True
""") + """
<h2 id="queue">Queue</h2>
<p><code>enqueue</code> lines up the next file; when the current one ends the decoder opens it
and continues on the same stream. Rate and channel count may differ between tracks.</p>
""" + code("""
with sound_rs.Player() as player:
    player.play_file("first.flac")
    player.enqueue("second.mp3")
    player.enqueue("third.wav")
    print(player.queue)        # ['second.mp3', 'third.wav']
    player.wait()              # plays all three
""") + """
<h2 id="feed">Pushing audio from Python</h2>
<p>For synthesis or a network stream, feed samples instead of a file. <code>feed</code> blocks
while the buffer is full, so a fast producer cannot exhaust memory. If the buffer cannot drain
for five seconds — a paused player, for example — it raises
<code>SoundRsError</code> with <code>code="stalled"</code> instead of hanging.</p>
""" + code("""
import numpy as np
import sound_rs

player = sound_rs.Player()
player.begin_feed(48000)
for _ in range(100):
    block = (0.2 * np.sin(np.arange(4800) * 0.057)).astype(np.float32)
    player.feed(block)         # returns when the buffer has room
player.end_feed()
player.wait()
""") + note(
        "Feed at the device rate for clean joins",
        "Chunks are converted and resampled independently, so a non-device rate can click at "
        "chunk boundaries. Query <code>player.sample_rate</code> and feed that.",
    ) + """
<h2 id="threading">Threading and the GIL</h2>
<p>Playback and decoding run on Rust threads, and every blocking call releases the GIL.
<code>player.wait()</code> in the main thread does not block an <code>asyncio</code> loop
running in another thread, and <code>threading.Thread(target=sound_rs.play_audio, ...)</code>
works as expected.</p>
<p>One player is one stream. Do not call methods on the same player from several threads at
once; create one player per thread instead.</p>

<h2 id="devices">Device selection</h2>
""" + code("""
import sound_rs

print(sound_rs.list_devices())
player = sound_rs.Player(device="MacBook Pro Speakers")
print(player.device, player.sample_rate, player.channels)
""") + """
<p>The device is opened once, at construction. If a device disappears mid-playback, the
player raises <code>SoundRsError</code> instead of hanging: playback stalls for two seconds
without progress, then fails.</p>
""",
)

page(
    "streaming.html",
    "Streaming and memory",
    "How sound_rs streams a file with constant memory, why a 10-minute file costs 43 MB instead of 255 MB, and when to use load() instead.",
    """
<p class="kicker">05 · Guides</p>
<h1>Streaming and memory</h1>
<p class="lead"><code>play_file</code> never loads the track. A decoder thread fills a one-second
ring buffer, and the audio callback drains it. Memory stays flat no matter how long the file is.</p>

<h2 id="how">How it works</h2>
<ol>
  <li>The decoder thread reads a packet, decodes it to float32, converts the channel count
      and resamples to the device rate.</li>
  <li>The result goes into a ring buffer sized for one second of device audio.</li>
  <li>The audio callback pops samples from that buffer. When it is empty, the callback
      writes silence and keeps the stream alive.</li>
  <li>Seeking replaces the ring buffer and repositions the decoder; the old audio that was
      still queued is discarded and does not count towards the new position.</li>
</ol>
<p>Because the decoder runs ahead by at most a second, the first sound arrives as soon as the
first packet is decoded, not when the file ends.</p>

<h2 id="numbers">Numbers</h2>
<p>Ten-minute stereo FLAC, 44.1 kHz, measured with <code>bench/run.py</code> on an Apple M5 Pro:</p>
<table>
  <tr><th></th><th class="num">sound_rs</th><th class="num">soundfile + sounddevice</th></tr>
  <tr><td>Time to first sound</td><td class="num"><strong>11.3 ms</strong></td><td class="num">302 ms</td></tr>
  <tr><td>Peak RSS</td><td class="num"><strong>43 MB</strong></td><td class="num">256 MB</td></tr>
</table>
<p>The competitor number is not a flaw in those libraries: their API takes an array, so the
file has to be decoded before playback can start. Streaming is a different design.</p>

<h2 id="choose">play_file or load?</h2>
<table>
  <tr><th>You want</th><th>Use</th><th>Memory</th></tr>
  <tr><td>Play a file</td><td class="num">Player.play_file</td><td>constant, ~1 s of audio</td></tr>
  <tr><td>Analyse the samples</td><td class="num">load</td><td>whole file as float32</td></tr>
  <tr><td>Play an array you already have</td><td class="num">play_array</td><td>one copy at device format</td></tr>
  <tr><td>Play a file, need the samples too</td><td class="num">load, then play_array</td><td>whole file</td></tr>
</table>
<p>A 10-minute stereo file is 211 MB in float32. That is fine for analysis and wasteful for
playback.</p>

""" + code("""
import sound_rs

# streaming: memory stays flat, start is immediate
with sound_rs.Player() as player:
    player.play_file("lecture-2h.m4a")
    player.wait()

# analysis: everything in RAM, decoded in parallel with other files
clips = sound_rs.load_many(["a.flac", "b.flac"])
""") + """
<h2 id="caveats">Caveats</h2>
<ul>
  <li><code>duration</code> for a streamed file comes from the header; a VBR MP3 without a
      frame count reports <code>None</code> until the decoder reaches the end.</li>
  <li>Seeking in OGG and MP4 is exact; seeking in a stream without an index (rare) may take
      longer because the decoder has to walk forward.</li>
  <li>Only playback streams. <code>load</code> and <code>load_many</code> always decode
      fully — that is their job.</li>
</ul>
""",
)

page(
    "formats.html",
    "Formats and gapless",
    "Which formats and bit depths sound_rs decodes, how encoder delay and padding are trimmed, and what the MP3 offset against libsndfile means.",
    """
<p class="kicker">06 · Guides</p>
<h1>Formats and gapless</h1>
<p class="lead">Five container families, every common bit depth, and gapless trimming for the
lossy ones.</p>

<h2 id="supported">Supported formats</h2>
<table>
  <tr><th>Format</th><th>Extensions</th><th>Decoder</th><th>Notes</th></tr>
  <tr><td>WAV / AIFF</td><td class="num">.wav</td><td>symphonia</td><td>PCM 8/16/24/32-bit float and integer</td></tr>
  <tr><td>FLAC</td><td class="num">.flac</td><td>symphonia</td><td>16/24-bit, any sample rate</td></tr>
  <tr><td>MP3</td><td class="num">.mp3</td><td>minimp3</td><td>~45% faster than the symphonia decoder</td></tr>
  <tr><td>OGG Vorbis</td><td class="num">.ogg</td><td>symphonia</td><td>—</td></tr>
  <tr><td>AAC in MP4</td><td class="num">.m4a, .mp4</td><td>symphonia</td><td>priming read from the edit list</td></tr>
</table>
<p>There is no extension allowlist: whatever the probe understands is accepted. A file with
the wrong extension but valid contents still opens.</p>

<h2 id="gapless">Gapless</h2>
<p>Lossy encoders add silence at both ends — a few thousand samples that are not part of the
recording. Left in, every track gains a click of silence and offsets by a millisecond or two.
sound_rs trims them:</p>
<ul>
  <li><strong>MP3 and OGG</strong> declare the delay and padding in the container.</li>
  <li><strong>AAC in MP4</strong> records the priming in an edit list that symphonia parses
      but does not expose, so the library reads the <code>elst</code> box itself.</li>
</ul>
<p>The result matches other decoders: a 60-second AAC file decodes to 2 646 016 frames and
agrees with ffmpeg to 8·10⁻⁸ across the whole file.</p>

""" + code("""
import sound_rs, soundfile

ours, _ = sound_rs.load("track.flac")
theirs, _ = soundfile.read("track.flac", dtype="float32")
print(ours.shape == theirs.shape, abs(ours - theirs).max())   # (True, 0.0)
""") + warn(
        "MP3 is offset by 47 samples",
        "Against libsndfile and ffmpeg, MP3 output starts 47 samples (~1 ms) earlier, because "
        "decoders trim the encoder delay differently. Lengths match exactly and sample values "
        "agree to 1.6·10⁻⁵ once aligned. It is inaudible; it matters if you compare PCM "
        "sample-by-sample across libraries.",
    ) + """
<h2 id="depth">Bit depth</h2>
<p>Everything is converted to float32 on decode. 24-bit files keep their resolution; there is
no silent truncation to 16-bit anywhere in the path. On output the device decides the sample
format — f32, f64, i16, i32, u8 or u16 — and conversion happens in the callback.</p>

<h2 id="errors">When a file will not open</h2>
<p>All failures raise <code>sound_rs.SoundRsError</code>, a <code>RuntimeError</code> subclass.</p>
""" + code("""
import sound_rs

try:
    sound_rs.play_audio("broken.mp3")
except sound_rs.SoundRsError as error:
    print(error)   # Unsupported or corrupt audio file 'broken.mp3': ...
""") + """
<p>Decoding stops at the first corrupt packet rather than returning truncated audio, except
for a file that ends early: a truncated tail is played up to the last good packet.</p>

<h2 id="probe">Checking without decoding</h2>
""" + code("""
import sound_rs

sound_rs.is_supported_format("maybe.mp3")   # True / False, header only
sound_rs.get_audio_info("maybe.mp3")        # (rate, channels, seconds)
""") + """
<p>Neither call decodes the audio, so both stay around a millisecond on a two-hour file.</p>
""",
)

page(
    "numpy.html",
    "NumPy and arrays",
    "Array shapes, dtype rules, playing and decoding arrays with sound_rs, batch decoding, and the mistakes that cost the most time.",
    """
<p class="kicker">07 · Guides</p>
<h1>NumPy and arrays</h1>
<p class="lead">Two directions: arrays in for playback, arrays out for analysis. Both use
float32 samples in the range −1.0 to 1.0.</p>

<h2 id="shape">Shapes</h2>
<table>
  <tr><th>Call</th><th>Shape</th><th>Meaning</th></tr>
  <tr><td class="num">play_array(mono, rate)</td><td class="num">(frames,)</td><td>1D, one channel</td></tr>
  <tr><td class="num">play_array(stereo, rate)</td><td class="num">(2, frames)</td><td>2D, channel first</td></tr>
  <tr><td class="num">load(path)</td><td class="num">(frames, channels)</td><td>2D, frames first</td></tr>
</table>
""" + warn(
        "The two directions have opposite layouts",
        "<code>play_array</code> takes <code>(channels, frames)</code> because that is how a "
        "matrix of channels is usually built; <code>load</code> returns "
        "<code>(frames, channels)</code> so <code>audio[i]</code> is one frame and "
        "<code>audio[:, 0]</code> is the left channel. Round-tripping needs a transpose: "
        "<code>sound_rs.play_array(audio.T, rate)</code>.",
    ) + """
<h2 id="play">Playing arrays</h2>
""" + code("""
import numpy as np
import sound_rs

rate = 48000
seconds = 3
t = np.arange(rate * seconds) / rate

left = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
right = (0.2 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)

sound_rs.play_array(np.stack([left, right]), rate)
""") + """
<p>The array is converted to the device layout once, before playback starts: channel folding,
then one FFT resample if the rates differ. Playing a 48 kHz array on a 44.1 kHz device costs a
few milliseconds and sounds correct — a naive nearest-neighbour resampler would alias.</p>

<h2 id="load">Decoding to arrays</h2>
""" + code("""
import sound_rs

audio, rate = sound_rs.load("track.mp3")
print(audio.shape, audio.dtype, rate)        # (9437184, 2) float32 44100
print(audio[:, 0].max())                     # peak of the left channel
""") + """
<p><code>load</code> returns the whole file. Ten minutes of stereo float32 is 211 MB, so for
long files prefer streaming playback or decode one channel at a time from your own pipeline.</p>

<h2 id="batch">Batch decoding</h2>
""" + code("""
import glob
import sound_rs

paths = sorted(glob.glob("dataset/*.flac"))
clips = sound_rs.load_many(paths)             # uses every core
print(len(clips), clips[0][0].shape)
""") + note(
        "Threads",
        "<code>load_many(paths, threads=2)</code> limits the pool. The default is one worker "
        "per core. Ordering matches the input list, and the first failing file raises.",
    ) + """
<h2 id="pitfalls">Pitfalls</h2>
<ul>
  <li><strong>float64 arrays are rejected.</strong> NumPy's <code>np.sin</code> returns
      float64; cast with <code>.astype(np.float32)</code> or use <code>dtype=np.float32</code>.</li>
  <li><strong>Integer arrays are rejected.</strong> Convert to float32 and scale to −1…1
      yourself; the library will not guess whether your int16 is scaled.</li>
  <li><strong>Non-contiguous arrays are copied.</strong> A slice like <code>audio[::2]</code>
      still works, but you pay for the copy.</li>
  <li><strong><code>play_array</code> copies the data.</strong> The stream callback needs an
      owned buffer; that is one copy of the track, not two.</li>
</ul>
""",
)

page(
    "migration.html",
    "Coming from another library",
    "Equivalent sound_rs code for sounddevice, soundfile, pydub and playsound, plus what changes when there is no ffmpeg and no array requirement.",
    """
<p class="kicker">08 · Guides</p>
<h1>Coming from another library</h1>
<p class="lead">Most Python audio code is three lines. Here is what those three lines become.</p>

<h2 id="sounddevice">From sounddevice</h2>
""" + code("""
# before
import sounddevice as sd
import soundfile as sf

data, rate = sf.read("track.flac")
sd.play(data, rate)
sd.wait()

# after
import sound_rs

sound_rs.play_audio("track.flac")
""") + """
<p>You lose the array, which you did not need for playback, and gain no ffmpeg and no
per-call device setup. If you do need the samples, <code>sound_rs.load</code> returns them in
the same shape as <code>sf.read</code>.</p>
<p>For a persistent stream with pause and seek, <code>sounddevice</code> needs an
<code>OutputStream</code> plus a callback queue; <code>sound_rs.Player</code> is that in one
object.</p>

<h2 id="soundfile">From soundfile</h2>
""" + code("""
# before
import soundfile as sf
data, rate = sf.read("track.m4a", dtype="float32")

# after
import sound_rs
data, rate = sound_rs.load("track.m4a")
""") + note(
        "AAC and M4A",
        "libsndfile cannot read MP4/AAC. That is the case where this swap is not optional.",
    ) + """
<h2 id="pydub">From pydub</h2>
""" + code("""
# before — needs an ffmpeg binary on PATH
from pydub import AudioSegment
from pydub.playback import play

segment = AudioSegment.from_file("track.mp3")
play(segment)

# after
import sound_rs

sound_rs.play_audio("track.mp3")
""") + """
<p>No subprocess, no temporary WAV, no ffmpeg install step for your users. The trade: pydub
also does cutting, concatenation and format conversion, which sound_rs does not.</p>

<h2 id="playsound">From playsound / simpleaudio</h2>
""" + code("""
# before — one shot, no controls, format support varies by platform
from playsound import playsound

playsound("track.mp3")

# after
import sound_rs

sound_rs.play_audio("track.mp3")
""") + """
<p>Same one-line call, but the format support is the same on every platform and the call
releases the GIL.</p>

<h2 id="tradeoffs">What you gain, what you give up</h2>
<table>
  <tr><th></th><th>sound_rs</th><th>soundfile + sounddevice</th><th>pydub</th></tr>
  <tr><td>Formats</td><td>WAV, FLAC, MP3, OGG, M4A</td><td>WAV, FLAC, MP3; no AAC</td><td>anything ffmpeg reads</td></tr>
  <tr><td>External tools</td><td>none</td><td>none</td><td>ffmpeg binary</td></tr>
  <tr><td>Playback without loading</td><td>yes</td><td>no</td><td>no</td></tr>
  <tr><td>Editing (cut, mix, convert)</td><td>no</td><td>no</td><td>yes</td></tr>
  <tr><td>Platforms</td><td>3, wheels</td><td>many</td><td>any with ffmpeg</td></tr>
</table>
""",
)

page(
    "api.html",
    "API reference",
    "Every function and method in sound_rs: signatures, arguments, return values, exceptions and a runnable example.",
    """
<p class="kicker">09 · Reference</p>
<h1>API reference</h1>
<p class="lead">Nine functions, one class. Everything raises <code>sound_rs.SoundRsError</code>,
a subclass of <code>RuntimeError</code>, on failure.</p>

<h2 id="module">Module</h2>
<p><code class="num">sound_rs.__version__</code> — the installed version as a string, for example
<code class="num">"0.2.1"</code>.</p>
<p><code>class sound_rs.SoundRsError(RuntimeError)</code> — the only exception type the library
raises. Catch it instead of the individual causes.</p>

<h2 id="play-audio">play_audio</h2>
""" + code("""
sound_rs.play_audio(file_path: str, device: str | None = None) -> None
""") + """
<p>Plays a file and blocks until it ends. Releases the GIL, so other threads keep running.
Opens the output device for this call, which costs about 0.1–0.2 s before the first sound —
use <a href="#player">Player</a> for repeated playback.</p>
<ul>
  <li><code>file_path</code> — any container the probe accepts.</li>
  <li><code>device</code> — exact device name from <code>list_devices()</code>; the default
      device when omitted.</li>
</ul>
<p>Raises <code>SoundRsError</code> for an unreadable file, a missing device, or a device that
disconnects mid-playback.</p>
""" + code("""
import sound_rs

sound_rs.play_audio("music.flac")
sound_rs.play_audio("music.flac", device="MacBook Pro Speakers")
""") + """
<h2 id="is-supported-format">is_supported_format</h2>
""" + code("""
sound_rs.is_supported_format(file_path: str) -> bool
""") + """
<p>Probes the container header. Does not decode, does not check that the audio data is intact,
and does not raise for a missing file — it returns <code>False</code>. About 1 ms on any file
size.</p>

<h2 id="get-audio-info">get_audio_info</h2>
""" + code("""
sound_rs.get_audio_info(file_path: str) -> tuple[int, int, float]
""") + """
<p>Returns <code>(sample_rate, channels, duration_seconds)</code>. Reads the header, so the cost
does not depend on file length. For AAC the container has no channel layout, so one packet is
decoded to learn it (~0.15 ms).</p>
<p>Duration is <code>None</code> in the Python sense only when the header has no frame count;
in that case the file is decoded to measure it.</p>

<h2 id="load">load</h2>
""" + code("""
sound_rs.load(file_path: str) -> tuple[numpy.ndarray, int]
""") + """
<p>Decodes the whole file to float32 with shape <code>(frames, channels)</code>, and returns it
with the sample rate. Requires NumPy. Peak memory is the decoded audio: 10 minutes of stereo
is 211 MB.</p>
<p>Raises <code>SoundRsError</code> for an unreadable file or when the build has no NumPy
support.</p>

<h2 id="load-many">load_many</h2>
""" + code("""
sound_rs.load_many(paths: Sequence[str], threads: int | None = None)
    -> list[tuple[numpy.ndarray, int]]
""") + """
<p>Decodes in parallel with a Rayon pool, releasing the GIL. Order matches <code>paths</code>.
<code>threads=None</code> uses one worker per core. The first file that fails raises, and the
rest of the batch is abandoned.</p>
""" + code("""
clips = sound_rs.load_many(["a.flac", "b.flac"], threads=2)
""") + """
<h2 id="play-array">play_array</h2>
""" + code("""
sound_rs.play_array(array: numpy.ndarray, sample_rate: int) -> None
""") + """
<p>Plays a float32 array: <code>(frames,)</code> for mono or <code>(channels, frames)</code>
for multichannel. Converts to the device layout and resamples if the rate differs, then blocks
until playback ends.</p>
<p>Raises <code>TypeError</code> for anything that is not a 1D or 2D float32 ndarray, and
<code>SoundRsError</code> for a device failure.</p>

<h2 id="play-sine-wave">play_sine_wave</h2>
""" + code("""
sound_rs.play_sine_wave(frequency: float, duration: float, sample_rate: int) -> None
""") + """
<p>Generates a mono sine tone at half scale and plays it. Useful for checking that audio
output works at all.</p>
""" + code("""
sound_rs.play_sine_wave(440.0, 1.0, 44100)
""") + """
<h2 id="cli">cli</h2>
""" + code("""
sound_rs.cli() -> int
""") + """
<p>Entry point for the <code>soundrs</code> console script; reads <code>sys.argv</code> and
returns an exit code. You do not call it from Python — see
<a href="cli.html">Command line</a>.</p>

<h2 id="write">write</h2>
""" + code("""
sound_rs.write(file_path: str, audio, sample_rate: int, subtype: str = "pcm16") -> None
""") + """
<p>Writes float32 audio to a WAV file. The array is <code>(frames,)</code> for mono or
<code>(frames, channels)</code> — the same layout <code>load</code> returns, so
<code>sound_rs.write(path, *sound_rs.load(src))</code> round-trips. Samples outside −1…1 are
clipped. Subtypes: <code>pcm16</code>, <code>pcm24</code>, <code>float32</code>.</p>
<p>Only <code>.wav</code> is accepted so far; any other extension raises
<code>SoundRsError</code> with <code>code="usage"</code>. Returns nothing.</p>
""" + code("""
audio, rate = sound_rs.load("take.flac")
sound_rs.write("take.wav", audio, rate, subtype="pcm24")
""") + """
<h2 id="load-bytes">load_bytes</h2>
""" + code("""
sound_rs.load_bytes(data: bytes) -> tuple[numpy.ndarray, int]
""") + """
<p>Decodes audio that is already in memory. Same shape, rate and gapless trimming as
<code>load</code>; the MP4 edit list is read from the buffer as well. Corrupt data raises
<code>SoundRsError</code> with <code>code="format"</code>.</p>

<h2 id="list-devices">list_devices</h2>
""" + code("""
sound_rs.list_devices() -> list[tuple[int, str, bool]]
""") + """
<p>Output devices as <code>(index, name, is_default)</code>. The index is only for display —
pass the name to <code>play_audio</code> or <code>Player</code>.</p>

<h2 id="recorder">Recorder</h2>
""" + code("""
sound_rs.Recorder(device: str | None = None)
""") + """
<p>Opens an input device. Raises <code>SoundRsError</code> with <code>code="device"</code> when
there is no microphone or the OS denies access.</p>
<table>
  <tr><th>Member</th><th>Returns</th><th>Notes</th></tr>
  <tr><td class="num">start()</td><td>None</td><td>Starts capturing</td></tr>
  <tr><td class="num">stop()</td><td>(ndarray, rate)</td><td>Stops and returns what was captured</td></tr>
  <tr><td class="num">record(seconds)</td><td>(ndarray, rate)</td><td>start + wait + stop</td></tr>
  <tr><td class="num">record_to(path, seconds, subtype)</td><td>None</td><td>record and write a WAV in one call</td></tr>
  <tr><td class="num">device / sample_rate / channels</td><td>str / int / int</td><td>Device properties</td></tr>
</table>

<h2 id="list-input-devices">list_input_devices</h2>
""" + code("""
sound_rs.list_input_devices() -> list[tuple[int, str, bool]]
""") + """
<p>Input devices as <code>(index, name, is_default)</code>. The name is what
<code>Recorder(device=...)</code> accepts.</p>

<h2 id="player">Player</h2>
""" + code("""
sound_rs.Player(device: str | None = None)
""") + """
<p>Opens one output stream and keeps it open. Device name matching is exact; the default device
is used when omitted. The stream closes when the object is deleted or when the
<code>with</code> block exits.</p>

<h3 id="methods">Methods</h3>
<table>
  <tr><th>Signature</th><th>Returns</th><th>Notes</th></tr>
  <tr><td class="num">play_file(path)</td><td>None</td><td>Streams from disk, returns immediately</td></tr>
  <tr><td class="num">play_array(array, rate)</td><td>None</td><td>1D or 2D float32</td></tr>
  <tr><td class="num">pause()</td><td>None</td><td>Position freezes</td></tr>
  <tr><td class="num">resume()</td><td>None</td><td>Continues, or replays if finished</td></tr>
  <tr><td class="num">stop()</td><td>None</td><td>Rewinds to zero</td></tr>
  <tr><td class="num">seek(seconds)</td><td>None</td><td>Frame-accurate, clamped to the track</td></tr>
  <tr><td class="num">wait()</td><td>None</td><td>Blocks until the source ends; GIL released</td></tr>
  <tr><td class="num">enqueue(path)</td><td>None</td><td>Plays after the current track; up to 128 entries</td></tr>
  <tr><td class="num">clear_queue()</td><td>None</td><td>Drops everything queued</td></tr>
  <tr><td class="num">begin_feed(rate)</td><td>None</td><td>Switches to samples pushed from Python</td></tr>
  <tr><td class="num">feed(array)</td><td>None</td><td>Blocks while the ring is full; call from one thread</td></tr>
  <tr><td class="num">end_feed()</td><td>None</td><td>No more samples; finishes when the ring drains</td></tr>
</table>

<h3 id="properties">Properties</h3>
<table>
  <tr><th>Name</th><th>Type</th><th>Notes</th></tr>
  <tr><td class="num">device</td><td>str</td><td>Name the player opened</td></tr>
  <tr><td class="num">sample_rate</td><td>int</td><td>Device rate</td></tr>
  <tr><td class="num">channels</td><td>int</td><td>Device channel count</td></tr>
  <tr><td class="num">volume</td><td>float</td><td>Read/write, clamped to 0…4</td></tr>
  <tr><td class="num">position</td><td>float</td><td>Seconds played</td></tr>
  <tr><td class="num">duration</td><td>float | None</td><td>Seconds; <code>None</code> until known</td></tr>
  <tr><td class="num">is_playing</td><td>bool</td><td><code>False</code> when paused, stopped or finished</td></tr>
  <tr><td class="num">queue</td><td>list[str]</td><td>Paths waiting behind the current track</td></tr>
  <tr><td class="num">underruns</td><td>int</td><td>Callbacks that starved after audio started; healthy streams stay at 0</td></tr>
  <tr><td class="num">latency</td><td>float | None</td><td>Last device latency in milliseconds</td></tr>
</table>

""" + code("""
import sound_rs

with sound_rs.Player() as player:
    player.play_file("track.flac")
    while player.position < 10:
        pass                      # do work while it plays
    player.seek(120.0)
    player.wait()
""") + """
<h2 id="exceptions">Exceptions</h2>
<table>
  <tr><th>Raised by</th><th>When</th></tr>
  <tr><td class="num">SoundRsError</td><td>Unreadable file, no output device, device disconnected, stalled stream, no NumPy support</td></tr>
  <tr><td class="num">TypeError</td><td>An array that is not a 1D/2D float32 ndarray</td></tr>
</table>
<p><code>SoundRsError</code> subclasses <code>RuntimeError</code>, so existing
<code>except RuntimeError</code> handlers keep working.</p>
""",
)

page(
    "benchmarks.html",
    "Benchmarks",
    "How sound_rs compares with libsndfile and sounddevice: methodology, full tables, and the cases where it loses.",
    """
<p class="kicker">10 · Reference</p>
<h1>Benchmarks</h1>
<p class="lead">Numbers with the method next to them, including the rows where the library is
slower.</p>

<h2 id="method">Method</h2>
<ul>
  <li>Every measurement runs in a <strong>separate process</strong>: peak RSS is process-wide,
      so timing and memory cannot share one.</li>
  <li>Median and p95 over repeated runs; the corpus is generated by the harness so it is
      reproducible.</li>
  <li>Correctness is checked in the same run: decoded PCM is compared with libsndfile.</li>
  <li>Hardware: Apple M5 Pro, macOS 26.6.2, Python 3.14.7, release build with
      <code>lto = "fat"</code> and <code>opt-level = 3</code>.</li>
</ul>

<h2 id="scoreboard">Scoreboard</h2>
<table>
  <tr><th>Axis</th><th class="num">sound_rs</th><th class="num">Competitor</th><th class="num">Ratio</th></tr>
  <tr><td>Import, cold process</td><td class="num"><strong>1.1 ms</strong></td><td class="num">31.8 ms (soundfile)</td><td class="num">29×</td></tr>
  <tr><td>Metadata, 60 s file</td><td class="num"><strong>1.2 ms</strong></td><td class="num">31.7 ms</td><td class="num">26×</td></tr>
  <tr><td>Decode WAV</td><td class="num"><strong>130 ms</strong></td><td class="num">157 ms</td><td class="num">1.2×</td></tr>
  <tr><td>Decode FLAC</td><td class="num"><strong>265 ms</strong></td><td class="num">455 ms</td><td class="num">1.7×</td></tr>
  <tr><td>Decode MP3</td><td class="num"><strong>241 ms</strong></td><td class="num">268 ms</td><td class="num">1.1×</td></tr>
  <tr><td>Decode AAC</td><td class="num">349 ms</td><td class="num">not supported</td><td class="num">—</td></tr>
  <tr><td>Time to first sound, 10 min</td><td class="num"><strong>11.3 ms</strong></td><td class="num">302 ms</td><td class="num">27×</td></tr>
  <tr><td>Peak RSS, 10 min file</td><td class="num"><strong>43 MB</strong></td><td class="num">256 MB</td><td class="num">6×</td></tr>
  <tr><td>Playback overhead, 3 s tone</td><td class="num"><strong>30 ms</strong></td><td class="num">241 ms</td><td class="num">8×</td></tr>
  <tr><td>Batch decode, 2 FLAC files</td><td class="num"><strong>177 ms</strong></td><td class="num">299 ms</td><td class="num">1.7×</td></tr>
</table>

<h2 id="decode">Decode detail</h2>
<p>60 seconds of stereo 44.1 kHz, median and p95 over seven runs.</p>
<table>
  <tr><th>Format</th><th class="num">sound_rs</th><th class="num">libsndfile</th><th>PCM check</th></tr>
  <tr><td>WAV 16-bit</td><td class="num">131 / 160 ms</td><td class="num">155 ms</td><td>identical</td></tr>
  <tr><td>FLAC</td><td class="num">265 / 267 ms</td><td class="num">465 ms</td><td>identical</td></tr>
  <tr><td>MP3</td><td class="num">241 / 249 ms</td><td class="num">268 ms</td><td>1.6·10⁻⁵, 47 samples offset</td></tr>
  <tr><td>M4A (AAC)</td><td class="num">361 / 365 ms</td><td class="num">unsupported</td><td>8·10⁻⁸ vs ffmpeg</td></tr>
</table>

<h2 id="memory">Playback and memory</h2>
<table>
  <tr><th></th><th class="num">First sound</th><th class="num">Peak RSS</th><th class="num">Overhead</th></tr>
  <tr><td>sound_rs Player, 10 min stream</td><td class="num">11.3 ms</td><td class="num">43 MB</td><td class="num">+30 ms</td></tr>
  <tr><td>soundfile + sounddevice</td><td class="num">302 ms</td><td class="num">256 MB</td><td class="num">+241 ms</td></tr>
</table>
<p>The playback overhead row is a 3-second tone: wall clock minus the track duration. The
player number uses a stream that is already open; the one-shot <code>play_array</code> pays
the device setup and lands around +180 ms.</p>

<h2 id="reproduce">Reproduce it</h2>
""" + code("""
python bench/run.py                 # sound_rs only
python bench/run.py --compare       # + soundfile / sounddevice
python bench/run.py --reps 9
""", "bash") + """
<p>The harness generates its corpus (3 s and 10 min, WAV plus MP3/FLAC/M4A when ffmpeg is
available), runs each measurement in a subprocess, and prints median, p95 and peak RSS. A full
run takes about three minutes on the machine above.</p>

<h2 id="caveats">Where it loses</h2>
<ul>
  <li><strong>MP3 alignment</strong>: 47 samples (~1 ms) earlier than libsndfile and ffmpeg.
      Length is exact; values match after alignment.</li>
  <li><strong>WAV decode</strong>: a memcpy race with libsndfile, 1.2× is all there is.</li>
  <li><strong>AAC decode</strong> is slower than the other formats — a newer decoder with less
      tuning — though libsndfile cannot decode it at all.</li>
  <li><strong>Very small files</strong>: for a 3-second tone the fixed device setup dominates
      everything, and all libraries land within a few tens of milliseconds of each other.</li>
</ul>
""",
)

page(
    "faq.html",
    "Troubleshooting",
    "Common sound_rs problems: no sound, busy device, ALSA and PipeWire on Linux, Jupyter, asyncio, and why the first play is slower.",
    """
<p class="kicker">11 · Reference</p>
<h1>Troubleshooting</h1>
<p class="lead">What to check when there is no sound, no device, or a call that blocks longer
than expected.</p>

<h2 id="no-sound">No sound</h2>
<ol>
  <li><code>sound_rs.list_devices()</code> — if the list is empty, the OS sees no output device.</li>
  <li><code>sound_rs.play_sine_wave(440.0, 1.0, 44100)</code> — if this is silent, the problem
      is the device, not the file.</li>
  <li>Check the system volume and the default output device; the player follows the OS default.</li>
  <li>On macOS, the terminal or IDE may need microphone/audio permission for output routing in
      some sandboxed setups.</li>
</ol>

<h2 id="busy">The device is busy</h2>
<p><code>SoundRsError: Cannot open output stream</code> usually means another application holds
the device in exclusive mode — a DAW, a call, or another Python process with an open player.
Close one player before opening another, or use a different device:</p>
""" + code("""
import sound_rs

print(sound_rs.list_devices())
sound_rs.play_audio("track.mp3", device="Built-in Audio Analog Stereo")
""") + """
<h2 id="linux">Linux: ALSA and PipeWire</h2>
<p>Playback goes through ALSA. On PipeWire systems install <code>pipewire-alsa</code>; on
PulseAudio systems install <code>pulseaudio-alsa</code>, so ALSA has a bridge to the sound
server.</p>
<p>In a container there is no sound server and no device: that is expected. Use
<code>load</code> for decoding — it needs no audio device.</p>

<h2 id="slow-start">Why the first play is slow</h2>
<p><code>play_audio</code> and <code>play_array</code> open the output device on every call:
roughly 0.1–0.2 s before the first sample. <code>Player</code> opens it once, which brings the
overhead down to about 30 ms.</p>
""" + code("""
with sound_rs.Player() as player:
    player.play_file("a.flac"); player.wait()
    player.play_file("b.flac"); player.wait()   # no device setup
""") + """
<h2 id="blocking">The call blocks forever</h2>
<p>It should not. Playback without progress for two seconds raises
<code>SoundRsError: Playback stalled</code> instead of hanging. If you see it, the device
probably disappeared — reopen the player and retry.</p>
<p>If you need playback in the background, use a thread or <code>Player.play_file</code>,
which already returns immediately.</p>

<h2 id="jupyter">Jupyter and notebooks</h2>
<p>Blocking calls work, but a long track freezes the cell. Use the player:</p>
""" + code("""
player = sound_rs.Player()
player.play_file("lecture.mp3")
# the cell returns; more code runs while it plays
player.position
""") + """
<h2 id="asyncio">asyncio</h2>
<p><code>play_audio</code> and <code>wait</code> release the GIL but they are still blocking
calls. Off-load them:</p>
""" + code("""
import asyncio
import sound_rs

await asyncio.to_thread(sound_rs.play_audio, "track.flac")
""") + """
<h2 id="mp3-offset">MP3 samples do not line up with another library</h2>
<p>Different decoders trim the encoder delay slightly differently. sound_rs starts 47 samples
(~1 ms) earlier than libsndfile and ffmpeg on LAME files; lengths match exactly. If you compare
waveforms, align first. See <a href="formats.html#gapless">Formats and gapless</a>.</p>

<h2 id="numpy">"Input must be a 1D or 2D numpy array of float32"</h2>
<p>NumPy's maths returns float64 by default:</p>
""" + code("""
samples = (0.2 * np.sin(2 * np.pi * 440 * np.arange(48000) / 48000))
samples = samples.astype(np.float32)          # required
sound_rs.play_array(samples, 48000)
""") + """
<h2 id="m1">Intel Macs and ARM Linux</h2>
<p>Only macOS arm64, Windows amd64 and Linux x86_64 have wheels. Elsewhere pip builds from
source, which needs Rust. <code>pip install sound_rs --no-binary sound_rs</code> forces that
path if a stale wheel is being picked up.</p>
""",
)

page(
    "changelog.html",
    "Changelog",
    "sound_rs release history: what changed in each version and where to find the full release notes.",
    """
<p class="kicker">12 · Reference</p>
<h1>Changelog</h1>
<p class="lead">Short history. Full notes with diffs live on
<a href="https://github.com/Neksand/sound_rs/releases">GitHub Releases</a>.</p>

<h2 id="v040">0.4.0</h2>
<p class="kicker">recording era — writing files</p>
<ul>
  <li><code>sound_rs.write(path, audio, rate, subtype=)</code> writes WAV in PCM16,
      PCM24 or float32. Arrays are <code>(frames, channels)</code>, matching
      <code>load</code>, so a load/write round-trip needs no transpose.</li>
  <li><code>Recorder.record_to(path, seconds, subtype=)</code> records and saves in one call.</li>
  <li>Non-contiguous arrays (<code>audio[:, 0]</code>) are accepted by <code>write</code> and
      <code>play_array</code> instead of raising.</li>
</ul>

<h2 id="v032">0.3.2</h2>
<p class="kicker">accuracy</p>
<ul>
  <li><code>load_bytes</code> trims AAC priming too: the MP4 edit list is read from the
      buffer, so bytes decode to the same frame count as a file.</li>
  <li><code>record(seconds)</code> never returns more than the requested frames; the extra
      device buffer at the end is cut.</li>
</ul>

<h2 id="v031">0.3.1</h2>
<p class="kicker">documentation</p>
<ul>
  <li>PyPI carries the corrected README: roadmap, and one figure for time-to-first-sound.</li>
</ul>

<h2 id="v030">0.3.0</h2>
<p class="kicker">reliability, datasets, recording</p>
<ul>
  <li>Stable error codes, mutation fuzzing, deterministic decode hashes, underrun and
      latency reporting.</li>
  <li><code>load(..., sample_rate, mono, offset, duration)</code>, <code>load_bytes</code>,
      <code>Recorder</code>, <code>Player.enqueue</code>, <code>begin_feed/feed</code>.</li>
  <li>Benchmark harness v2 and abi3 wheels for five platforms.</li>
</ul>

<h2 id="v022">0.2.2</h2>
<p class="kicker">command line</p>
<ul>
  <li><code>soundrs</code> console command: <code>play</code>, <code>info</code>,
      <code>devices</code>, <code>docs</code>, <code>--version</code>, <code>--help</code>.</li>
  <li>Documentation link in the package metadata.</li>
</ul>

<h2 id="v021">0.2.1</h2>
<p class="kicker">packaging and metadata</p>
<ul>
  <li><code>sound_rs.__version__</code>.</li>
  <li>Metadata for discovery: keywords, classifiers, documentation links; wheels and
      <code>requires-python</code> now agree on Python 3.10+.</li>
  <li>Small internal cleanups; no behaviour change.</li>
</ul>

<h2 id="v020">0.2.0</h2>
<p class="kicker">streaming player</p>
<ul>
  <li><code>Player</code> with play, pause, resume, stop, seek, volume, position, duration.</li>
  <li>Streaming decode: constant memory, ~11 ms to first sound on a 10-minute file.</li>
  <li>rubato resampler replaces linear interpolation; downsampling no longer aliases.</li>
  <li>MP3 decoded with minimp3 (1.1× faster than libsndfile, previously slower).</li>
  <li><code>load_many</code>, <code>list_devices</code>, device selection, typed stubs,
      <code>SoundRsError</code>.</li>
</ul>

<h2 id="v018">0.1.8</h2>
<p class="kicker">correctness</p>
<ul>
  <li>Gapless trimming for MP3 and OGG; AAC priming read from the MP4 edit list.</li>
  <li>Metadata without decoding; header-only duration.</li>
  <li>Playback waits for the exact device drain time instead of a fixed 100 ms.</li>
</ul>

<h2 id="v017">0.1.7</h2>
<p class="kicker">builds</p>
<ul>
  <li>Wheels for Linux, Windows and macOS across Python 3.10–3.14, published from CI.</li>
</ul>
""",
)


page(
    "404.html",
    "Page not found",
    "That page does not exist in the sound_rs documentation.",
    """
<p class="kicker">404</p>
<h1>That page does not exist.</h1>
<p class="lead">The link is broken or the page moved. Everything the docs cover is reachable
from the list on the left.</p>
<ul>
  <li><a href="index.html">Overview</a></li>
  <li><a href="quickstart.html">Quickstart</a></li>
  <li><a href="api.html">API reference</a></li>
</ul>
""",
    hidden=True,
)


page(
    "cli.html",
    "Command line",
    "The soundrs command: play a file, inspect metadata, list output devices, and find the documentation from a terminal.",
    """
<p class="kicker">03 · Start</p>
<h1>Command line</h1>
<p class="lead">Installing the package also installs <code>soundrs</code>. It exists so that a
minute after <code>pip install</code> you can check that audio works and find the docs, without
writing Python.</p>

""" + code("""
soundrs --help
""", "bash") + code("""
sound_rs — audio for Python

Usage
  soundrs <command> [options]
  soundrs play <file> [--device <name>] [--volume <0-4>]
  soundrs info <file>
  soundrs devices
  soundrs docs
  soundrs --version
  soundrs --help

Commands
  play     Play a file and wait until it ends.
           --device <name>   output device, see `soundrs devices`
           --volume <number> 0 to 4, default 1
  info     Sample rate, channels and duration, read from the header only.
  devices  Output devices, one per line.
  docs     Print the documentation link.

Links
  Documentation  https://neksand.github.io/sound_rs/
  Source         https://github.com/Neksand/sound_rs
""", "text") + """
<h2 id="commands">Commands</h2>
<table>
  <tr><th>Command</th><th>What it does</th></tr>
  <tr><td class="num">soundrs play FILE</td><td>Plays a file and waits. <code>--device</code> picks an output device, <code>--volume</code> takes 0…4.</td></tr>
  <tr><td class="num">soundrs info FILE</td><td>Sample rate, channel count and duration from the container header — no decoding.</td></tr>
  <tr><td class="num">soundrs devices</td><td>Output devices, the default marked with <code>*</code>.</td></tr>
  <tr><td class="num">soundrs docs</td><td>Prints the documentation URL, for scripts and <code>open $(soundrs docs)</code>.</td></tr>
  <tr><td class="num">soundrs --version</td><td>Version on one line, safe to parse.</td></tr>
</table>
<p><code>sound-rs</code> is installed as an alias of the same command.</p>

<h2 id="examples">Examples</h2>
""" + code("""
# does audio work at all?
soundrs play ~/Music/test.flac

# which device names can I pass to Player(device=...)?
soundrs devices

# what is this file before writing any code?
soundrs info podcast.m4a
""", "bash") + code("""
$ soundrs info podcast.m4a
file         podcast.m4a
sample rate  44100 Hz
channels     2
duration     42:13.6 (2533.612 s)
""", "text") + note(
        "Exit codes",
        "<code>0</code> success, <code>1</code> runtime failure (unreadable file, no device), "
        "<code>2</code> bad usage. Errors go to stderr, so the commands compose in shell "
        "pipelines.",
    ) + """
<h2 id="install-check">Checking an install</h2>
<p>After <code>pip install sound_rs</code>:</p>
""" + code("""
soundrs --version      # soundrs 0.2.2
soundrs devices        # a device list means the audio backend loaded
soundrs play test.wav  # actual output
""", "bash") + """
<p>If any of these fail with an import or library error, the audio backend did not load — see
<a href="faq.html">Troubleshooting</a>. Decoding and <code>soundrs info</code> do not need an
output device at all.</p>
""",
)


page(
    "datasets.html",
    "Datasets and ML",
    "Load audio for machine learning: one call to resample to 16 kHz mono, parallel decoding across cores, and honest memory numbers.",
    """
<p class="kicker">08 · Guides</p>
<h1>Datasets and ML</h1>
<p class="lead">Most dataset code wants the same thing: 16 kHz, mono, one array per
file, as fast as the machine allows. That is one call.</p>

<h2 id="one-call">One call</h2>
""" + code("""
import glob
import sound_rs

paths = sorted(glob.glob("dataset/**/*.flac", recursive=True))
clips = sound_rs.load_many(paths, sample_rate=16000, mono=True)

for audio, rate in clips[:3]:
    print(audio.shape, rate)        # (frames, 1), 16000
""") + """
<p><code>load_many</code> decodes in parallel with the GIL released. On two FLAC files the
harness measures 168 ms against 301 ms for a <code>soundfile</code> loop; the gap grows with
file count until the cores are saturated.</p>

<h2 id="single">One file, converted</h2>
""" + code("""
import sound_rs

audio, rate = sound_rs.load(
    "clip.m4a",
    sample_rate=16000,   # resampled with an anti-aliasing filter
    mono=True,           # channels averaged
)
assert audio.dtype == "float32" and audio.ndim == 2
""") + """
<p>The order is fixed: fold channels first, then resample. Resampling before folding would do
the same work twice on stereo.</p>

<h2 id="window">A window, not the whole file</h2>
""" + code("""
# first 10 seconds at 16 kHz, without decoding the rest
audio, rate = sound_rs.load("lecture.m4a", duration=10.0, sample_rate=16000)

# 30 s starting at 5 minutes
audio, rate = sound_rs.load("lecture.m4a", offset=300.0, duration=30.0)
""") + note(
        "Windows come from the streaming reader",
        "<code>offset</code> uses the same frame-accurate seek as the player, so the window "
        "starts on the requested sample, and the rest of the file is never decoded.",
    ) + """
<h2 id="memory">Memory</h2>
<table>
  <tr><th>Task</th><th>Peak RAM</th></tr>
  <tr><td>Decode 10 min stereo to float32</td><td class="num">211 MB of audio</td></tr>
  <tr><td>Decode the same file as 16 kHz mono</td><td class="num">38 MB</td></tr>
  <tr><td>Stream the same file for playback</td><td class="num">43 MB, independent of length</td></tr>
</table>
<p>Resampling to 16 kHz mono is not just faster downstream — it is 5× less memory than
keeping 44.1 kHz stereo.</p>

<h2 id="threads">Choosing the thread count</h2>
""" + code("""
clips = sound_rs.load_many(paths, threads=4)   # explicit
clips = sound_rs.load_many(paths)              # one worker per core
""") + """
<p>Decoding is CPU-bound, so the ceiling is the core count. Feeding more threads than cores
does not help; the benchmark's scaling scenario measures 1, 2, 4 and 8 workers.</p>

<h2 id="not">When not to use this</h2>
<ul>
  <li><strong>Training on the fly</strong> — decode once and store the arrays; repeated
      decoding of the same file is wasted CPU.</li>
  <li><strong>Huge corpora</strong> — <code>load_many</code> returns everything at once. For a
      corpus larger than RAM, call <code>load</code> per file inside your own batching.</li>
  <li><strong>Augmentation and feature extraction</strong> — not implemented. This library
      gets you clean float32 audio; the DSP belongs to your framework.</li>
</ul>
""",
)

page(
    "recording.html",
    "Recording",
    "Record from a microphone with sound_rs: input devices, start/stop, fixed-length captures, macOS permissions, and writing a WAV.",
    """
<p class="kicker">09 · Guides</p>
<h1>Recording</h1>
<p class="lead">The mirror of playback: one input stream, float32 samples, and the same error
contract.</p>

""" + code("""
import sound_rs

print(sound_rs.list_input_devices())     # [(0, 'MacBook Pro Microphone', True)]

with sound_rs.Recorder() as recorder:
    audio, rate = recorder.record(5.0)   # five seconds
print(audio.shape, rate)                 # (240000, 1) 48000
""") + """
<h2 id="devices">Devices</h2>
<p><code>list_input_devices()</code> returns <code>(index, name, is_default)</code>. Pass the
name to the constructor:</p>
""" + code("""
recorder = sound_rs.Recorder(device="MacBook Pro Microphone")
print(recorder.device, recorder.sample_rate, recorder.channels)
""") + """
<p>The rate and channel count are whatever the device reports; the library does not resample
on input. If you need 16 kHz mono, ask for it after the fact:</p>
""" + code("""
import sound_rs

mono = audio.mean(axis=1, keepdims=True)     # or use numpy's resampler of choice
""") + """
<h2 id="start-stop">start and stop</h2>
<p><code>record(seconds)</code> is the convenient form. For an open-ended take, use
<code>start()</code> and <code>stop()</code>:</p>
""" + code("""
recorder = sound_rs.Recorder()
recorder.start()
input("press enter to stop")
audio, rate = recorder.stop()
""") + """
<h2 id="wav">Writing a WAV</h2>
<p>The arrays are plain float32, so the standard library writes them:</p>
""" + code("""
import wave
import numpy as np
import sound_rs

audio, rate = sound_rs.Recorder().record(3.0)
pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")

with wave.open("take.wav", "w") as handle:
    handle.setnchannels(audio.shape[1])
    handle.setsampwidth(2)
    handle.setframerate(rate)
    handle.writeframes(pcm.tobytes())
""") + warn(
        "Microphone permission",
        "macOS and Windows ask for microphone access the first time. A denied permission "
        "raises <code>SoundRsError</code> with <code>code == 'device'</code>; there is no "
        "silent fallback.",
    ) + """
<h2 id="save">Saving a take</h2>
""" + code("""
import sound_rs

with sound_rs.Recorder() as recorder:
    recorder.record_to("take.wav", 5.0, subtype="pcm24")
""") + """
<p>For anything else — resampling, trimming, writing a different name — record to an array
and use <code>sound_rs.write</code>, which writes the same
<code>(frames, channels)</code> layout that <code>load</code> returns.</p>
""" + code("""
import sound_rs

audio, rate = sound_rs.Recorder().record(5.0)
sound_rs.write("take.wav", audio, rate, subtype="float32")
""") + """
<h2 id="limits">Limits</h2>
<ul>
  <li>The capture starts when the device delivers its first buffer. On Bluetooth headsets that
      is 100–150 ms, so a short take can be that much shorter than requested — the recorder
      never adds frames to hide it.</li>
  <li>Captured samples are held in memory: the buffer caps at 230 M samples (~40 minutes of
      48 kHz stereo). Past that the recorder raises instead of exhausting RAM.</li>
  <li>No resampling, mixing or effects on input — capture is raw device audio.</li>
  <li>Duplex (playing while recording) works because they are separate streams, but they are
      independent objects with independent device handles.</li>
</ul>
""",
)

page(
    "guarantees.html",
    "Guarantees",
    "What sound_rs promises: error codes, no panics on bad input, released GIL, deterministic decode, bounded memory. And what it does not promise.",
    """
<p class="kicker">11 · Reference</p>
<h1>Guarantees</h1>
<p class="lead">A library is trustworthy when its promises are written down and tested. This is
the list, and the ones that are deliberately not on it.</p>

<h2 id="errors">Errors</h2>
<p>Every failure is <code>sound_rs.SoundRsError</code> (a <code>RuntimeError</code>) with a
stable <code>.code</code>:</p>
<table>
  <tr><th>Code</th><th>Meaning</th></tr>
  <tr><td class="num">io</td><td>The file could not be opened or read.</td></tr>
  <tr><td class="num">format</td><td>The container or codec was not understood, or no audio was decodable.</td></tr>
  <tr><td class="num">device</td><td>No output/input device, or the device refused to open.</td></tr>
  <tr><td class="num">stalled</td><td>Playback made no progress for two seconds; the device probably disappeared.</td></tr>
  <tr><td class="num">unsupported</td><td>The build lacks a feature (NumPy) or hit a documented limit.</td></tr>
  <tr><td class="num">usage</td><td>The call order was wrong, for example <code>feed()</code> before <code>begin_feed()</code>.</td></tr>
</table>
<p>Codes are part of the API: renaming one is a breaking change.</p>

<h2 id="panics">No panics on bad input</h2>
<p>Corrupt files return an error, they do not abort the interpreter. This is enforced in
<code>cargo test</code> by mutating every committed fixture ~200 times per file (bit flips,
truncation, appended noise) and by feeding random bytes; every case must return <code>Ok</code>
with finite samples or a non-empty error.</p>

<h2 id="threads">Threads and the GIL</h2>
<ul>
  <li>Decoding, playback, recording and <code>wait()</code> release the GIL.</li>
  <li>One <code>Player</code> is one stream: call its methods from one thread at a time.</li>
  <li>Different players can be used from different threads, each with its own device handle.</li>
</ul>

<h2 id="determinism">Determinism</h2>
<p>Decoding is deterministic: the same file produces the same float32 samples on every run.
Across CPUs the integer decoder's SIMD paths can differ by a bit, so the regression hashes are
taken at 10⁻⁴ (about three LSB at 16-bit scale). <code>tests/decode_regression.py</code> pins a SHA-256 of the PCM for five
fixtures; a decoder change that moves a single sample fails CI.</p>

<h2 id="memory">Memory</h2>
<ul>
  <li>Streaming playback uses a one-second ring buffer: memory does not grow with file length
      (43 MB peak on a 10-minute file, measured).</li>
  <li><code>load</code> allocates the decoded audio once; <code>load_many</code> allocates all
      results at once, so size your batches.</li>
  <li>The recorder caps its buffer at 230 M samples instead of growing without bound.</li>
  <li>A 100-cycle playback soak grows RSS by less than 20 MB (measured: 0.0 MB).</li>
</ul>

<h2 id="quality">Audio quality</h2>
<ul>
  <li>Resampling uses rubato's FFT resampler: a 20 kHz tone downsampled to 24 kHz is filtered
      below 10% of its level instead of folding back as an alias.</li>
  <li>Gapless: encoder delay and padding are trimmed for MP3, OGG and AAC/MP4.</li>
  <li>WAV and FLAC decode to PCM identical to libsndfile; MP3 matches to 1.6·10⁻⁵ with a
      47-sample offset; AAC matches ffmpeg to 8·10⁻⁸.</li>
</ul>

<h2 id="not">What is not promised</h2>
<ul>
  <li><strong>A latency figure.</strong> Latency depends on the OS, the device and its buffer;
      <code>Player.latency</code> reports what the backend measured, nothing more.</li>
  <li><strong>Zero underruns everywhere.</strong> <code>Player.underruns</code> counts them;
      on a loaded machine or a Bluetooth sink they will appear.</li>
  <li><strong>Real-time safety of the Python API.</strong> Calling <code>feed()</code> from
      Python is fine, but Python's scheduler is not a real-time scheduler.</li>
  <li><strong>Bit-identical output across decoders.</strong> Different MP3 decoders trim delay
      differently; we match length exactly and samples to 1.6·10⁻⁵.</li>
</ul>
""",
)
