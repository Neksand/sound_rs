import time
import psutil
import numpy as np
import sounddevice as sd
import sound_rs
from threading import Thread

DURATION = 3.0
SAMPLERATE = 44100
RUNS = 5

t = np.linspace(0, DURATION, int(SAMPLERATE * DURATION), endpoint=False, dtype=np.float32)
wave = 0.1 * np.sin(2 * np.pi * 440 * t, dtype=np.float32)


def cpu_monitor(stop_flag, result_list):
    p = psutil.Process()
    usage = []
    while not stop_flag[0]:
        usage.append(p.cpu_percent(interval=0.05))
    result_list.append(sum(usage) / len(usage) if usage else 0.0)


def benchmark_play(play_func, name):
    best_time = float('inf')
    best_cpu = 0.0

    for i in range(RUNS):
        cpu_data = []
        stop_flag = [False]
        monitor = Thread(target=cpu_monitor, args=(stop_flag, cpu_data))
        monitor.start()

        start = time.perf_counter()
        play_func()
        elapsed = time.perf_counter() - start

        stop_flag[0] = True
        monitor.join()
        cpu_usage = cpu_data[0] if cpu_data else 0.0

        if elapsed < best_time:
            best_time = elapsed
            best_cpu = cpu_usage

        print(f"[{name}] Run {i+1}: {elapsed:.4f}s, CPU={cpu_usage:.2f}%")

    print(f"✅ {name}: Min time = {best_time:.4f}s, Avg CPU = {best_cpu:.2f}%\n")
    return best_time, best_cpu


benchmark_play(lambda: sd.play(wave, SAMPLERATE) or sd.wait(), "sounddevice")

benchmark_play(lambda: sound_rs.play_array(wave, SAMPLERATE), "sound_rs")
