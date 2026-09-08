#!/usr/bin/env python3
"""
HERMES voice shifter
====================
Real-time voice disguise (pitch + timbre) via a phase-vocoder pitch shifter.
Feeds the OBS "Desktop/Mic Audio" chain (or any headset output) live.

Usage:
  python voice_shifter.py                    # real-time, default disguise
  python voice_shifter.py --ratio 0.8        # lower pitch more (0.7-0.9 = deeper)
  python voice_shifter.py --ratio 1.3        # higher pitch (thin / chipmunk)
  python voice_shifter.py --tilt 2.0         # darken timbre (raise = darker)
  python voice_shifter.py --list             # list audio devices
  python voice_shifter.py --file in.wav --out out.wav   # offline, testable

The DSP itself (pitch_shift_pv) is pure numpy and needs no audio hardware —
run the --test flag on any machine to verify it shifts pitch correctly.

Zero-latency alternative: OBS has a native "Pitch Shifter" audio filter on the
mic source. Use THIS script when you want deeper disguise than OBS provides,
or when you're routing audio outside OBS entirely.
"""
import argparse
import wave

import numpy as np


# ---------------------------------------------------------------------------
# Phase-vocoder pitch shifter (single pass, preserves duration).
# ratio > 1  -> higher pitch, ratio < 1 -> lower pitch. 1.0 = unchanged.
# ---------------------------------------------------------------------------
def time_stretch_pv(x, stretch, n_fft=1024, hop=256):
    """Phase-vocoder time-stretch, PRESERVES pitch. stretch>1 = slower/longer."""
    h_a = hop
    h_s = max(2, int(round(hop * stretch)))
    win = np.hanning(n_fft)
    n = len(x)
    out = np.zeros(int(n * stretch) + n_fft)
    freqs = 2 * np.pi * np.arange(n_fft // 2 + 1) / n_fft

    prev_an_phase = None
    synth_phase = np.zeros(n_fft // 2 + 1)

    pos_a = 0
    pos_s = 0
    while pos_a + n_fft <= n:
        frame = x[pos_a:pos_a + n_fft] * win
        spec = np.fft.rfft(frame)
        an_mag, an_phase = np.abs(spec), np.angle(spec)
        if prev_an_phase is None:
            true_freq = freqs.copy()
            synth_phase = an_phase.copy()
        else:
            delta = an_phase - prev_an_phase - freqs * h_a
            delta = (delta + np.pi) % (2 * np.pi) - np.pi
            true_freq = freqs + delta / h_a
            synth_phase = synth_phase + true_freq * h_s
            synth_phase = (synth_phase + np.pi) % (2 * np.pi) - np.pi
        prev_an_phase = an_phase

        out_frame = np.fft.irfft(an_mag * np.exp(1j * synth_phase), n_fft) * win
        out[pos_s:pos_s + n_fft] += out_frame
        pos_a += h_a
        pos_s += h_s

    norm = np.zeros_like(out)
    pos_s = 0
    pos_a = 0
    while pos_a + n_fft <= n:
        norm[pos_s:pos_s + n_fft] += win * win
        pos_a += h_a
        pos_s += h_s
    norm = np.where(norm > 1e-6, norm, 1.0)
    return (out / norm).astype(np.float32)


def _resample(x, factor):
    """Linear resample by an arbitrary factor (>1 upsample, <1 downsample)."""
    if abs(factor - 1.0) < 1e-6:
        return x
    n_out = int(round(len(x) * factor))
    idx = np.arange(n_out) / factor
    idx = np.clip(idx, 0, len(x) - 1)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


def pitch_shift_pv(x, ratio, n_fft=1024, hop=256):
    """Pitch shift preserving duration: stretch by 1/ratio, then resample by
    ratio (standard two-stage phase-vocoder method). ratio>1 = higher."""
    if abs(ratio - 1.0) < 1e-6:
        return x.copy()
    # stretch by `ratio` (pitch unchanged), then resample by 1/ratio, which
    # changes pitch by (1/(1/ratio))=ratio and restores original length.
    stretched = time_stretch_pv(x, ratio, n_fft, hop)
    shifted = _resample(stretched, 1.0 / ratio)
    if len(shifted) > len(x):
        return shifted[:len(x)]
    if len(shifted) < len(x):
        return np.pad(shifted, (0, len(x) - len(shifted))).astype(np.float32)
    return shifted


def spectral_tilt(x, gain_db, sr, n_fft=1024, hop=256):
    """Gentle shelf to darken/brighten timbre (formant-ish disguise)."""
    if abs(gain_db) < 1e-6:
        return x
    win = np.hanning(n_fft)
    n = len(x)
    out = np.zeros(n + n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # smooth gain from 0 dB at low end to `gain_db` at nyquist
    g = np.linspace(0.0, gain_db, len(freqs))
    gain = 10 ** (g / 20.0)
    for start in range(0, n - n_fft, hop):
        spec = np.fft.rfft(x[start:start + n_fft] * win)
        out[start:start + n_fft] += np.fft.irfft(spec * gain, n_fft) * win
    norm = np.zeros_like(out)
    for start in range(0, n - n_fft, hop):
        norm[start:start + n_fft] += win * win
    norm = np.where(norm > 1e-6, norm, 1.0)
    return (out[:n] / norm[:n]).astype(np.float32)


# ---------------------------------------------------------------------------
# Offline / testable WAV processing (no audio hardware needed).
# ---------------------------------------------------------------------------
def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        x = (x[0::2] + x[1::2]) / 2.0  # downmix to mono
    return x, sr


def write_wav(path, x, sr):
    data = np.clip(x, -1.0, 1.0)
    pcm = (data * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def process_file(in_path, out_path, ratio, tilt, sr=0):
    x, sr_in = read_wav(in_path)
    sr = sr or sr_in
    y = pitch_shift_pv(x, ratio)
    y = spectral_tilt(y, tilt, sr)
    write_wav(out_path, y, sr_in)
    return x, y, sr_in


# ---------------------------------------------------------------------------
# Real-time streaming (needs a working audio device).
# ---------------------------------------------------------------------------
def stream(ratio, tilt, device=None, blocksize=1024, sr=44100):
    import sounddevice as sd
    if device is not None:
        sd.default.device = device

    prev = np.zeros(blocksize, dtype=np.float32)  # reuse norm buffer pattern

    def callback(indata, outdata, frames, time_info, status):
        # sounddevice gives indata.shape = (frames, ch) float32
        mono = indata[:, 0] if indata.ndim == 2 else indata
        y = pitch_shift_pv(mono.copy(), ratio)
        y = spectral_tilt(y, tilt, sr)
        if outdata.ndim == 2:
            outdata[:, 0] = y
            for c in range(1, outdata.shape[1]):
                outdata[:, c] = y
        else:
            outdata[:] = y

    print(f"[hermes] voice shifter live  ratio={ratio}  tilt={tilt:+}dB "
          f"sr={sr}  blocksize={blocksize}")
    print("[hermes] press Ctrl+C to stop")
    with sd.Stream(device=device, samplerate=sr, blocksize=blocksize,
                   channels=1, callback=callback):
        while True:
            sd.sleep(1000)


# ---------------------------------------------------------------------------
def self_test():
    """Verify the DSP actually shifts pitch. Pure numpy, no audio needed."""
    sr = 16000
    n = sr // 2
    t = np.arange(n) / sr
    f0 = 200.0
    x = (0.9 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)

    for ratio, expect in [(0.8, f0 * 0.8), (1.3, f0 * 1.3)]:
        y = pitch_shift_pv(x, ratio)
        spec = np.abs(np.fft.rfft(y * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        peak = freqs[np.argmax(spec)]
        err = abs(peak - expect) / expect * 100
        status = "OK " if err < 3.0 else "FAIL"
        print(f"ratio={ratio:<4} input_f0={f0:.1f}Hz  "
              f"expected={expect:.1f}Hz  got={peak:.1f}Hz  "
              f"err={err:.2f}%  [{status}]")
        assert err < 3.0, f"pitch shift failed for ratio {ratio}"
    print("DSP SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description="Hermes voice shifter")
    ap.add_argument("--ratio", type=float, default=0.85,
                    help="pitch ratio (0.5-0.95 deeper, 1.05+ higher). 1.0 = off")
    ap.add_argument("--tilt", type=float, default=1.5,
                    help="timbre shelf in dB (positive = darker disguise)")
    ap.add_argument("--list", action="store_true", help="list audio devices")
    ap.add_argument("--file", help="offline mode: input wav")
    ap.add_argument("--out", default="shifted.wav", help="offline output wav")
    ap.add_argument("--device", type=int, help="audio device index")
    ap.add_argument("--test", action="store_true",
                    help="run DSP self-test and exit")
    args = ap.parse_args()

    if args.test:
        self_test()
        return
    if args.list:
        import sounddevice as sd
        print(sd.query_devices())
        return
    if args.file:
        x, y, sr = process_file(args.file, args.out, args.ratio, args.tilt)
        print(f"[hermes] {args.file} -> {args.out}  "
              f"{len(x)/sr:.2f}s, ratio={args.ratio}, tilt={args.tilt:+}dB")
        return

    stream(args.ratio, args.tilt, device=args.device)


if __name__ == "__main__":
    main()
