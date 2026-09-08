#!/usr/bin/env python3
"""
HERMES voice shifter — pitch + formant disguise
================================================
Real-time voice disguise built from three DSP stages, all pure numpy:
  1. Pitch shift   — phase-vocoder (changes fundamental frequency)
  2. Formant shift — cepstral-liftered spectral-envelope warp (changes the
                     vocal-tract "character" so it sounds like a different
                     person, not just deeper)
  3. Spectral tilt — gentle brightness shelf for extra disguise depth

Usage:
  python voice_shifter.py                       # live, default disguise
  python voice_shifter.py --preset deep         # named disguise presets
  python voice_shifter.py --ratio 0.8 --formant 0.9   # manual control
  python voice_shifter.py --list                # list audio devices
  python voice_shifter.py --file in.wav --out out.wav   # offline, testable
  python voice_shifter.py --test                # verify DSP (pitch+formant)

All DSP is deterministic and hardware-free; --test validates it on any box.
"""
import argparse
import wave

import numpy as np


# ---------------------------------------------------------------------------
# Stage 1: phase-vocoder pitch shift (verified: exact pitch change)
# ---------------------------------------------------------------------------
def time_stretch_pv(x, stretch, n_fft=1024, hop=256):
    """Phase-vocoder time-stretch, PRESERVES pitch. stretch>1 = longer."""
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
    """Pitch shift preserving duration: stretch by ratio, then resample by 1/ratio."""
    if abs(ratio - 1.0) < 1e-6:
        return x.copy()
    stretched = time_stretch_pv(x, ratio, n_fft, hop)
    shifted = _resample(stretched, 1.0 / ratio)
    if len(shifted) > len(x):
        return shifted[:len(x)]
    if len(shifted) < len(x):
        return np.pad(shifted, (0, len(x) - len(shifted))).astype(np.float32)
    return shifted


# ---------------------------------------------------------------------------
# Stage 2: formant (vocal-tract) shift via cepstral envelope warping.
#   k > 1  -> raise formants (brighter, "younger"/thinner vocal tract)
#   k < 1  -> lower formants (darker, "larger"/deeper vocal tract)
# This is what makes a disguise sound like a different PERSON, not just a
# deeper version of the same voice.
# ---------------------------------------------------------------------------
def formant_shift(x, k, sr, n_fft=1024, hop=256, lifter=40):
    if abs(k - 1.0) < 1e-6:
        return x
    win = np.hanning(n_fft)
    n = len(x)
    out = np.zeros(n + n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    n_f = len(freqs)
    nyq = freqs[-1]
    # env_out(f) = env(f/k): sample the OLD envelope at positions freqs/k
    old_pos = np.clip(freqs / k, 0.0, nyq)
    old_idx = old_pos / nyq * (n_f - 1)

    for start in range(0, n - n_fft, hop):
        frame = x[start:start + n_fft] * win
        spec = np.fft.rfft(frame)
        mag = np.abs(spec) + 1e-10
        logmag = np.log(mag)
        # real cepstrum
        cep = np.fft.irfft(logmag, n_fft)
        # envelope = low-quefrency part; source = the rest
        cep_env = cep.copy()
        cep_env[lifter + 1:] = 0.0
        env = np.fft.rfft(cep_env).real
        res = logmag - env
        # warp the envelope along frequency -> moves formant peaks by k
        env_warp = np.interp(old_idx, np.arange(n_f), env)
        new_logmag = env_warp + res
        new_mag = np.exp(new_logmag)
        out_frame = np.fft.irfft(new_mag * np.exp(1j * np.angle(spec)), n_fft) * win
        out[start:start + n_fft] += out_frame

    norm = np.zeros_like(out)
    for start in range(0, n - n_fft, hop):
        norm[start:start + n_fft] += win * win
    norm = np.where(norm > 1e-6, norm, 1.0)
    return (out[:n] / norm[:n]).astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 3: spectral tilt (brightness shelf)
# ---------------------------------------------------------------------------
def spectral_tilt(x, gain_db, sr, n_fft=1024, hop=256):
    if abs(gain_db) < 1e-6:
        return x
    win = np.hanning(n_fft)
    n = len(x)
    out = np.zeros(n + n_fft)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
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
# Full disguise chain + presets
# ---------------------------------------------------------------------------
PRESETS = {
    "subtle":   {"ratio": 0.92, "formant": 0.97, "tilt": 0.5,   "desc": "barely changed"},
    "deep":     {"ratio": 0.80, "formant": 0.88, "tilt": 2.0,   "desc": "clearly deeper"},
    "demon":    {"ratio": 0.65, "formant": 0.80, "tilt": 4.0,   "desc": "dark & huge"},
    "alien":    {"ratio": 0.75, "formant": 1.30, "tilt": 1.0,   "desc": "deep pitch, thin formants"},
    "chipmunk": {"ratio": 1.30, "formant": 1.25, "tilt": -3.0,  "desc": "high & small"},
    "radio":    {"ratio": 1.00, "formant": 1.10, "tilt": 3.0,   "desc": "tinny broadcast"},
}


def disguise(x, sr, ratio=0.85, formant=0.90, tilt=1.5):
    y = pitch_shift_pv(x, ratio)
    y = formant_shift(y, formant, sr)
    y = spectral_tilt(y, tilt, sr)
    return y


# ---------------------------------------------------------------------------
# Offline / testable WAV processing
# ---------------------------------------------------------------------------
def read_wav(path):
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch == 2:
        x = (x[0::2] + x[1::2]) / 2.0
    return x, sr


def write_wav(path, x, sr):
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def process_file(in_path, out_path, ratio, formant, tilt, sr=0):
    x, sr_in = read_wav(in_path)
    sr = sr or sr_in
    y = disguise(x, sr, ratio, formant, tilt)
    write_wav(out_path, y, sr_in)
    return x, y, sr_in


# ---------------------------------------------------------------------------
# Real-time streaming
# ---------------------------------------------------------------------------
def stream(ratio, formant, tilt, device=None, out_device=None, blocksize=1024, sr=44100):
    import sounddevice as sd
    # device = input (mic); out_device = where processed audio goes
    dev = (device, out_device) if out_device is not None else device
    if dev is not None:
        sd.default.device = dev

    def callback(indata, outdata, frames, time_info, status):
        mono = indata[:, 0] if indata.ndim == 2 else indata
        y = disguise(mono.copy(), sr, ratio, formant, tilt)
        if outdata.ndim == 2:
            outdata[:, 0] = y
            for c in range(1, outdata.shape[1]):
                outdata[:, c] = y
        else:
            outdata[:] = y

    print(f"[hermes] voice shifter live  ratio={ratio}  formant={formant}  "
          f"tilt={tilt:+}dB  sr={sr}  blocksize={blocksize}")
    print(f"[hermes] input={device} output={out_device}")
    print("[hermes] press Ctrl+C to stop")
    with sd.Stream(device=dev, samplerate=sr, blocksize=blocksize,
                   channels=1, callback=callback):
        while True:
            sd.sleep(1000)


# ---------------------------------------------------------------------------
def _measure_formant(x, sr, n_fft=1024):
    """Find the strongest spectral-envelope peak (a formant) in Hz."""
    win = np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(x[:len(x) // n_fft * n_fft].reshape(-1, n_fft) * win, axis=1)).mean(axis=0)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    # smooth to find envelope peak
    cep = np.fft.irfft(np.log(spec + 1e-10), n_fft)
    cep[30:] = 0
    env = np.exp(np.fft.rfft(cep).real)
    # search above 150 Hz to skip fundamental clutter
    lo = int(np.searchsorted(freqs, 150))
    return freqs[lo + np.argmax(env[lo:])]


def _measure_pitch(x, sr):
    ac = np.correlate(x, x, "full")[len(x) - 1:]
    lo = int(sr / 500)
    hi = int(sr / 80)
    lag = np.argmax(ac[lo:hi]) + lo
    return sr / lag


def self_test():
    """Verify BOTH pitch and formant shifting. Pure numpy, no audio."""
    sr = 16000
    n = sr  # 1 second
    t = np.arange(n) / sr
    f0 = 200.0
    x = (0.9 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)

    # --- pitch ---
    print("== pitch shift ==")
    for ratio, expect in [(0.8, f0 * 0.8), (1.3, f0 * 1.3)]:
        y = pitch_shift_pv(x, ratio)
        got = _measure_pitch(y, sr)
        err = abs(got - expect) / expect * 100
        print(f"  ratio={ratio:<4} expected={expect:.0f}Hz  got={got:.0f}Hz  "
              f"err={err:.1f}%  [{'OK ' if err < 3 else 'FAIL'}]")
        assert err < 3.0

    # --- formant: broadband (noise) source through a resonator at F1 ---
    print("== formant shift ==")
    F1 = 1000.0
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n).astype(np.float32) * 0.01
    r = 0.94
    y_src = np.zeros(n)
    for i in range(2, n):
        y_src[i] = noise[i] + 2 * r * np.cos(2 * np.pi * F1 / sr) * y_src[i - 1] - r * r * y_src[i - 2]
    y_src = y_src / (np.max(np.abs(y_src)) + 1e-9)

    f1_measured = _measure_formant(y_src, sr)
    print(f"  source F1 measured ~{f1_measured:.0f}Hz (target 1000)")
    for k, tol in [(0.8, 12.0), (1.2, 15.0)]:
        y = formant_shift(y_src, k, sr)
        got = _measure_formant(y, sr)
        expect = f1_measured * k
        err = abs(got - expect) / expect * 100
        print(f"  formant k={k:<4} expected~{expect:.0f}Hz  got~{got:.0f}Hz  "
              f"err={err:.1f}%  [{'OK ' if err < tol else 'FAIL'}]")
        assert err < tol

    print("DSP SELF-TEST PASSED")


def main():
    ap = argparse.ArgumentParser(description="Hermes voice shifter")
    ap.add_argument("--preset", choices=list(PRESETS) + ["none"],
                    default="none", help="named disguise preset")
    ap.add_argument("--ratio", type=float, help="pitch ratio (<1 deeper, >1 higher)")
    ap.add_argument("--formant", type=float, help="formant ratio (<1 darker tract, >1 thinner)")
    ap.add_argument("--tilt", type=float, help="brightness shelf in dB")
    ap.add_argument("--list", action="store_true", help="list audio devices")
    ap.add_argument("--file", help="offline mode: input wav")
    ap.add_argument("--out", default="shifted.wav", help="offline output wav")
    ap.add_argument("--device", type=int, help="audio input device index (mic)")
    ap.add_argument("--out-device", type=int,
                    help="audio OUTPUT device index (where shifted audio goes, "
                         "e.g. a BlackHole virtual cable OBS will capture)")
    ap.add_argument("--test", action="store_true", help="run DSP self-test")
    args = ap.parse_args()

    if args.test:
        self_test()
        return
    if args.list:
        import sounddevice as sd
        print(sd.query_devices())
        return

    # resolve params: preset provides defaults, manual flags override
    p = PRESETS.get(args.preset, PRESETS["deep"]) if args.preset != "none" else \
        {"ratio": 0.85, "formant": 0.90, "tilt": 1.5}
    ratio = args.ratio if args.ratio is not None else p["ratio"]
    formant = args.formant if args.formant is not None else p["formant"]
    tilt = args.tilt if args.tilt is not None else p["tilt"]
    if args.preset != "none":
        print(f"[hermes] preset '{args.preset}': {PRESETS[args.preset]['desc']}")

    if args.file:
        x, y, sr = process_file(args.file, args.out, ratio, formant, tilt)
        print(f"[hermes] {args.file} -> {args.out}  "
              f"{len(x)/sr:.2f}s, ratio={ratio}, formant={formant}, tilt={tilt:+}dB")
        return

    stream(ratio, formant, tilt, device=args.device,
           out_device=args.out_device)


if __name__ == "__main__":
    main()
