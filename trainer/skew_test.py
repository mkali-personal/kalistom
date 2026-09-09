#!/usr/bin/env python3
"""
Are desktop-recorded and phone-recorded audio interchangeable as training data?

Both carry the same live broadcast, but by different routes: the desktop decodes the 202 kbps AAC
HLS stream with ffmpeg, while the phone receives whatever rendition its app requests and passes it
through Android's audio pipeline. If the two differ meaningfully below 8 kHz - all YAMNet ever sees
after the 16 kHz resample - then a model trained on desktop audio would meet different numbers in
production, and would quietly underperform.

The two recordings are not started at the same instant and HLS buffering adds tens of seconds of
delay, so the first job is to find the offset. That is done on the energy envelope, which tolerates
a wide search, then confirmed on the waveform itself.

Usage:
    python trainer/skew_test.py PHONE.wav DESKTOP.wav
"""
from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

SR = 16000
ENV_HOP = 160          # 10 ms envelope frames for the coarse search


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SR:
            raise SystemExit(f"{path.name}: expected {SR} Hz, got {w.getframerate()}")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            x = x.reshape(-1, 2).mean(1)
    return x


def envelope(x: np.ndarray) -> np.ndarray:
    n = len(x) // ENV_HOP
    e = np.sqrt((x[:n * ENV_HOP].reshape(n, ENV_HOP) ** 2).mean(1) + 1e-12)
    return np.log(e)


def best_lag(a: np.ndarray, b: np.ndarray,
             max_lag: int | None = None,
             min_overlap: int | None = None) -> tuple[int, float]:
    """
    Lag m maximising the correlation of a[k+m] against b[k], with the score normalised
    over the overlapping region only.

    Normalising by the whole-signal energy (the obvious shortcut) biases the peak towards zero
    lag, because fewer samples overlap at large offsets and the raw sum shrinks with them. That
    silently returns a fraction of the true offset - which is the regime that matters here, since
    HLS buffering can put tens of seconds between two recordings of the same broadcast.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    la, lb = len(a), len(b)
    if max_lag is None:
        max_lag = max(la, lb)
    if min_overlap is None:
        min_overlap = max(16, min(la, lb) // 8)

    # Cross-products of the RAW signals; the means are removed per-lag below, over the overlap.
    n = 1 << int(np.ceil(np.log2(la + lb)))
    raw = np.fft.irfft(np.fft.rfft(a, n) * np.conj(np.fft.rfft(b, n)), n)
    sab = np.concatenate([raw[-(lb - 1):], raw[:la]])
    lags = np.arange(-(lb - 1), la)

    ca = np.concatenate([[0.0], np.cumsum(a)])
    ca2 = np.concatenate([[0.0], np.cumsum(a ** 2)])
    cb = np.concatenate([[0.0], np.cumsum(b)])
    cb2 = np.concatenate([[0.0], np.cumsum(b ** 2)])

    k0 = np.maximum(0, -lags)                  # first index of b in the overlap
    k1 = np.minimum(lb, la - lags)             # one past the last
    cnt = (k1 - k0).astype(np.float64)

    valid = (k1 - k0 >= min_overlap) & (np.abs(lags) <= max_lag)
    if not valid.any():
        return 0, 0.0

    ia0 = np.clip(k0 + lags, 0, la)            # matching index range in a
    ia1 = np.clip(k1 + lags, 0, la)
    sa = ca[ia1] - ca[ia0]
    saa = ca2[ia1] - ca2[ia0]
    sb = cb[np.clip(k1, 0, lb)] - cb[np.clip(k0, 0, lb)]
    sbb = cb2[np.clip(k1, 0, lb)] - cb2[np.clip(k0, 0, lb)]

    # Pearson correlation over the overlap, computed per lag.
    num = cnt * sab - sa * sb
    va = np.maximum(cnt * saa - sa ** 2, 1e-12)
    vb = np.maximum(cnt * sbb - sb ** 2, 1e-12)
    score = np.full(len(lags), -np.inf)
    score[valid] = num[valid] / np.sqrt(va[valid] * vb[valid])
    i = int(np.argmax(score))
    return int(lags[i]), float(score[i])


def band_table(x: np.ndarray, label: str) -> dict:
    N = 1 << 14
    segs = [np.abs(np.fft.rfft(x[s:s + N] * np.hanning(N))) ** 2
            for s in range(0, len(x) - N, N)]
    if not segs:
        return {}
    S = np.mean(segs, axis=0)
    f = np.fft.rfftfreq(N, 1 / SR)
    tot = S.sum()
    out = {}
    for lo, hi in [(0, 300), (300, 1000), (1000, 3400), (3400, 6000), (6000, 8000)]:
        m = (f >= lo) & (f < hi)
        out[(lo, hi)] = 100 * S[m].sum() / tot
    return out


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    phone_p, desk_p = Path(sys.argv[1]), Path(sys.argv[2])
    ph, dk = read_wav(phone_p), read_wav(desk_p)
    print(f"phone   {phone_p.name}: {len(ph)/SR:8.1f}s")
    print(f"desktop {desk_p.name}: {len(dk)/SR:8.1f}s\n")

    lag_env, score = best_lag(envelope(ph), envelope(dk))
    lag = lag_env * ENV_HOP
    print(f"coarse alignment: desktop lags phone by {lag/SR:+.2f}s "
          f"(envelope correlation {score:.3f})")
    # Two recordings of the same broadcast score ~0.99 on envelopes even when the encodes differ
    # (verified against synthetic gain and noise differences). Anything below ~0.7 means there is
    # no real alignment to find - most often because the recordings do not overlap in wall-clock
    # time at all, in which case no lag is correct.
    if score < 0.7:
        print("\nThe envelopes do not match well enough to trust an alignment. Most likely the two")
        print("recordings do not actually overlap in time, or they are not the same broadcast.")
        print("Check that both cover the same wall-clock window, and use a longer sample.")
        return 1

    # Overlap the two on the discovered lag.
    if lag >= 0:
        a, b = ph[lag:], dk
    else:
        a, b = ph, dk[-lag:]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    print(f"overlapping region: {n/SR:.1f}s")
    if n < SR * 20:
        print("Overlap too short to judge; record a longer simultaneous sample.")
        return 1

    # Refine to sample accuracy on a clean chunk from the middle.
    mid = n // 2
    w = min(SR * 10, n // 3)
    fa, fb = a[mid - w // 2: mid + w // 2], b[mid - w // 2: mid + w // 2]
    fine, _ = best_lag(fa.astype(np.float64), fb.astype(np.float64))
    print(f"fine alignment within that window: {fine:+d} samples ({fine/SR*1000:+.1f} ms)")
    if fine:
        if fine > 0:
            fa = fa[fine:]; fb = fb[:len(fa)]
        else:
            fb = fb[-fine:]; fa = fa[:len(fb)]

    r = float(np.corrcoef(fa, fb)[0, 1])
    print(f"\nwaveform correlation on aligned audio: r = {r:.4f}")

    pb, db_ = band_table(a, "phone"), band_table(b, "desktop")
    print(f"\n{'band':>16}  {'phone':>8}  {'desktop':>8}  {'diff':>7}")
    worst = 0.0
    for k in pb:
        d = db_[k] - pb[k]
        worst = max(worst, abs(d))
        print(f"{k[0]:6d}-{k[1]:<5d} Hz  {pb[k]:7.1f}%  {db_[k]:7.1f}%  {d:+6.1f}")

    print()
    if r > 0.8:
        print("VERDICT: the two sources carry essentially the same signal. Desktop recordings can")
        print("         be used as training data for the phone model.")
    elif r > 0.4:
        print("VERDICT: same broadcast, but the waveforms differ (different encode or processing).")
        print("         Usable, but validate the trained model on phone-recorded audio before")
        print("         trusting it - and prefer computing embeddings on the phone.")
    else:
        print("VERDICT: the audio differs substantially. Training on desktop audio risks a model")
        print("         that does not transfer. Record on the phone, or re-embed desktop audio")
        print("         through the phone before training.")
    if worst > 3:
        print(f"\nNote: largest band difference is {worst:.1f} percentage points - worth a look even")
        print("if the correlation is high, since YAMNet is sensitive to spectral balance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
