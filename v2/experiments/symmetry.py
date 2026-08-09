"""
Translational-symmetry detection: many real girih panels (including the
strip-style historical scan this was first tried against) repeat a single
unit cell across the surface. Detecting that period turns "detect the whole
pattern" into "detect one unit cell, then replicate it" -- a much smaller
problem to solve.

Note on scope: pixel-averaging multiple detected copies together (to denoise
a noisy real photo before reconstruction) was tried and dropped -- on a
hand-drawn source the drift *within* a single copy's height (not just
between copies) meant no single per-copy offset aligned it well, and mean/
median blending either blurred lines or erased them outright (confirmed
both empirically). Line-level voting across copies would be the correct
fix, but is real, separate work. This module's validated, in-use scope is
just the period detection itself, used by `periodic.py` to crop one unit
and replicate its (independently) reconstructed tiling across the
detected extent.
"""
import cv2
import numpy as np

from . import lines as lines_mod


def _signal(img):
    """
    Autocorrelation needs a signal dominated by the periodic *line*
    structure, not by paper texture/staining/shading -- raw inverted
    grayscale was tried first and failed exactly this way: those smooth,
    low-frequency variations have far more pixel-energy than the thin ink
    lines, so they swamped the correlation and produced a featureless
    monotonic decay with no real peak (confirmed empirically: the profile
    had no lag where correlation stood out from the general falloff trend).
    The Canny edge map is already the high-frequency, line-only signal
    `lines.py` uses for detection -- reusing it here sidesteps the problem
    entirely instead of hand-tuning a high-pass filter.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges = lines_mod.auto_canny(gray).astype(np.float32)
    edges -= edges.mean()
    return edges


def _autocorrelation(signal):
    h, w = signal.shape
    # pad to avoid circular wraparound contaminating the correlation
    ph, pw = 2 * h, 2 * w
    F = np.fft.fft2(signal, s=(ph, pw))
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)
    return ac  # center = (ph//2, pw//2) = zero lag


def detect_period(img, min_frac=0.03, max_frac=0.3, min_harmonics=5, dy_range=15):
    """
    Returns (dx, dy) -- the dominant non-trivial translation the pattern is
    (approximately) invariant under -- or None if no confident period is
    found. Search is restricted to horizontal-ish periods (small `dy_range`)
    since that's the case this was built and validated against (a strip
    panel); a full 2D lattice search is a natural extension, not needed yet.

    Scores each candidate period `P` by the *average* correlation across
    all its harmonics (P, 2P, 3P, ...) that fit in the image, not just the
    single strongest raw autocorrelation peak. Tried the single-peak
    version first and it failed on real data: real ink/paper noise easily
    produces one lucky-high peak at an unrelated lag, while a *true*
    period shows elevated correlation consistently across many multiples
    of itself. `min_harmonics` additionally guards against the reverse
    problem (long candidate periods with only 2-3 harmonics to average
    over are noisy small samples and win unfairly often) -- confirmed
    empirically: without it, 3-harmonic candidates dominated the ranking
    over the period that was visually and harmonically obviously correct.
    """
    signal = _signal(img)
    h, w = signal.shape
    ac = _autocorrelation(signal)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    zero_lag = ac[cy, cx]
    if zero_lag <= 0:
        return None

    def best_corr(dx):
        window = ac[cy - dy_range: cy + dy_range + 1, cx + dx]
        idx = int(np.argmax(window))
        return window[idx] / zero_lag, idx - dy_range

    lo = max(1, int(w * min_frac))
    hi = int(w * max_frac)
    if lo >= hi:
        return None

    best_score, best_P, best_dy = -1.0, None, 0
    for P in range(lo, hi):
        n_harmonics = int(0.9 * w // P)
        if n_harmonics < min_harmonics:
            continue
        vals_dys = [best_corr(k * P) for k in range(1, n_harmonics + 1)]
        score = sum(v for v, _ in vals_dys) / len(vals_dys)
        if score > best_score:
            best_score, best_P = score, P
            best_dy = vals_dys[0][1]  # dy at the fundamental (k=1)

    if best_P is None or best_score <= 0:
        return None
    return float(best_P), float(best_dy)


def confidence(img, period, min_frac=0.06, max_frac=0.6, dy_range=40):
    """Peak autocorrelation at the detected period, normalized by zero-lag
    (self-similarity) -- 1.0 would mean a perfect, noiseless repeat."""
    signal = _signal(img)
    ac = _autocorrelation(signal)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    dx, dy = period
    val = ac[cy + int(round(dy)), cx + int(round(dx))]
    zero_lag = ac[cy, cx]
    return float(val / zero_lag) if zero_lag else 0.0


def _rotate_edges(edges, center, angle_deg):
    h, w = edges.shape
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(edges, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)


def _rotational_score(edges, edges_area, center, n):
    """Fraction of edge pixels that still land on an edge pixel after
    rotating by each non-trivial multiple of 360/n around `center`,
    averaged over all n-1 rotations -- the rotational analog of
    `detect_period`'s harmonic-averaging (checking every fold, not just
    one, for the same small-sample-noise reason)."""
    if edges_area == 0:
        return 0.0
    total = 0.0
    for k in range(1, n):
        rotated = _rotate_edges(edges, center, k * 360.0 / n)
        total += float(np.count_nonzero((edges > 0) & (rotated > 0)))
    return total / (edges_area * (n - 1))


def detect_rotational_symmetry(img, folds=(10, 5), grid_step=None, downscale=4, dilate_px=2):
    """
    Returns {"center": (x, y), "n": int, "score": float} in *original*
    image pixel coordinates, or None if nothing scores above a token
    floor. Searches a coarse grid of candidate centers (on a downscaled
    image, for speed -- rotation+correlation at full res for every grid
    point over two fold counts is too slow to be worth the precision at
    the search stage) then refines the best candidate at full resolution
    with a tight local grid.

    The edge map is dilated before scoring: rotating a 1px-wide sparse
    line with nearest-neighbor interpolation is extremely unstable --
    confirmed empirically, moving the candidate center by a single pixel
    swung the score from 0.90 to 0.45 at what should be a smooth optimum.
    Dilating gives self-overlap after rotation some tolerance instead of
    requiring near-exact pixel alignment, smoothing the search landscape
    enough for a grid search to actually work.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edges_full = lines_mod.auto_canny(gray)
    if dilate_px > 0:
        kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), np.uint8)
        edges_full = cv2.dilate(edges_full, kernel)
    h, w = edges_full.shape

    small = cv2.resize(edges_full, (w // downscale, h // downscale), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape
    area_small = float(np.count_nonzero(small))
    if area_small == 0:
        return None

    step = grid_step or max(4, min(sh, sw) // 12)
    best = None
    for cy in range(step, sh - step, step):
        for cx in range(step, sw - step, step):
            for n in folds:
                score = _rotational_score(small, area_small, (cx, cy), n)
                if best is None or score > best[2]:
                    best = ((cx, cy), n, score)

    if best is None:
        return None

    # refine at full resolution around the best coarse candidate
    (bcx, bcy), best_n, _ = best
    fcx, fcy = bcx * downscale, bcy * downscale
    area_full = float(np.count_nonzero(edges_full))
    refine_radius = downscale * step
    refine_step = max(1, refine_radius // 6)

    best_full = None
    for cy in range(fcy - refine_radius, fcy + refine_radius + 1, refine_step):
        for cx in range(fcx - refine_radius, fcx + refine_radius + 1, refine_step):
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            for n in folds:
                score = _rotational_score(edges_full, area_full, (cx, cy), n)
                if best_full is None or score > best_full[2]:
                    best_full = ((cx, cy), n, score)

    if best_full is None:
        return None
    (fx, fy), n, score = best_full
    return {"center": (float(fx), float(fy)), "n": n, "score": score}
