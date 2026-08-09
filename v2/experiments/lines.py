"""
Stage 2: extract straight strapwork/tile-edge segments from an image.

Girih edges only ever run along a small discrete set of directions fixed by
the pattern's local 10-fold (36 degree) rotational symmetry. As an
*undirected* line, that collapses to 5 directions spread across a 180 degree
range (180 = 5 x 36). We exploit that hard: after generic Hough line
detection, find the dominant phase offset of the angle-mod-36 histogram
(circular mean, weighted by segment length) and snap every segment onto the
nearest of those 5 canonical directions. This single prior is what makes
extraction tractable on noisy photos -- it turns "detect an arbitrary
polygon mesh" into "detect lines near 5 known directions."
"""
import math
from collections import defaultdict

import cv2
import numpy as np


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def auto_canny(gray, sigma=0.33):
    med = float(np.median(gray))
    lo = int(max(0, (1.0 - sigma) * med))
    hi = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(gray, lo, hi)


def detect_segments(img, min_len=12, max_gap=8, threshold=25):
    gray = to_gray(img)
    edges = auto_canny(gray)
    raw = cv2.HoughLinesP(
        edges, 1, np.pi / 360, threshold=threshold,
        minLineLength=min_len, maxLineGap=max_gap,
    )
    if raw is None:
        return []
    raw = np.asarray(raw).reshape(-1, 4)
    return [
        ((float(x1), float(y1)), (float(x2), float(y2)))
        for x1, y1, x2, y2 in raw
    ]


def segment_angle_deg(seg):
    (x1, y1), (x2, y2) = seg
    return math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0


def segment_length(seg):
    (x1, y1), (x2, y2) = seg
    return math.hypot(x2 - x1, y2 - y1)


def find_dominant_phase(segments):
    """Length-weighted circular mean of angle-mod-36 (period 36 mapped to
    period 360 so the standard circular-mean trick applies)."""
    sx = sy = 0.0
    for seg in segments:
        L = segment_length(seg)
        theta = math.radians((segment_angle_deg(seg) % 36.0) * 10.0)
        sx += L * math.cos(theta)
        sy += L * math.sin(theta)
    if sx == 0.0 and sy == 0.0:
        return 0.0
    return (math.degrees(math.atan2(sy, sx)) / 10.0) % 36.0


def snap_segment(seg, phase_deg):
    (x1, y1), (x2, y2) = seg
    a = segment_angle_deg(seg)
    k = round((a - phase_deg) / 36.0)
    target = math.radians(phase_deg + 36.0 * k)
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = segment_length(seg) / 2.0
    dx, dy = math.cos(target) * half, math.sin(target) * half
    return ((mx - dx, my - dy), (mx + dx, my + dy))


def _merge_group(segs, gap_tol):
    """Segments sharing (near-)identical angle+perpendicular-offset get
    merged along their common direction by 1-D interval union.

    `gap_tol` must stay small (a few px): two *different*, colinear tile
    edges that share a real vertex are common in girih tilings (straight
    strap lines commonly run through several tile edges), and they touch at
    that vertex with ~0 gap -- exactly like a Hough detection fragmented by
    anti-aliasing noise. A large gap_tol can't tell those apart and silently
    fuses real vertices away; it should only bridge sub-pixel/few-px noise,
    not real joins."""
    a = segment_angle_deg(segs[0]) % 180.0
    rad = math.radians(a)
    ux, uy = math.cos(rad), math.sin(rad)
    nx, ny = -uy, ux
    ax, ay = segs[0][0]

    intervals = []
    for (x1, y1), (x2, y2) in segs:
        t1 = (x1 - ax) * ux + (y1 - ay) * uy
        t2 = (x2 - ax) * ux + (y2 - ay) * uy
        lo, hi = (t1, t2) if t1 <= t2 else (t2, t1)
        perp = (((x1 - ax) * nx + (y1 - ay) * ny) + ((x2 - ax) * nx + (y2 - ay) * ny)) / 2.0
        intervals.append([lo, hi, perp])
    intervals.sort(key=lambda iv: iv[0])

    merged = [intervals[0]]
    for iv in intervals[1:]:
        cur = merged[-1]
        if iv[0] <= cur[1] + gap_tol:
            cur[1] = max(cur[1], iv[1])
            cur[2] = (cur[2] + iv[2]) / 2.0
        else:
            merged.append(iv)

    out = []
    for lo, hi, perp in merged:
        p1 = (ax + ux * lo + nx * perp, ay + uy * lo + ny * perp)
        p2 = (ax + ux * hi + nx * perp, ay + uy * hi + ny * perp)
        out.append((p1, p2))
    return out


def merge_colinear(segments, perp_tol=4.0, gap_tol=3.0):
    """
    Group by angle bucket (safe to hard-bucket: segments here are already
    36-degree-snapped, so true duplicates share the exact same canonical
    angle) then cluster by perpendicular offset via sorted chaining rather
    than hard-bucketing (`round(perp / perp_tol)`): a hard bucket silently
    splits two near-duplicate detections of the same real line whenever
    they straddle a bucket boundary by just a couple of pixels -- traced
    directly to a real duplicate-detection case (~3px apart) that a hard
    bucket was leaving unmerged.
    """
    by_angle = defaultdict(list)
    for seg in segments:
        a = round(segment_angle_deg(seg) % 180.0)
        by_angle[a].append(seg)

    merged = []
    for a, segs in by_angle.items():
        rad = math.radians(a)
        nx, ny = -math.sin(rad), math.cos(rad)

        def perp_of(seg):
            (x1, y1), _ = seg
            return x1 * nx + y1 * ny

        segs_sorted = sorted(segs, key=perp_of)
        group = [segs_sorted[0]]
        group_perp = perp_of(segs_sorted[0])
        for seg in segs_sorted[1:]:
            p = perp_of(seg)
            if p - group_perp <= perp_tol:
                group.append(seg)
                group_perp = p
            else:
                merged.extend(_merge_group(group, gap_tol))
                group = [seg]
                group_perp = p
        merged.extend(_merge_group(group, gap_tol))

    return merged


def extract_lines(img, min_length=None):
    """Returns {"segments": [((x1,y1),(x2,y2)), ...], "phase_deg": float}."""
    raw = detect_segments(img)
    if not raw:
        return {"segments": [], "phase_deg": 0.0}

    phase = find_dominant_phase(raw)
    snapped = [snap_segment(s, phase) for s in raw]
    merged = merge_colinear(snapped)

    if min_length is not None:
        merged = [s for s in merged if segment_length(s) >= min_length]

    return {"segments": merged, "phase_deg": phase}
