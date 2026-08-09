"""
EXPERIMENTAL / PARKED -- see this package's __init__ docstring context in
the project plan (`M1.5` section). The closed-form math here
(`strap_invert.py`) is validated exact; this file (crossing detection) is
solid too. What's unresolved is upstream of both: real-photo crossing-
matching recall plateaus around 50%, traced to detection completeness, not
a geometry bug. Moved out of the active `experiments/` package to keep
that folder focused on the current working model (tile-edges + periodic);
not deleted because the math is real, tested work worth keeping.

Strapwork input mode, stage 1: find true line-crossings (an X, not just two
segments touching endpoint-to-endpoint) in a bag of 36-degree-snapped
strapwork segments from `lines.py`.

Each crossing corresponds to exactly one girih tile edge's shared midpoint
-- see `strap_invert.py`'s module docstring for why, and for the closed-form
inversion that turns a crossing into that edge's two real corner points.
A boundary edge (missing/damaged neighbor) shows as a partial "V" -- two
segments meeting at an *endpoint*, not crossing through their interiors --
so it's naturally excluded here rather than misread as a crossing.
"""
from collections import defaultdict

from ..lines import segment_angle_deg, segment_length
from ..vertices import _cluster_points


def _line_intersection(seg_a, seg_b):
    (x1, y1), (x2, y2) = seg_a
    (x3, y3), (x4, y4) = seg_b
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(d) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / d
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / d
    return t, u, (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _angle_bucket(angle_deg, phase_deg):
    return round((angle_deg - phase_deg) / 36.0) % 10


def find_crossings(segments, phase_deg, interior_margin=0.08, cluster_tol=None):
    """
    segments: 36-degree-snapped segments (image pixel space) from lines.py.
    Returns [{"point": (x, y), "angles": (a1_deg, a2_deg)}, ...] -- angles
    are the two canonical (already phase-snapped) undirected line angles
    crossing at that point, mod 180.
    """
    raw = []
    n = len(segments)
    buckets = [_angle_bucket(segment_angle_deg(s), phase_deg) for s in segments]

    for i in range(n):
        for j in range(i + 1, n):
            if buckets[i] == buckets[j]:
                continue  # parallel/colinear, not a crossing

            hit = _line_intersection(segments[i], segments[j])
            if hit is None:
                continue
            t, u, pt = hit
            if not (interior_margin <= t <= 1 - interior_margin):
                continue
            if not (interior_margin <= u <= 1 - interior_margin):
                continue
            raw.append((pt, buckets[i], buckets[j]))

    if not raw:
        return []

    if cluster_tol is None:
        avg_len = sum(segment_length(s) for s in segments) / len(segments)
        cluster_tol = max(3.0, avg_len * 0.08)

    points = [r[0] for r in raw]
    centers, cluster_of = _cluster_points(points, cluster_tol)

    bucket_sets = defaultdict(set)
    for idx, (_, bi, bj) in enumerate(raw):
        cid = cluster_of[idx]
        bucket_sets[cid].add(bi)
        bucket_sets[cid].add(bj)

    crossings = []
    for cid, center in enumerate(centers):
        buckets_here = sorted(bucket_sets[cid])
        if len(buckets_here) != 2:
            continue  # noisy/ambiguous crossing (not exactly 2 lines); skip
        angles = tuple((phase_deg + 36.0 * b) % 180.0 for b in buckets_here)
        crossings.append({"point": center, "angles": angles})
    return crossings
