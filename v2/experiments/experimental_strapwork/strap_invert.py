"""
EXPERIMENTAL / PARKED -- see `crossings.py` in this same package for why.

Strapwork input mode, stage 2: invert crossings back into tile-edge corner
points, then hand off to the same `vertices.build_graph_from_pairs` core
the tile-edge path uses -- so `faces.py`/`assemble.py` are unchanged.

The inversion formula below was *not* hand-derived from `library/export.py`'s
`_mid` and trusted -- it was derived, found wrong, and re-derived against
real generator output: `_mid` computes `alfa = atan2(x2 - x1, y2 - y1)` with
the (dx, dy) argument order swapped relative to the standard
`atan2(dy, dx)`, which introduces a 90-degree rotation between `alfa` and
the edge's actual on-screen direction. Accounting for that:

    beta_deg  = 144 - alfa_deg
    gamma_deg = 216 - alfa_deg
    =>  true_edge_dir = (beta_angle  - 54)  % 180   (if this angle is beta's)
    true_edge_dir = (gamma_angle - 126) % 180   (if this angle is gamma's)

Confirmed against 153 real edge instances from `GirihGenerator` to
floating-point-exact precision (error ~1e-14 deg). Since a crossing gives
two undirected angles with no inherent "which one is beta" label, both role
assignments are tried; `_mid` always derives beta and gamma from the same
`alfa` exactly 72 degrees apart, so only the physically-correct assignment
reproduces that relationship after mod-180 reduction -- the other diverges
by a large, easily-rejected margin (see `edge_direction_deg`).
"""
import math

from ..vertices import _cluster_points, build_graph_from_pairs


def _circ_diff(a, b, period=180.0):
    d = abs(a - b) % period
    return min(d, period - d)


def edge_direction_deg(angle1_deg, angle2_deg, tol_deg=3.0):
    """Undirected tile-edge direction (mod 180) recovered from a crossing's
    two line angles, or None if they aren't a physically consistent
    beta/gamma pair (e.g. a spurious/noisy crossing)."""
    cand1 = (angle1_deg - 54.0) % 180.0
    check1 = (angle2_deg - 126.0) % 180.0
    cand2 = (angle2_deg - 54.0) % 180.0
    check2 = (angle1_deg - 126.0) % 180.0

    d1 = _circ_diff(cand1, check1)
    d2 = _circ_diff(cand2, check2)

    if min(d1, d2) > tol_deg:
        return None
    return cand1 if d1 <= d2 else cand2


def corners_from_crossings(crossings, unit_length):
    """Two candidate corner points (the edge's real endpoints) per crossing,
    for a given candidate unit tile-edge length."""
    pairs = []
    for c in crossings:
        a1, a2 = c["angles"]
        edge_dir = edge_direction_deg(a1, a2)
        if edge_dir is None:
            continue
        rad = math.radians(edge_dir)
        ux, uy = math.cos(rad), math.sin(rad)
        mx, my = c["point"]
        half = unit_length / 2.0
        p1 = (mx + ux * half, my + uy * half)
        p2 = (mx - ux * half, my - uy * half)
        pairs.append((p1, p2))
    return pairs


def _coincidence_score(pairs, tol):
    """Fraction of candidate corner points that land in a cluster shared
    with at least one other point -- high when `unit_length` is right,
    since real shared tile corners should coincide; low/scattered when it's
    wrong, since every edge's corners then float independently."""
    points = [p for pair in pairs for p in pair]
    if not points:
        return 0.0
    _, cluster_of = _cluster_points(points, tol)
    counts = {}
    for cid in cluster_of.values():
        counts[cid] = counts.get(cid, 0) + 1
    matched = sum(v for v in counts.values() if v > 1)
    return matched / len(points)


def estimate_unit_length(crossings, steps=40, tol_frac=0.12):
    """
    Fit the unit tile-edge length by finding the L that makes the most
    candidate corner points from different edges coincide. Unlike the
    tile-edge path's simple "histogram peak of detected lengths" estimator,
    strapwork ray lengths are `L * DLENS_ratio` for a small per-tile-type
    ratio (see `library/tiles.DLENS`), so there's no single dominant peak to
    read off directly -- coincidence of independently-reconstructed corners
    is a robust proxy that doesn't need those ratios at all.
    """
    if len(crossings) < 2:
        return None

    pts = [c["point"] for c in crossings]
    dists = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])
            if d > 1e-6:
                dists.append(d)
    if not dists:
        return None
    dists.sort()
    rough = dists[len(dists) // 4]
    search_min, search_max = rough * 0.4, rough * 3.0

    best_L, best_score = None, -1.0
    steps = max(steps, 1)
    for i in range(steps + 1):
        L = search_min + (search_max - search_min) * i / steps
        pairs = corners_from_crossings(crossings, L)
        tol = max(2.0, L * tol_frac)
        score = _coincidence_score(pairs, tol)
        if score > best_score:
            best_score, best_L = score, L

    return best_L


def build_graph_from_strapwork(crossings, phase_deg=0.0):
    """
    crossings: output of crossings.find_crossings, already in *world* space
    (y up) -- see `pipeline_strapwork.py`'s `run`, which flips segments to
    world space before crossing-detection. This matters here specifically:
    `edge_direction_deg`'s formula is calibrated to `_mid`'s absolute
    y-up convention (unlike plain lattice-snapping, which only cares about
    angles *relative* to each other and so is orientation-agnostic), so
    feeding it un-flipped pixel-space angles would silently invert edges in
    the wrong direction. `crossings.find_crossings` and `phase_deg` must
    both already be in that same world-space convention.

    Returns the same {"vertices", "edges", "unit_length"} shape as
    vertices.build_graph, ready for faces.py.
    """
    unit_length = estimate_unit_length(crossings)
    if unit_length is None:
        return {"vertices": {}, "edges": [], "unit_length": 1.0}

    pairs = corners_from_crossings(crossings, unit_length)
    return build_graph_from_pairs(pairs, phase_deg=phase_deg, unit_length=unit_length)
