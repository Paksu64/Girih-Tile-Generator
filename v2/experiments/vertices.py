"""
Stage 3: turn a bag of line segments into a clean planar graph -- unique
vertices at consistent (x, y) positions and the edges between them -- refined
onto the exact algebraic girih lattice (unit edge length, 36-degree-multiple
directions).

Coordinates are converted here from image pixel space (y grows downward) to
the "y grows upward" world-space convention `library/generator.py` already
uses, so every downstream angle computation (face interior angles, tile
offsets, vertex masks) lines up with `library/tiles.TILE_ANGLES` without any
special-casing.
"""
import math
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr


def _flip_y(segments):
    return [((x1, -y1), (x2, -y2)) for (x1, y1), (x2, y2) in segments]


class _UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def _cluster_points(points, tol):
    """Grid-bucket + union-find clustering of near-duplicate points."""
    uf = _UnionFind()
    buckets = defaultdict(list)
    for i, (x, y) in enumerate(points):
        buckets[(round(x / tol), round(y / tol))].append(i)

    for (gx, gy), idxs in buckets.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for j in buckets.get((gx + dx, gy + dy), []):
                    for i in idxs:
                        if i == j:
                            continue
                        xi, yi = points[i]
                        xj, yj = points[j]
                        if math.hypot(xi - xj, yi - yj) <= tol:
                            uf.union(i, j)

    groups = defaultdict(list)
    for i in range(len(points)):
        groups[uf.find(i)].append(i)

    cluster_of = {}
    centers = []
    for members in groups.values():
        cx = sum(points[i][0] for i in members) / len(members)
        cy = sum(points[i][1] for i in members) / len(members)
        cid = len(centers)
        centers.append((cx, cy))
        for i in members:
            cluster_of[i] = cid
    return centers, cluster_of


def _estimate_unit_length(centers, raw_edges):
    lengths = []
    for u, v in raw_edges:
        xu, yu = centers[u]
        xv, yv = centers[v]
        L = math.hypot(xv - xu, yv - yu)
        if L > 1e-3:
            lengths.append(L)
    if not lengths:
        return 1.0
    lengths.sort()
    lo, hi = lengths[0], lengths[-1]
    if hi - lo < 1e-6:
        return lengths[len(lengths) // 2]
    nbins = max(10, min(50, len(lengths)))
    hist, edges = np.histogram(lengths, bins=nbins, range=(lo, hi))
    peak = int(np.argmax(hist))
    lo_e, hi_e = edges[peak], edges[peak + 1]
    in_peak = [l for l in lengths if lo_e <= l <= hi_e]
    return float(np.mean(in_peak)) if in_peak else float(np.median(lengths))


def _refine_lattice(centers, raw_edges, unit_length, phase_deg, anchor_weight=0.05):
    """
    Least-squares snap of vertex positions onto the exact lattice: for every
    edge (u, v), (v - u) should equal unit_length * (cos, sin) of the nearest
    canonical 36-degree direction to the edge's observed angle. Edge rows
    never mix x and y, so the two coordinates are solved as independent
    linear systems, each regularized by a light anchor back to the originally
    detected position (otherwise the edge constraints alone only pin down
    the graph up to a rigid translation).
    """
    n = len(centers)
    if n == 0 or not raw_edges:
        return list(centers)

    rows, cols, data = [], [], []
    rhs_x, rhs_y = [], []
    r = 0
    for u, v in raw_edges:
        xu, yu = centers[u]
        xv, yv = centers[v]
        ang = math.degrees(math.atan2(yv - yu, xv - xu))
        k = round((ang - phase_deg) / 36.0)
        target = math.radians(phase_deg + 36.0 * k)
        cu, cv = math.cos(target), math.sin(target)

        rows += [r, r]
        cols += [u, v]
        data += [-1.0, 1.0]
        rhs_x.append(unit_length * cu)
        rhs_y.append(unit_length * cv)
        r += 1

    for i in range(n):
        rows.append(r)
        cols.append(i)
        data.append(anchor_weight)
        rhs_x.append(anchor_weight * centers[i][0])
        rhs_y.append(anchor_weight * centers[i][1])
        r += 1

    A = coo_matrix((data, (rows, cols)), shape=(r, n)).tocsr()
    sol_x = lsqr(A, np.array(rhs_x))[0]
    sol_y = lsqr(A, np.array(rhs_y))[0]
    return list(zip(sol_x.tolist(), sol_y.tolist()))


def build_graph_from_pairs(pairs, phase_deg=0.0, unit_length=None, cluster_tol=None):
    """
    Shared core: turn a list of point-pairs (each pair = the two endpoints
    of one graph edge, already in world/y-up space) into the clean planar
    graph faces.py expects. Used by `build_graph` (tile-edge input, pairs =
    raw segment endpoints), and previously also by the parked strapwork
    mode's `experimental_strapwork.strap_invert.build_graph_from_strapwork`
    (pairs = each crossing's two inverted corner points) -- kept generic
    enough that reusing it there needed no changes here.

    If `unit_length` is given, it's trusted as-is instead of being
    re-estimated (the strapwork path fit its own via a method suited to
    that input's different geometry; not needed by the current caller).
    """
    if not pairs:
        return {"vertices": {}, "edges": [], "unit_length": unit_length or 1.0}

    raw_points = []
    pair_idx = []
    for a, b in pairs:
        pair_idx.append((len(raw_points), len(raw_points) + 1))
        raw_points.append(a)
        raw_points.append(b)

    def cluster_and_link(tol):
        centers, cluster_of = _cluster_points(raw_points, tol)
        edges = set()
        for i0, i1 in pair_idx:
            u, v = cluster_of[i0], cluster_of[i1]
            if u != v:
                edges.add((min(u, v), max(u, v)))
        return centers, list(edges)

    # Bootstrap: a first, deliberately loose pass just to get a ballpark
    # unit length, since the "right" clustering tolerance is a fraction of
    # that length -- not of average pair length, which is dominated by long
    # multi-edge straps and badly overestimates how close two detections of
    # the *same* real corner can land.
    if cluster_tol is None:
        avg_len = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in pairs) / len(pairs)
        bootstrap_tol = max(3.0, avg_len * 0.15)
        _, bootstrap_edges = cluster_and_link(bootstrap_tol)
        bootstrap_centers, _ = _cluster_points(raw_points, bootstrap_tol)
        bootstrap_unit = unit_length or _estimate_unit_length(bootstrap_centers, bootstrap_edges)
        cluster_tol = max(bootstrap_tol, bootstrap_unit * 0.3)

    centers, raw_edges = cluster_and_link(cluster_tol)

    if unit_length is None:
        unit_length = _estimate_unit_length(centers, raw_edges)
    refined = _refine_lattice(centers, raw_edges, unit_length, phase_deg)

    # Final cleanup: refinement can occasionally pull two originally-distinct
    # raw clusters onto (near-)identical final positions -- the pre-
    # refinement topology alone doesn't always carry enough constraints to
    # keep them apart. Re-cluster the refined positions and remap edges so
    # the planar graph handed to faces.py has no duplicate vertices (even a
    # single leftover duplicate can make face-finding leak a large merged
    # face instead of the many small ones it should find).
    dedup_tol = max(3.0, unit_length * 0.15)
    final_centers, final_cluster_of = _cluster_points(refined, dedup_tol)
    final_edges = set()
    for u, v in raw_edges:
        fu, fv = final_cluster_of[u], final_cluster_of[v]
        if fu != fv:
            final_edges.add((min(fu, fv), max(fu, fv)))

    vertices = {i: final_centers[i] for i in range(len(final_centers))}
    return {"vertices": vertices, "edges": list(final_edges), "unit_length": unit_length}


def build_graph(segments, phase_deg=0.0, cluster_tol=None):
    """
    segments: list of ((x1,y1),(x2,y2)) in image pixel space (y down).
    Returns {"vertices": {id: (x, y)}, "edges": [(u, v), ...], "unit_length": float}
    in world space (y up), ready for faces.py.
    """
    segments = _flip_y(segments)
    return build_graph_from_pairs(segments, phase_deg=phase_deg, cluster_tol=cluster_tol)
