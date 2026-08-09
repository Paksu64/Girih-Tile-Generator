"""
Stage 4: find the planar graph's minimal faces and classify each one as a
girih tile (or leave it "unresolved" -- e.g. photo damage/occlusion).

The interior-angle-sequence <-> TILE_ANGLES offset convention, and the
geometric formula for each vertex's angular "mask" contribution, were both
empirically validated against `GirihGenerator._addTileSingle`'s actual
output (placing every tile type at every offset and inspecting the resulting
point order / masks) rather than hand-derived from the turtle-graphics code,
since that point-ordering/offset convention is easy to get subtly wrong:

  * `_addTileSingle` always produces `pointIndexes` in CCW order (positive
    shoelace area) in the same y-up math convention used here.
  * The measured interior angle at `pointIndexes[i]` equals
    `TILE_ANGLES[tile_type][(i + offset - 1) % n]` -- i.e. the `offset`
    stored on a `Tile` is one step ahead of the naive rotation that matches
    a measured angle sequence to the canonical one.
  * A vertex's mask start sector is `round(outgoing_edge_angle_deg / 36) % 10`
    (the edge *leaving* that vertex, not the one arriving), spanning `count`
    sectors where `count` is that vertex's interior angle in 36-degree units.
"""
import math
from collections import defaultdict

from library.tiles import TILE_ANGLES

VALID_SIZES = {len(a) for a in TILE_ANGLES}  # {4, 5, 6, 10}


def _polygon_signed_area(pts):
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def _interior_angles_deg(pts):
    n = len(pts)
    out = []
    for i in range(n):
        pprev, pcur, pnext = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        a_in = math.atan2(pcur[1] - pprev[1], pcur[0] - pprev[0])
        a_out = math.atan2(pnext[1] - pcur[1], pnext[0] - pcur[0])
        turn = math.degrees((a_out - a_in + math.pi) % (2 * math.pi) - math.pi)
        out.append(180.0 - turn)
    return out


def _find_raw_faces(vertices, edges):
    """Standard planar-graph face traversal: at each vertex, sort incident
    edges by angle, and always continue into the next edge clockwise from
    the one just arrived on. Every directed edge is used in exactly one
    face cycle, including the outer/unbounded one(s)."""
    adj = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    for u in adj:
        adj[u].sort(key=lambda w: math.atan2(
            vertices[w][1] - vertices[u][1], vertices[w][0] - vertices[u][0]
        ))

    visited = set()
    faces = []
    for u, v in edges:
        for a, b in ((u, v), (v, u)):
            if (a, b) in visited:
                continue
            face = []
            cur_a, cur_b = a, b
            guard = 0
            limit = 4 * len(edges) + 16
            while (cur_a, cur_b) not in visited and guard < limit:
                visited.add((cur_a, cur_b))
                face.append(cur_a)
                neighbors = adj[cur_b]
                idx = neighbors.index(cur_a)
                nxt = neighbors[(idx - 1) % len(neighbors)]
                cur_a, cur_b = cur_b, nxt
                guard += 1
            if len(face) >= 3:
                faces.append(face)
    return faces


def _match_tile(seq):
    """seq: measured interior angles (36-degree units) in CCW vertex order.
    Returns (tile_type, offset) or None -- see module docstring for the
    offset convention."""
    n = len(seq)
    for tile_type, angles in enumerate(TILE_ANGLES):
        if len(angles) != n:
            continue
        for shift in range(n):
            if all(seq[i] == angles[(i + shift) % n] for i in range(n)):
                return tile_type, (shift + 1) % n
    return None


def classify_faces(vertices, edges, unit_length, angle_tol_deg=12.0):
    """
    Returns (tiles, unresolved):
      tiles: [{"tile_type": int, "offset": int, "vertex_ids": [CCW ids]}]
      unresolved: [[vertex ids]] -- valid-sized faces whose angles didn't
        match any known tile (candidate damage/occlusion/mis-detection).
    """
    raw_faces = _find_raw_faces(vertices, edges)

    tiles = []
    unresolved = []
    seen_vertex_sets = set()

    for face in raw_faces:
        n = len(face)
        if n not in VALID_SIZES:
            continue

        pts = [vertices[i] for i in face]
        if _polygon_signed_area(pts) < 0:
            face = face[::-1]
            pts = pts[::-1]

        key = frozenset(face)
        if key in seen_vertex_sets:
            continue

        angles_deg = _interior_angles_deg(pts)
        seq = []
        ok = True
        for a in angles_deg:
            units = a / 36.0
            r = round(units)
            if abs(units - r) * 36.0 > angle_tol_deg or r < 1:
                ok = False
                break
            seq.append(int(r))
        if not ok:
            continue

        match = _match_tile(seq)
        if match is None:
            unresolved.append(face)
            continue

        tile_type, offset = match
        tiles.append({"tile_type": tile_type, "offset": offset, "vertex_ids": face})
        seen_vertex_sets.add(key)

    return tiles, unresolved


def vertex_masks(vertices, tiles):
    """
    Per-vertex 10-bit angular mask, accumulated across every tile touching
    that vertex (see module docstring). Also returns the set of vertex ids
    where accumulated masks overlap -- a classification inconsistency worth
    surfacing rather than silently ignoring.
    """
    masks = defaultdict(lambda: [0] * 10)
    conflicts = set()

    for tile in tiles:
        face = tile["vertex_ids"]
        pts = [vertices[i] for i in face]
        n = len(face)
        interior = _interior_angles_deg(pts)

        for i in range(n):
            vid = face[i]
            cur = pts[i]
            nxt = pts[(i + 1) % n]
            out_deg = math.degrees(math.atan2(nxt[1] - cur[1], nxt[0] - cur[0])) % 360.0
            start = round(out_deg / 36.0) % 10
            count = max(1, round(interior[i] / 36.0))
            for j in range(count):
                sector = (start + j) % 10
                if masks[vid][sector] == 1:
                    conflicts.add(vid)
                masks[vid][sector] = 1

    return dict(masks), conflicts
