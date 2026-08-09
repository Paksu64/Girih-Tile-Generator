"""
Take an existing girih pattern (a design.json, already fully known -- no
image, no detection, no guessing) and repeat it across a 2D grid, attaching
copies at their shared edges instead of just placing them side by side.

Deliberately independent of `experiments/` (the photo-reconstruction
package): this doesn't reconstruct anything from a picture, it just takes
tiles whose identity is already 100% known and translates + fuses them.
The one piece of machinery worth reusing conceptually from that package is
the coordinate-offset-then-fuse idea validated in
`experiments/periodic.py`'s `replicate_and_attach` -- reimplemented here
directly on `Point`/`Tile` objects (not a raw CV-detected graph), since a
loaded design already has known tile identities and doesn't need face
classification to figure out what a polygon is.

Two translation vectors, not one: `periodic.py` only ever attached copies
side-to-side along one direction (a strip). This tiles both axes at once
(side-to-side AND top-to-top), so vertices get merged at horizontal seams,
vertical seams, and the four-way corners where diagonal neighbors meet, all
in a single pass.
"""
import math
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(__file__))

from library.export import load_json, run_exports
from library.tiles import Point, Tile
from library.units import DesignSpec, mm_to_px, px_to_mm


# ── generic geometry utilities (self-contained, no experiments/ dependency) ─

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


def _count_coincidences(orig_coords, translated_coords, tol):
    """How many `translated_coords` land within `tol` of some point in
    `orig_coords` -- grid-bucket lookup (same trick as `_cluster_points`),
    just counting instead of merging."""
    buckets = defaultdict(list)
    for i, (x, y) in enumerate(orig_coords):
        buckets[(round(x / tol), round(y / tol))].append(i)

    count = 0
    for x, y in translated_coords:
        gx, gy = round(x / tol), round(y / tol)
        hit = False
        for dgx in (-1, 0, 1):
            for dgy in (-1, 0, 1):
                for i in buckets.get((gx + dgx, gy + dgy), []):
                    ox, oy = orig_coords[i]
                    if math.hypot(x - ox, y - oy) <= tol:
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break
        if hit:
            count += 1
    return count


def find_attach_vector(points, tiles, axis, merge_tol_frac=0.05,
                        mag_range=(0.3, 2.0), dominance=3.0):
    """
    Search for the translation vector along `axis` ('x' or 'y') that
    maximizes vertex coincidence between the pattern and a translated copy
    of itself, *without any tile-tile area collision*. Directly optimizes
    what was asked for -- attach without colliding, more coincidence is
    better -- rather than assuming a fixed formula (the nominal canvas
    size was tried first and is wrong by ~2x, since `isVisiblePoint` in
    `library/generator.py` only bounds which points can *seed* growth, not
    how far a triggered tile's other vertices can land).

    Candidates come from the pattern's own pairwise vertex-coordinate
    differences, not a blind 2D grid search: the correct vector must equal
    one of these for at least the first vertex it successfully merges
    (confirmed in this session -- forcing two real vertices to coincide by
    using their exact coordinate difference as the translation produced an
    exact, correct merge). Filtered to candidates dominantly along the
    requested axis and within a plausible magnitude range of the pattern's
    own extent, to keep the search fast and avoid a tiny accidental
    coincidence winning by default.

    Returns (dx, dy, coincidence_count), or None if no collision-free
    candidate was found.
    """
    from shapely.geometry import Polygon
    from shapely.strtree import STRtree

    coords = [p.coords for p in points]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    bbox_w = max(xs) - min(xs) if coords else 0.0
    bbox_h = max(ys) - min(ys) if coords else 0.0
    scale_hint = bbox_w if axis == "x" else bbox_h
    if scale_hint <= 0:
        return None

    unit_length = tiles[0].sideLength if tiles else 1.0
    tol = max(2.0, unit_length * merge_tol_frac)
    lo, hi = mag_range[0] * scale_hint, mag_range[1] * scale_hint

    seen = set()
    candidates = []
    n = len(coords)
    for i in range(n):
        xi, yi = coords[i]
        for j in range(n):
            if i == j:
                continue
            dx = coords[j][0] - xi
            dy = coords[j][1] - yi
            primary, cross = (dx, dy) if axis == "x" else (dy, dx)
            if abs(primary) <= dominance * abs(cross):
                continue
            mag = abs(primary)
            if not (lo <= mag <= hi):
                continue
            key = (round(dx / tol), round(dy / tol))
            if key in seen:
                continue
            seen.add(key)
            candidates.append((dx, dy))

    if not candidates:
        return None

    orig_polys = []
    for t in tiles:
        poly = Polygon([points[i].coords for i in t.pointIndexes])
        if not poly.is_valid:
            poly = poly.buffer(0)
        orig_polys.append(poly)
    tree = STRtree(orig_polys)

    best = None
    for dx, dy in candidates:
        translated_coords = [(x + dx, y + dy) for x, y in coords]

        collided = False
        for t in tiles:
            tpoly = Polygon([translated_coords[i] for i in t.pointIndexes])
            if not tpoly.is_valid:
                tpoly = tpoly.buffer(0)
            for idx in tree.query(tpoly):
                if tpoly.intersection(orig_polys[idx]).area > 1e-6:
                    collided = True
                    break
            if collided:
                break
        if collided:
            continue

        count = _count_coincidences(coords, translated_coords, tol)
        if best is None or count > best[2]:
            best = (dx, dy, count)

    return best


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


def _recompute_masks(points, tiles):
    """
    Per-vertex 10-bit angular mask, accumulated across every tile touching
    that vertex -- same geometric formula validated in
    `experiments/faces.py` (mask start sector = outgoing edge direction /
    36, spanning `count` sectors = that vertex's interior angle in
    36-degree units), reimplemented here so this module doesn't import
    from experiments/. Needs recomputing after tiling: a vertex that used
    to sit on a design's outer boundary (partially-open mask) may now be an
    interior vertex shared by tiles from up to 4 grid copies.
    """
    masks = [[0] * 10 for _ in points]
    for tile in tiles:
        idxs = tile.pointIndexes
        n = len(idxs)
        pts = [points[i].coords for i in idxs]
        interior = _interior_angles_deg(pts)
        for i in range(n):
            vid = idxs[i]
            cur, nxt = pts[i], pts[(i + 1) % n]
            out_deg = math.degrees(math.atan2(nxt[1] - cur[1], nxt[0] - cur[0])) % 360.0
            start = round(out_deg / 36.0) % 10
            count = max(1, round(interior[i] / 36.0))
            for j in range(count):
                masks[vid][(start + j) % 10] = 1
    for p, m in zip(points, masks):
        p.mask = m


# ── design loading (mirrors main.py's reconstruct_generator) ────────────────

def load_design(path):
    data = load_json(path)
    spec_data = data["spec"]
    spec = DesignSpec(spec_data["width_mm"], spec_data["height_mm"], spec_data["side_mm"])
    side_px = spec_data.get("side_px", mm_to_px(spec_data["side_mm"]))

    points = []
    for vertex in data["vertices"]:
        points.append(Point(coords=(vertex["x_px"], vertex["y_px"]), mask=vertex["mask"]))

    tiles = []
    for tile_data in data["tiles"]:
        tile_side_px = tile_data.get("side_px", mm_to_px(tile_data["side_mm"]))
        tiles.append(Tile(
            tileType=tile_data["type_id"],
            pointIndexes=tile_data["vertex_ids"],
            offset=tile_data["offset"],
            sideLength=tile_side_px,
            rotationDeg=tile_data.get("rotation_deg", 0.0),
            masks={},
        ))

    return points, tiles, spec, side_px


# ── the actual 2D tiling ─────────────────────────────────────────────────

def tile_grid(points, tiles, vec_x, vec_y, nx, ny, merge_tol_frac=0.15, side_px=None):
    """
    Repeat (points, tiles) across an `nx` x `ny` grid using two translation
    vectors `vec_x = (dx, dy)` (side-to-side) and `vec_y = (dx, dy)`
    (top-to-top) -- full 2D vectors, not axis-aligned scalars, since
    `find_attach_vector` can return a vector with a small cross-axis
    component. Copy (i, j) is offset by `i*vec_x + j*vec_y`. Fuses
    vertices that land on the same point across a copy boundary (same
    tolerance-based clustering approach validated in
    `experiments/periodic.replicate_and_attach`, generalized here to two
    independent translation axes instead of one).

    Returns (new_points, new_tiles).
    """
    vx_dx, vx_dy = vec_x
    vy_dx, vy_dy = vec_y

    raw_coords = []
    raw_masks = []
    tile_refs = []  # (tile_type, offset, side_length, rotation_deg, [raw point indices])

    for i in range(nx):
        for j in range(ny):
            ox, oy = i * vx_dx + j * vy_dx, i * vx_dy + j * vy_dy
            base = len(raw_coords)
            for p in points:
                x, y = p.coords
                raw_coords.append((x + ox, y + oy))
                raw_masks.append(p.mask)
            for t in tiles:
                tile_refs.append((
                    t.tileType, t.offset, t.sideLength, t.rotationDeg,
                    [base + v for v in t.pointIndexes],
                ))

    unit_length = side_px if side_px is not None else (tiles[0].sideLength if tiles else 1.0)
    tol = max(3.0, unit_length * merge_tol_frac)
    centers, cluster_of = _cluster_points(raw_coords, tol)

    new_points = [Point(coords=c, mask=[0] * 10) for c in centers]

    new_tiles = []
    for tile_type, offset, side_length, rotation_deg, vids in tile_refs:
        merged_vids = [cluster_of[v] for v in vids]
        new_tiles.append(Tile(
            tileType=tile_type, pointIndexes=merged_vids, offset=offset,
            sideLength=side_length, rotationDeg=rotation_deg, masks={},
        ))

    _recompute_masks(new_points, new_tiles)
    return new_points, new_tiles


def run(load_path, nx, ny, output_folder, groove_width_mm=1.5, preview_mode="full"):
    points, tiles, spec, side_px = load_design(load_path)

    x_found = find_attach_vector(points, tiles, axis="x")
    y_found = find_attach_vector(points, tiles, axis="y")
    if x_found is None:
        raise ValueError(
            "No collision-free side-to-side attach vector found -- pattern "
            "may be too small/sparse for a reliable match."
        )
    if y_found is None:
        raise ValueError(
            "No collision-free top-to-top attach vector found -- pattern "
            "may be too small/sparse for a reliable match."
        )

    x_dx, x_dy, x_count = x_found
    y_dx, y_dy, y_count = y_found

    new_points, new_tiles = tile_grid(
        points, tiles, (x_dx, x_dy), (y_dx, y_dy), nx, ny, side_px=side_px
    )

    xs = [p.coords[0] for p in new_points]
    ys = [p.coords[1] for p in new_points]
    margin_px = side_px if side_px else mm_to_px(spec.side_mm)
    result_width_mm = px_to_mm((max(xs) - min(xs)) + 2 * margin_px) if xs else spec.width_mm
    result_height_mm = px_to_mm((max(ys) - min(ys)) + 2 * margin_px) if ys else spec.height_mm
    new_spec = DesignSpec(result_width_mm, result_height_mm, spec.side_mm)

    gen = SimpleNamespace(points=new_points, tiles=new_tiles)

    os.makedirs(output_folder, exist_ok=True)
    run_exports(gen, new_spec, output_folder, groove_width_mm=groove_width_mm, preview_mode=preview_mode)

    return {
        "vertex_count": len(new_points),
        "tile_count": len(new_tiles),
        "naive_vertex_count": len(points) * nx * ny,
        "grid": (nx, ny),
        "spec": new_spec,
        "x_vector_mm": (px_to_mm(x_dx), px_to_mm(x_dy)),
        "y_vector_mm": (px_to_mm(y_dx), px_to_mm(y_dy)),
        "x_coincidence_count": x_count,
        "y_coincidence_count": y_count,
    }


def _resolve_designs_dir():
    designs_dir = os.path.join(os.path.dirname(__file__), "designs")
    os.makedirs(designs_dir, exist_ok=True)
    return designs_dir


def _resolve_under_designs(value, designs_dir):
    """Same convention as main.py: an absolute path is used as-is, a bare
    name resolves under designs/, so output lands alongside everything else
    generated by this tool instead of scattered relative to wherever the
    command happened to be run from."""
    if os.path.isabs(value):
        return value
    if os.path.isdir(value):
        return value
    return os.path.join(designs_dir, value)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Repeat an existing girih design.json across a 2D grid, "
                    "attaching copies at shared edges."
    )
    p.add_argument("--load", required=True, metavar="DIR",
                   help="Folder containing design.json -- a bare name resolves under designs/")
    p.add_argument("--nx", type=int, default=2, help="Copies side-to-side (default 2)")
    p.add_argument("--ny", type=int, default=2, help="Copies top-to-top (default 2)")
    p.add_argument("--output", required=True, metavar="DIR",
                   help="A bare name resolves under designs/, same as --load")
    p.add_argument("--groove", type=float, default=1.5, metavar="MM")
    p.add_argument("--preview", choices=["full", "strapwork"], default="full")
    args = p.parse_args()

    designs_dir = _resolve_designs_dir()
    load_folder = _resolve_under_designs(args.load, designs_dir)
    output_folder = _resolve_under_designs(args.output, designs_dir)

    load_path = load_folder
    if not load_path.lower().endswith(".json"):
        load_path = os.path.join(load_path, "design.json")

    result = run(load_path, args.nx, args.ny, output_folder,
                 groove_width_mm=args.groove, preview_mode=args.preview)
    xv = result["x_vector_mm"]
    yv = result["y_vector_mm"]
    print(f"X attach vector : ({xv[0]:.2f}, {xv[1]:.2f}) mm  "
          f"({result['x_coincidence_count']} vertices coincide)")
    print(f"Y attach vector : ({yv[0]:.2f}, {yv[1]:.2f}) mm  "
          f"({result['y_coincidence_count']} vertices coincide)")
    print(f"Grid       : {result['grid'][0]} x {result['grid'][1]}")
    print(f"Vertices   : {result['vertex_count']} (naive copy-paste would be {result['naive_vertex_count']})")
    print(f"Tiles      : {result['tile_count']}")
    print(f"Canvas     : {result['spec'].width_mm:.1f} x {result['spec'].height_mm:.1f} mm")
    print(f"\nWriting {output_folder}/ ...")
    print(
        "  ✓  design.json\n"
        "  ✓  design.dxf\n"
        "  ✓  design_engrave.dxf\n"
        "  ✓  design.svg\n"
        "  ✓  design.png\n"
        "  ✓  design_info.txt\n"
        "Done."
    )
