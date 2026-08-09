"""
Stage 5: turn classified faces into the same Point/Tile objects
`library/generator.py` produces, then hand off to the existing
`library/export.py` pipeline unchanged -- reconstructed designs get
DXF/DXF-engrave/SVG/PNG/JSON for free, and load cleanly through
`v2/main.py --load <folder> --continue` (masks are populated so the
generator can legally extend/repair the tiling around reconstructed
vertices instead of treating them as wide open).
"""
import os
from types import SimpleNamespace

from library.tiles import Point, Tile
from library.units import DesignSpec, mm_to_px, px_to_mm
from library.export import run_exports

from .faces import vertex_masks


def build_generator(vertices, edges, tiles, unresolved, unit_length, side_mm=25.0):
    """
    Returns (gen, spec, stats). `gen` duck-types the subset of
    GirihGenerator that library/export.py's save_* functions read:
    .points and .tiles.
    """
    scale = mm_to_px(side_mm) / unit_length if unit_length > 1e-9 else 1.0

    xs = [x for x, _ in vertices.values()]
    ys = [y for _, y in vertices.values()]
    cx = (max(xs) + min(xs)) / 2.0 if xs else 0.0
    cy = (max(ys) + min(ys)) / 2.0 if ys else 0.0

    masks_by_vertex, conflicts = vertex_masks(vertices, tiles)

    points_by_id = {}
    for vid, (x, y) in vertices.items():
        px = (x - cx) * scale
        py = (y - cy) * scale
        mask = masks_by_vertex.get(vid, [0] * 10)
        points_by_id[vid] = Point(coords=(px, py), mask=mask[:])

    ordered_ids = sorted(points_by_id)
    id_to_index = {vid: i for i, vid in enumerate(ordered_ids)}
    points = [points_by_id[vid] for vid in ordered_ids]

    gen_tiles = []
    for t in tiles:
        idxs = [id_to_index[vid] for vid in t["vertex_ids"]]
        gen_tiles.append(Tile(
            tileType=t["tile_type"],
            pointIndexes=idxs,
            offset=t["offset"],
            sideLength=mm_to_px(side_mm),
            rotationDeg=0.0,
            masks={},
        ))

    margin_px = mm_to_px(side_mm)
    width_px = ((max(xs) - min(xs)) * scale if xs else 0.0) + 2 * margin_px
    height_px = ((max(ys) - min(ys)) * scale if ys else 0.0) + 2 * margin_px
    spec = DesignSpec(
        width_mm=px_to_mm(width_px),
        height_mm=px_to_mm(height_px),
        side_mm=side_mm,
    )

    gen = SimpleNamespace(
        points=points,
        tiles=gen_tiles,
        width=width_px,
        height=height_px,
        symmetry_mode="none",
    )

    stats = {
        "vertex_count": len(points),
        "tile_count": len(gen_tiles),
        "unresolved_faces": len(unresolved),
        "mask_conflicts": len(conflicts),
    }
    return gen, spec, stats


def reconstruct_to_folder(vertices, edges, tiles, unresolved, unit_length, folder,
                           side_mm=25.0, groove_width_mm=1.5):
    gen, spec, stats = build_generator(vertices, edges, tiles, unresolved, unit_length, side_mm=side_mm)
    os.makedirs(folder, exist_ok=True)
    run_exports(gen, spec, folder, groove_width_mm=groove_width_mm, preview_mode="full")
    return gen, spec, stats
