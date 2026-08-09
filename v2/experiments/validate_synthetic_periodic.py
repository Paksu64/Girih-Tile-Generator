"""
M2.5 validation: build a synthetic *periodic* test image (one real
generator-produced unit, tiled side by side N times), then check that
`periodic.py` correctly detects the period, reconstructs the unit, and
*attaches* copies at shared seam vertices instead of placing them as
independent islands -- same round-trip-against-ground-truth methodology as
`validate_synthetic.py` and the parked
`experimental_strapwork/validate_synthetic_strapwork.py`.

Note the synthetic source here isn't a single legal mega-tiling with
matching boundary edges (it's the same unit copy-pasted, not a pattern
grown to be genuinely period-P), so a tile straddling a seam in the
*source* generally won't exist to recover -- what this harness validates is
that (a) vertices really do get merged at seams (full vertex count should
be well below `unit_vertices * n_copies`) and (b) nothing breaks/regresses
in the process (tile match rate, false positives) relative to the older
copy-paste behavior.

Usage:
  python experiments/validate_synthetic_periodic.py --seed 1 --iterations 40 --copies 5
"""
import argparse
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from library.generator import GirihGenerator
from library.export import TILE_NAMES
from library.units import mm_to_px

from experiments import periodic


def render_tile_edges_tight(gen, path, width_px, height_px, dpi=150):
    """
    Like `validate_synthetic.render_tile_edges_only`, but clipped exactly to
    the nominal canvas box (no bbox-plus-margin framing). Needed here
    specifically: `render_tile_edges_only`'s margin means concatenated
    copies never actually touch at the pixel level (confirmed empirically --
    the old harness showed exactly unit_count * n_copies vertices every
    time, i.e. zero merging), so it can't exercise `replicate_and_attach`'s
    seam-merging at all. Rendering flush to the canvas edge means a tile
    that happens to land exactly on the boundary (real: `isVisiblePoint`'s
    `<=` bound allows this) gives the merge step actual seam material.

    Figure size is a fixed 10x10in (same as `render_tile_edges_only`) --
    *not* derived from the physical mm size -- so output resolution stays
    high regardless of canvas size. Tried tying it to the physical size
    first and detection quality collapsed (tiny canvases render at very
    few pixels per tile edge, confirmed empirically); resolution needs to
    be picked independently of what's being rendered.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_xlim(-width_px / 2, width_px / 2)
    ax.set_ylim(-height_px / 2, height_px / 2)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for tile in gen.tiles:
        verts = [gen.points[i].coords for i in tile.pointIndexes]
        verts.append(verts[0])
        xs_, ys_ = zip(*verts)
        ax.plot(xs_, ys_, color="black", linewidth=1.4)

    ax.axis("off")
    ax.margins(0)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def build_periodic_image(gen, out_dir, n_copies, width_px, height_px):
    unit_path = os.path.join(out_dir, "unit.png")
    render_tile_edges_tight(gen, unit_path, width_px, height_px)

    unit_img = cv2.imread(unit_path)
    tiled = cv2.hconcat([unit_img] * n_copies)
    tiled_path = os.path.join(out_dir, "periodic.png")
    cv2.imwrite(tiled_path, tiled)
    return tiled_path, unit_img.shape[1]


def run(seed=0, width_mm=750.0, height_mm=750.0, side_mm=25.0, iterations=40, symmetry_mode="c10", n_copies=5, out_dir=None):
    random.seed(seed)

    gen = GirihGenerator(
        width=mm_to_px(width_mm), height=mm_to_px(height_mm),
        sideLength=mm_to_px(side_mm), iterations=iterations, retries=10, symmetry_mode=symmetry_mode
    )
    gen.runGen3()

    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "_validation_out_periodic")
    os.makedirs(out_dir, exist_ok=True)
    img_path, unit_width_px = build_periodic_image(
        gen, out_dir, n_copies, mm_to_px(width_mm), mm_to_px(height_mm)
    )

    truth_counts = Counter(TILE_NAMES[t.tileType] for t in gen.tiles)
    truth_tile_count = len(gen.tiles)
    truth_vertex_count = len(gen.points)

    result = periodic.run(img_path, side_mm=side_mm, folder=os.path.join(out_dir, "reconstructed"),
                           n_copies=n_copies)

    print("=== Periodic (symmetry-based) round-trip validation ===")
    print(f"True unit image width   : {unit_width_px}px")
    print(f"Detected period         : {result['period_px']}")
    period_err = abs(result["period_px"][0] - unit_width_px) / unit_width_px
    print(f"Period detection error  : {period_err:.1%}")
    print(f"Copies requested/used   : {n_copies} / {result['n_copies']}")
    print()

    unit_counts = Counter(TILE_NAMES[t["tile_type"]] for t in result["unit_tiles"])
    print(f"Ground truth (1 unit)   : {truth_tile_count} tiles, {truth_vertex_count} vertices")
    print(f"Reconstructed (1 unit)  : {len(result['unit_tiles'])} tiles, "
          f"{len(result['unit_vertices'])} vertices, {len(result['unit_unresolved'])} unresolved")
    print()
    print(f"{'tile type':<12} {'truth':>6} {'recon':>6}")
    for name in ["decagon", "elong_hex", "pentagon", "bowtie", "rhombus"]:
        print(f"{name:<12} {truth_counts.get(name, 0):>6} {unit_counts.get(name, 0):>6}")

    over_counts = {k: v for k, v in (unit_counts - truth_counts).items() if v > 0}
    overlap = sum((unit_counts & truth_counts).values())
    tile_match_rate = overlap / max(1, sum(truth_counts.values()))
    print(f"\nUnit tile-type multiset match rate : {tile_match_rate:.1%}")
    print(f"False positives (unit)             : {over_counts or 'none'}")

    naive_vertices = len(result["unit_vertices"]) * result["n_copies"]
    merged_away = naive_vertices - len(result["vertices"])
    naive_tiles = len(result["unit_tiles"]) * result["n_copies"]
    print(f"\nFull attached design : {len(result['tiles'])} tiles, {len(result['vertices'])} vertices")
    print(f"  vs. naive copy-paste  : {naive_tiles} tiles, {naive_vertices} vertices")
    print(f"  seam vertices merged  : {merged_away}")
    print(f"  extra tiles recovered at seams (source isn't truly periodic, "
          f"so this is usually 0 here -- see docstring) : {len(result['tiles']) - naive_tiles}")
    if "stats" in result:
        print(f"Export stats            : {result['stats']}")

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=40)
    p.add_argument("--width", type=float, default=150.0)
    p.add_argument("--height", type=float, default=150.0)
    p.add_argument("--side", type=float, default=25.0)
    p.add_argument("--copies", type=int, default=5)
    p.add_argument("--symmetry", type=str, default="c10")
    args = p.parse_args()
    run(seed=args.seed, width_mm=args.width, height_mm=args.height,
        side_mm=args.side, iterations=args.iterations, n_copies=args.copies, symmetry_mode=args.symmetry)
