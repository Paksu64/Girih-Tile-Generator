"""
EXPERIMENTAL / PARKED -- see `crossings.py` in this same package.

M1.5 validation: round-trip a synthetic design through the *strapwork* input
mode and compare against ground truth.

Renders the actual strapwork rays (`library/export.py`'s `_strap_segments`,
built from `_mid`/`DLENS`) as the test image -- the same geometry the
inversion in `strap_invert.py` was calibrated against -- rather than bare
tile edges (that's `experiments/validate_synthetic.py`, the active
tile-edges-mode harness).

Usage:
  python experiments/experimental_strapwork/validate_synthetic_strapwork.py --seed 1 --iterations 60
"""
import argparse
import os
import random
import sys
from collections import Counter

# Three levels up: experimental_strapwork/ -> experiments/ -> v2/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from library.generator import GirihGenerator
from library.export import TILE_NAMES, _strap_segments
from library.units import mm_to_px

from experiments.experimental_strapwork import pipeline_strapwork


def render_strapwork_only(gen, path, dpi=150, margin_px=40):
    xs = [p.coords[0] for p in gen.points]
    ys = [p.coords[1] for p in gen.points]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_xlim(min(xs) - margin_px, max(xs) + margin_px)
    ax.set_ylim(min(ys) - margin_px, max(ys) + margin_px)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for (x1, y1), (x2, y2) in _strap_segments(gen):
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=1.4, solid_capstyle="round")

    ax.axis("off")
    plt.tight_layout(pad=0.2)
    fig.savefig(path, dpi=dpi, facecolor="white")
    plt.close(fig)


def run(seed=0, width_mm=250.0, height_mm=250.0, side_mm=25.0, iterations=120, out_dir=None):
    random.seed(seed)

    gen = GirihGenerator(
        width=mm_to_px(width_mm), height=mm_to_px(height_mm),
        sideLength=mm_to_px(side_mm), iterations=iterations, retries=10,
    )
    gen.runGen3()

    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "_validation_out_strapwork")
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, "synthetic_strapwork.png")
    render_strapwork_only(gen, img_path)

    truth_counts = Counter(TILE_NAMES[t.tileType] for t in gen.tiles)
    truth_vertex_count = len(gen.points)
    truth_tile_count = len(gen.tiles)

    half_w, half_h = mm_to_px(width_mm) / 2.0, mm_to_px(height_mm) / 2.0

    def fully_interior(tile):
        return all(
            -half_w <= gen.points[i].coords[0] <= half_w
            and -half_h <= gen.points[i].coords[1] <= half_h
            for i in tile.pointIndexes
        )

    interior_tiles = [t for t in gen.tiles if fully_interior(t)]
    interior_counts = Counter(TILE_NAMES[t.tileType] for t in interior_tiles)

    result = pipeline_strapwork.run(
        img_path, side_mm=side_mm, folder=os.path.join(out_dir, "reconstructed"),
    )

    recon_counts = Counter(TILE_NAMES[t["tile_type"]] for t in result["tiles"])

    print("=== Strapwork round-trip validation (experimental / parked) ===")
    print(f"Ground truth  : {truth_tile_count} tiles ({len(interior_tiles)} fully interior), "
          f"{truth_vertex_count} vertices")
    print(f"Reconstructed : {len(result['tiles'])} tiles, {len(result['vertices'])} vertices, "
          f"{len(result['unresolved'])} unresolved faces")
    print(f"Fitted unit length: {result['unit_length_px']:.2f}px")
    if "stats" in result:
        print(f"Mask conflicts: {result['stats']['mask_conflicts']}")
    print()
    print(f"{'tile type':<12} {'truth':>6} {'interior':>9} {'recon':>6}")
    for name in ["decagon", "elong_hex", "pentagon", "bowtie", "rhombus"]:
        print(f"{name:<12} {truth_counts.get(name, 0):>6} {interior_counts.get(name, 0):>9} "
              f"{recon_counts.get(name, 0):>6}")

    over_counts = {k: v for k, v in (recon_counts - truth_counts).items() if v > 0}
    overlap = sum((recon_counts & truth_counts).values())
    tile_match_rate = overlap / max(1, sum(truth_counts.values()))
    interior_recall = overlap / max(1, sum(interior_counts.values()))
    vertex_match_rate = min(1.0, len(result["vertices"]) / max(1, truth_vertex_count))
    print(f"\nTile-type multiset match rate (of all truth tiles) : {tile_match_rate:.1%}")
    print(f"Recall against fully-interior truth tiles only      : {interior_recall:.1%}")
    print(f"Vertex count recovery rate                          : {vertex_match_rate:.1%}")
    print(f"False positives (recon tile-type counts exceeding truth): {over_counts or 'none'}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--width", type=float, default=200.0)
    p.add_argument("--height", type=float, default=200.0)
    p.add_argument("--side", type=float, default=25.0)
    args = p.parse_args()
    run(seed=args.seed, width_mm=args.width, height_mm=args.height,
        side_mm=args.side, iterations=args.iterations)
