"""
M2.6 validation: build a synthetic rotationally-symmetric (c5/c10) test
image using the generator's own `symmetry_mode`, optionally blank out one
angular sector, then check that `rotational.py` correctly detects the
center and fold order and (when damaged) fills the missing sector by
rotation -- same round-trip-against-ground-truth methodology as the other
harnesses (`validate_synthetic.py`, `validate_synthetic_periodic.py`,
the parked `experimental_strapwork/validate_synthetic_strapwork.py`).

Usage:
  python experiments/validate_synthetic_rotational.py --seed 1 --symmetry c10
  python experiments/validate_synthetic_rotational.py --seed 1 --symmetry c10 --damage
"""
import argparse
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from library.generator import GirihGenerator
from library.export import TILE_NAMES
from library.units import mm_to_px

from experiments import rotational


def render_rosette(gen, path, dpi=150):
    """Renders tight to the pattern's own extent, centered at (0,0) -- the
    generator's own coordinate origin -- so the true rotation center is
    always exactly the image center, giving a known ground truth to check
    detection against."""
    xs = [p.coords[0] for p in gen.points]
    ys = [p.coords[1] for p in gen.points]
    W = 2 * max(max(abs(min(xs)), abs(max(xs))), max(abs(min(ys)), abs(max(ys)))) + 40
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.set_xlim(-W / 2, W / 2)
    ax.set_ylim(-W / 2, W / 2)
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


def apply_sector_damage(img_path, out_path, n, sector_to_damage=0, start_angle_deg=10):
    """Blanks out one 360/n-degree pie slice via cv2's own pie-slice
    drawing (not a hand-guessed polygon -- confirmed empirically that a
    rough triangular mask mostly missed the image bounds and damaged
    nothing)."""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    mask = np.zeros((h, w), dtype=np.uint8)
    big_r = int(1.5 * max(w, h))
    a0 = start_angle_deg + sector_to_damage * (360.0 / n)
    a1 = a0 + (360.0 / n)
    cv2.ellipse(mask, (int(cx), int(cy)), (big_r, big_r), 0, a0, a1, 255, -1)
    damaged = img.copy()
    damaged[mask > 0] = (255, 255, 255)
    cv2.imwrite(out_path, damaged)


def run(seed=0, width_mm=200.0, height_mm=200.0, side_mm=25.0, iterations=200,
        symmetry="c10", damage=False, out_dir=None):
    random.seed(seed)

    gen = GirihGenerator(
        width=mm_to_px(width_mm), height=mm_to_px(height_mm),
        sideLength=mm_to_px(side_mm), iterations=iterations, retries=10,
        symmetry_mode=symmetry,
    )
    gen.runGen3()

    out_dir = out_dir or os.path.join(os.path.dirname(__file__), "_validation_out_rotational")
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, "rosette.png")
    render_rosette(gen, img_path)

    true_n = {"c10": 10, "c5": 5}.get(symmetry)
    truth_counts = Counter(TILE_NAMES[t.tileType] for t in gen.tiles)
    truth_tile_count = len(gen.tiles)
    truth_vertex_count = len(gen.points)

    test_img_path = img_path
    if damage:
        test_img_path = os.path.join(out_dir, "rosette_damaged.png")
        apply_sector_damage(img_path, test_img_path, true_n, sector_to_damage=0)

    result = rotational.run(test_img_path, side_mm=side_mm,
                             folder=os.path.join(out_dir, "reconstructed"))

    print("=== Rotational (symmetry-based) round-trip validation ===")
    print(f"Source symmetry_mode : {symmetry}  (true fold order: {true_n})")
    print(f"Damage applied        : {'yes, sector 0 blanked' if damage else 'no'}")
    print(f"Detected center       : {result['center_px']}  (image center: "
          f"{(cv2.imread(img_path).shape[1] / 2, cv2.imread(img_path).shape[0] / 2)})")
    print(f"Detected fold order   : {result['n']}  "
          f"({'correct' if result['n'] == true_n else 'WRONG'})")
    print(f"Symmetry score        : {result['symmetry_score']:.3f}")
    print(f"Damaged sectors filled by rotation: {result['sector_stats']['damaged_sectors']}")
    print()

    recon_counts = Counter(TILE_NAMES[t["tile_type"]] for t in result["tiles"])
    print(f"Ground truth  : {truth_tile_count} tiles, {truth_vertex_count} vertices")
    print(f"Reconstructed : {len(result['tiles'])} tiles, {len(result['vertices'])} vertices, "
          f"{len(result['unresolved'])} unresolved")
    print()
    print(f"{'tile type':<12} {'truth':>6} {'recon':>6}")
    for name in ["decagon", "elong_hex", "pentagon", "bowtie", "rhombus"]:
        print(f"{name:<12} {truth_counts.get(name, 0):>6} {recon_counts.get(name, 0):>6}")

    over_counts = {k: v for k, v in (recon_counts - truth_counts).items() if v > 0}
    overlap = sum((recon_counts & truth_counts).values())
    tile_match_rate = overlap / max(1, sum(truth_counts.values()))
    print(f"\nTile-type multiset match rate : {tile_match_rate:.1%}")
    print(f"False positives                : {over_counts or 'none'}")
    if "stats" in result:
        print(f"Export stats                   : {result['stats']}")

    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--width", type=float, default=200.0)
    p.add_argument("--height", type=float, default=200.0)
    p.add_argument("--side", type=float, default=25.0)
    p.add_argument("--symmetry", choices=["c5", "c10"], default="c10")
    p.add_argument("--damage", action="store_true", help="blank out one sector to test gap-filling")
    args = p.parse_args()
    run(seed=args.seed, width_mm=args.width, height_mm=args.height,
        side_mm=args.side, iterations=args.iterations, symmetry=args.symmetry,
        damage=args.damage)
