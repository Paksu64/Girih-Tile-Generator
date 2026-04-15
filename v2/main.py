#!/usr/bin/env python3
"""
Command-line entry point for the Girih tiling generator.

Examples
--------
Create a new design:
  python main.py --width 600mm --height 400mm --side 25mm

Create a new symmetric design:
  python main.py --width 600mm --height 600mm --side 25mm --symmetry c10

Load an existing design and continue growing it:
  python main.py --load girih_design_001 --continue --iterations 300
"""

import argparse
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from girih.export import TILE_NAMES, load_json, next_design_folder, run_exports
from girih.generator import GirihGenerator
from girih.tiles import Point, Tile
from girih.units import DesignSpec, mm_to_px, parse_mm


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Generate Girih Islamic geometric tilings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--width", metavar="MM")
    parser.add_argument("--height", metavar="MM")
    parser.add_argument("--side", metavar="MM")
    parser.add_argument("--output", metavar="DIR")
    parser.add_argument("--load", metavar="DIR")
    parser.add_argument("--continue", dest="do_continue", action="store_true")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--symmetry",
        choices=["none", "c5", "c10"],
        default=None,
        help="Rotational symmetry mode for a new design, or an override when loading one.",
    )
    return parser


def reconstruct_generator(data):
    spec_data = data["spec"]
    spec = DesignSpec(spec_data["width_mm"], spec_data["height_mm"], spec_data["side_mm"])

    side_px = spec_data.get("side_px", mm_to_px(spec_data["side_mm"]))
    symmetry_mode = spec_data.get("symmetry_mode", "none")

    gen = GirihGenerator(
        width=mm_to_px(spec.width_mm),
        height=mm_to_px(spec.height_mm),
        sideLength=side_px,
        symmetry_mode=symmetry_mode,
    )

    for vertex in data["vertices"]:
        point = Point(coords=(vertex["x_px"], vertex["y_px"]), mask=vertex["mask"])
        gen.points.append(point)
        dist = (vertex["x_px"] ** 2 + vertex["y_px"] ** 2) ** 0.5
        gen._registerDistance(vertex["id"], dist)

    for tile_data in data["tiles"]:
        tile_side_px = tile_data.get("side_px", mm_to_px(tile_data["side_mm"]))
        tile = Tile(
            tileType=tile_data["type_id"],
            pointIndexes=tile_data["vertex_ids"],
            offset=tile_data["offset"],
            sideLength=tile_side_px,
            rotationDeg=tile_data.get("rotation_deg", 0.0),
            masks={},
        )
        gen.tiles.append(tile)

    return gen, spec, side_px


def print_stats(gen):
    counts = Counter(TILE_NAMES[tile.tileType] for tile in gen.tiles)
    print(f"\nTiles {len(gen.tiles)}   Vertices {len(gen.points)}")
    for name in ["decagon", "elong_hex", "pentagon", "bowtie", "rhombus"]:
        count = counts.get(name, 0)
        if count:
            print(f"  {name:<14} {count:>4}")


def main():
    args = build_parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if args.load:
        json_path = os.path.join(args.load, "design.json")
        if not os.path.isfile(json_path):
            sys.exit(f"ERROR: no design.json in '{args.load}'")

        print(f"Loading {json_path} …")
        data = load_json(json_path)
        gen, spec, original_side_px = reconstruct_generator(data)

        if args.symmetry is not None:
            gen.symmetry_mode = args.symmetry

        if args.width or args.height:
            width_mm = parse_mm(args.width) if args.width else spec.width_mm
            height_mm = parse_mm(args.height) if args.height else spec.height_mm
            spec = DesignSpec(width_mm, height_mm, spec.side_mm)
            print(
                f"New canvas: {width_mm:.0f} × {height_mm:.0f} mm "
                f"(side kept at {spec.side_mm:.1f} mm)"
            )

        if args.do_continue:
            print(f"Continuing ({args.iterations} iterations) …")
            gen.width = mm_to_px(spec.width_mm)
            gen.height = mm_to_px(spec.height_mm)
            gen.sideLength = original_side_px
            gen.iterations = args.iterations
            gen.retries = args.retries

            t0 = time.time()
            gen.runGen3()
            print(f"Done in {time.time() - t0:.1f}s")

    else:
        missing = [
            name
            for name, value in [("--width", args.width), ("--height", args.height), ("--side", args.side)]
            if value is None
        ]
        if missing:
            sys.exit(f"ERROR: new designs require {', '.join(missing)}")

        spec = DesignSpec(parse_mm(args.width), parse_mm(args.height), parse_mm(args.side))

        print(f"Canvas   : {spec.width_mm:.0f} × {spec.height_mm:.0f} mm")
        print(f"Tile     : {spec.side_mm:.1f} mm side length")
        print(f"Grid     : ~{spec.tiles_wide:.1f} × {spec.tiles_tall:.1f} tiles")
        print(f"Symmetry : {(args.symmetry or 'none').upper()}")
        print(f"Generating ({args.iterations} iterations) …")

        gen = GirihGenerator(
            width=spec.width_px,
            height=spec.height_px,
            sideLength=spec.side_px,
            iterations=args.iterations,
            retries=args.retries,
            symmetry_mode=(args.symmetry or "none"),
        )

        t0 = time.time()
        gen.runGen3()
        print(f"Done in {time.time() - t0:.1f}s")

    if args.output:
        folder = args.output
    elif args.load and not args.do_continue:
        folder = args.load
    else:
        folder = next_design_folder(".")

    os.makedirs(folder, exist_ok=True)

    print_stats(gen)

    print(f"\nWriting {folder}/ …")
    run_exports(gen, spec, folder)
    print(
        "  ✓  design.json\n"
        "  ✓  design.dxf\n"
        "  ✓  design.svg\n"
        "  ✓  design.png\n"
        "  ✓  design_info.txt\n"
        "Done."
    )


if __name__ == "__main__":
    main()
