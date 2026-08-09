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
  python main.py --load strapwork_001 --continue --iterations 300

Reconstruct a design from a photo of tile edges (experimental -- see
experiments/ for status, real-photo results aren't solid yet):
  python main.py --reconstruct photo.jpg --side 25mm --output recovered_001

Repeat an existing design across a 2D grid, attaching copies at shared
edges (side-to-side and top-to-top) -- no photo/detection involved, this
is a separate tool, not a --reconstruct mode:
  python pattern_tiling.py --load strapwork_001 --nx 3 --ny 2 --output big_panel
"""

import argparse
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))

from library.export import TILE_NAMES, load_json, next_design_folder, run_exports
from library.generator import GirihGenerator
from library.tiles import Point, Tile
from library.units import DesignSpec, mm_to_px, parse_mm


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
    parser.add_argument(
        "--reconstruct", metavar="IMAGE",
        help="Recover a design.json from a photo/scan of girih tile edges instead of generating one.",
    )
    parser.add_argument(
        "--corners", metavar="X1,Y1,X2,Y2,X3,Y3,X4,Y4",
        help="Manual 4-point rectification for --reconstruct, if automatic panel-quad "
             "detection fails or the image isn't a clean framed panel.",
    )
    parser.add_argument(
        "--input-mode",
        choices=["tile-edges", "periodic", "rotational"],
        default="tile-edges",
        help="What --reconstruct's image shows: tile-edges = physical tile/grout "
             "boundaries; periodic = a repeating strip/panel pattern -- detects "
             "the repeat period, reconstructs one clean unit, and attaches copies "
             "at shared seams across the full image; rotational = a medallion/"
             "rosette centered pattern -- detects the rotation center and fold "
             "order (5 or 10), and fills sparse/damaged sectors by rotating a "
             "well-reconstructed sector into place. Default: tile-edges. (A "
             "strapwork mode was tried and parked in "
             "experiments/experimental_strapwork/ -- its math is validated "
             "but real-photo recall isn't there yet.)",
    )
    parser.add_argument(
        "--copies", type=int, default=None,
        help="For --input-mode periodic: how many repeats to replicate (default: as "
             "many as fit the detected period).",
    )
    parser.add_argument(
        "--damage-frac", type=float, default=0.5,
        help="For --input-mode rotational: a sector with fewer than this fraction "
             "of the best sector's vertex count is treated as damaged/empty and "
             "filled by rotation (default: 0.5).",
    )
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
    parser.add_argument(
        "--groove", type=float, default=1.5, metavar="MM",
        help="Groove width in mm for the engrave DXF (default 1.5mm).",
    )
    parser.add_argument(
        "--preview",
        choices=["full", "strapwork"],
        default="full",
        help="PNG/SVG preview mode: full = polygons + strapwork, strapwork = strapwork only.",
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


def resolve_designs_dir():
    designs_dir = os.path.join(os.path.dirname(__file__), "designs")
    os.makedirs(designs_dir, exist_ok=True)
    return designs_dir


def resolve_load_folder(load_value, designs_dir):
    if os.path.isabs(load_value):
        return load_value

    if os.path.isdir(load_value):
        return load_value

    candidate = os.path.join(designs_dir, load_value)
    if os.path.isdir(candidate):
        return candidate

    return candidate


def resolve_output_folder(output_value, designs_dir):
    if output_value is None:
        return next_design_folder(designs_dir)

    if os.path.isabs(output_value):
        return output_value

    return os.path.join(designs_dir, output_value)


def parse_corners(value):
    if value is None:
        return None
    nums = [float(v) for v in value.split(",")]
    if len(nums) != 8:
        sys.exit("ERROR: --corners needs exactly 8 numbers: X1,Y1,X2,Y2,X3,Y3,X4,Y4")
    return list(zip(nums[0::2], nums[1::2]))


def run_reconstruct(args, designs_dir):
    side_mm = parse_mm(args.side) if args.side else 25.0
    folder = resolve_output_folder(args.output, designs_dir) if args.output else next_design_folder(designs_dir)
    os.makedirs(folder, exist_ok=True)

    print(f"Reconstructing from {args.reconstruct} (input mode: {args.input_mode}) …")

    if args.input_mode == "periodic":
        from experiments import periodic
        result = periodic.run(
            args.reconstruct, side_mm=side_mm, folder=folder,
            groove_width_mm=args.groove, n_copies=args.copies,
        )
        print(f"Detected period : {result['period_px'][0]:.1f}px  (used {result['n_copies']} copies, "
              f"confidence {result['period_confidence']:.3f})")
        print(f"Unit            : {len(result['unit_tiles'])} tiles, {len(result['unit_vertices'])} vertices, "
              f"{len(result['unit_unresolved'])} unresolved")
    elif args.input_mode == "rotational":
        from experiments import rotational
        result = rotational.run(
            args.reconstruct, side_mm=side_mm, folder=folder,
            groove_width_mm=args.groove, damage_frac=args.damage_frac,
        )
        print(f"Detected center : {result['center_px']}  fold: {result['n']}  "
              f"score: {result['symmetry_score']:.3f}")
        print(f"Damaged sectors filled by rotation: {result['sector_stats']['damaged_sectors']}")
    else:
        from experiments import pipeline
        corners = parse_corners(args.corners)
        result = pipeline.run(
            args.reconstruct, corners=corners, side_mm=side_mm,
            folder=folder, groove_width_mm=args.groove, input_mode=args.input_mode,
        )
        print(f"Rectified   : {'yes' if result['rectified'] else 'no (no confident panel quad found)'}")
        print(f"Unresolved  : {len(result['unresolved'])} face(s) (damage/occlusion/mis-detection)")

    stats = result["stats"]
    print(f"Vertices (full) : {stats['vertex_count']}")
    print(f"Tiles (full)    : {stats['tile_count']}")
    print(f"Mask conflicts  : {stats['mask_conflicts']}")
    print(f"\nWriting {folder}/ …")
    print(
        "  ✓  design.json\n"
        "  ✓  design.dxf\n"
        "  ✓  design_engrave.dxf\n"
        "  ✓  design.svg\n"
        "  ✓  design.png\n"
        "  ✓  design_info.txt\n"
        "Done."
    )


def main():
    args = build_parser().parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    designs_dir = resolve_designs_dir()

    if args.reconstruct:
        run_reconstruct(args, designs_dir)
        return

    if args.load:
        load_folder = resolve_load_folder(args.load, designs_dir)
        json_path = os.path.join(load_folder, "design.json")
        if not os.path.isfile(json_path):
            sys.exit(f"ERROR: no design.json in '{load_folder}'")

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
        folder = resolve_output_folder(args.output, designs_dir)
    elif args.load and not args.do_continue:
        folder = load_folder
    else:
        folder = next_design_folder(designs_dir)

    os.makedirs(folder, exist_ok=True)

    print_stats(gen)

    print(f"\nWriting {folder}/ …")
    run_exports(
        gen,
        spec,
        folder,
        groove_width_mm=args.groove,
        preview_mode=args.preview,
    )
    print(
        "  ✓  design.json\n"
        "  ✓  design.dxf\n"
        "  ✓  design_engrave.dxf\n"
        "  ✓  design.svg\n"
        "  ✓  design.png\n"
        "  ✓  design_info.txt\n"
        "Done."
    )


if __name__ == "__main__":
    main()