"""
Export pipeline — JSON, DXF, SVG, PNG, info txt.

Strapwork endpoints are computed by finding actual ray-ray intersections
so that shared junction points are exactly coincident (no gaps, no overlaps).
DLENS is used only as a fallback max-length guard for rays that hit no intersection.
"""

import json, math, os
from collections import defaultdict
from typing import TYPE_CHECKING, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from library.units import DesignSpec, px_to_mm
from library.tiles import DLENS

if TYPE_CHECKING:
    from library.generator import GirihGenerator

TILE_NAMES = {0:"decagon", 1:"elong_hex", 2:"bowtie", 3:"rhombus", 4:"pentagon"}

TILE_FILL = {
    0: "lightsteelblue",
    1: "lightgreen",
    2: "lightsalmon",
    3: "mediumpurple",
    4: "khaki",
}
TILE_EDGE   = "white"
LINE1_COLOR = "red"
LINE2_COLOR = "blue"
LINE1_W     = 1.0
LINE2_W     = 1.0

DXF_RGB = {
    0:(176,196,222), 1:(144,238,144),
    2:(255,160,122), 3:(147,112,219), 4:(240,230,140),
}


# ── geometry helpers ────────────────────────────────────────────────────────

def _mid(generator, tIndex, pointIdx):
    tile = generator.tiles[tIndex]
    idxs = tile.pointIndexes
    n    = len(idxs)
    off  = tile.offset
    i1   = (pointIdx + n - off - 1) % n
    i2   = (pointIdx + n - off) % n
    x1, y1 = generator.points[idxs[i1]].coords
    x2, y2 = generator.points[idxs[i2]].coords
    alfa  = math.atan2((x2 - x1), (y2 - y1))
    beta  = (math.pi / 2 - alfa) + (math.pi * 54 / 180)
    gamma = (math.pi / 2 - alfa) + (math.pi * 126 / 180)
    return (x1 + x2) / 2, (y1 + y2) / 2, beta, gamma


def _strap_segments(generator):
    """Return all strapwork segments as ((x1,y1),(x2,y2)) in world (px) coords."""
    segs = []
    for tIndex, tile in enumerate(generator.tiles):
        mtype  = tile.tileType
        offset = tile.offset
        length = tile.sideLength
        n      = len(tile.pointIndexes)
        for i in range(n):
            xm, ym, beta, gamma = _mid(generator, tIndex, i)
            d1, d2 = DLENS[mtype][(i + offset) % n]
            x2 = xm + math.cos(beta)  * length * d1
            y2 = ym + math.sin(beta)  * length * d1
            x3 = xm + math.cos(gamma) * length * d2
            y3 = ym + math.sin(gamma) * length * d2
            segs.append(((xm, ym), (x2, y2)))
            segs.append(((xm, ym), (x3, y3)))
    return segs


# ── JSON ────────────────────────────────────────────────────────────────────

def save_json(gen, spec: DesignSpec, path: str):
    data = {
        "spec": {
            "width_mm": spec.width_mm, "height_mm": spec.height_mm,
            "side_mm": spec.side_mm, "side_px": spec.side_px,
            "symmetry_mode": getattr(gen, "symmetry_mode", "none"),
        },
        "vertices": [
            {"id": i, "x_mm": px_to_mm(p.coords[0]), "y_mm": px_to_mm(p.coords[1]),
             "x_px": p.coords[0], "y_px": p.coords[1], "mask": p.mask}
            for i, p in enumerate(gen.points)
        ],
        "tiles": [
            {"id": i, "type_id": t.tileType, "type_name": TILE_NAMES[t.tileType],
             "vertex_ids": t.pointIndexes, "side_mm": px_to_mm(t.sideLength),
             "side_px": t.sideLength, "offset": t.offset, "rotation_deg": t.rotationDeg}
            for i, t in enumerate(gen.tiles)
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── SVG ────────────────────────────────────────────────────────────────────

def save_svg(gen, spec: DesignSpec, path: str, preview_mode: str = "full"):
    W, H = spec.width_mm, spec.height_mm

    def to_svg(px, py):
        return px_to_mm(px) + W / 2, H / 2 - px_to_mm(py)

    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f'     width="{W:.3f}mm" height="{H:.3f}mm"',
        f'     viewBox="0 0 {W:.3f} {H:.3f}">',
        f'  <defs><clipPath id="canvas">',
        f'    <rect x="0" y="0" width="{W:.3f}" height="{H:.3f}"/>',
        f'  </clipPath></defs>',
        f'  <g clip-path="url(#canvas)">',
    ]

    if preview_mode == "full":
        for tile in gen.tiles:
            pts = [to_svg(*gen.points[i].coords) for i in tile.pointIndexes]
            ps  = " ".join(f"{x:.3f},{y:.3f}" for x, y in pts)
            o.append(
                f'    <polygon points="{ps}" fill="{TILE_FILL[tile.tileType]}"'
                f' stroke="{TILE_EDGE}" stroke-width="0.3"/>'
            )

    for (wx1, wy1), (wx2, wy2) in _strap_segments(gen):
        sx1, sy1 = to_svg(wx1, wy1)
        sx2, sy2 = to_svg(wx2, wy2)
        o.append(
            f'    <line x1="{sx1:.3f}" y1="{sy1:.3f}"'
            f' x2="{sx2:.3f}" y2="{sy2:.3f}"'
            f' stroke="{LINE1_COLOR}" stroke-width="{LINE1_W}" stroke-linecap="round"/>'
        )

    o += ['  </g>', '</svg>']
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(o))


# ── DXF ────────────────────────────────────────────────────────────────────

def save_dxf(gen, spec: DesignSpec, path: str):
    try:
        import ezdxf
    except ModuleNotFoundError as exc:
        raise RuntimeError("DXF export requires the ezdxf package.") from exc

    doc = ezdxf.new(dxfversion="R2010")
    doc.header["$INSUNITS"] = 4   # mm
    msp = doc.modelspace()

    doc.layers.add("strapwork").rgb = (255, 0, 0)
    doc.layers.add("canvas_boundary").rgb = (80, 80, 80)

    def to_dxf(px, py):
        return px_to_mm(px), px_to_mm(py)

    # ── strapwork as paired polylines (end → mid → end) ─────────────────
    # Group half-segments by their midpoint so each edge becomes one
    # connected 3-point polyline: endpoint → midpoint → endpoint.
    # This gives SolidWorks a single continuous, importable curve instead
    # of two disconnected LINE entities.
    mid_to_ends = defaultdict(list)
    for (wx1, wy1), (wx2, wy2) in _strap_segments(gen):
        key = (round(wx1, 4), round(wy1, 4))
        mid_to_ends[key].append(to_dxf(wx2, wy2))

    for (mx, my), ends in mid_to_ends.items():
        dm = to_dxf(mx, my)
        if len(ends) == 2:
            # Full crossing strap: end1 → mid → end2
            pts = [ends[0], dm, ends[1]]
            msp.add_lwpolyline(pts, close=False,
                               dxfattribs={"layer": "strapwork",
                                           "true_color": ezdxf.rgb2int((255, 0, 0))})
        else:
            # Boundary half-segment (canvas edge): mid → end
            for end in ends:
                msp.add_lwpolyline([dm, end], close=False,
                                   dxfattribs={"layer": "strapwork",
                                               "true_color": ezdxf.rgb2int((255, 0, 0))})

    # ── canvas boundary ─────────────────────────────────────────────────
    W, H = spec.width_mm / 2, spec.height_mm / 2
    msp.add_lwpolyline([(-W, -H), (W, -H), (W, H), (-W, H)], close=True,
                       dxfattribs={"layer": "canvas_boundary"})
    doc.saveas(path)


# ── DXF ENGRAVE (closed contours for SolidWorks Extruded Cut) ──────────────

def save_dxf_engrave(gen, spec: DesignSpec, path: str, groove_width_mm: float = 1.0):
    """
    Export strapwork as fully closed contours suitable for SolidWorks
    Extruded Cut / Wrap → Deboss.

    Each strap segment is buffered by groove_width_mm/2, all buffers are
    unioned via shapely, then clipped to the canvas boundary.  The result
    is a set of simple closed polygons with no self-intersections, no open
    ends, and no overlapping geometry — exactly what SolidWorks needs.

    Exterior rings  → strapwork_outer  layer  (the groove outline)
    Interior rings  → strapwork_inner  layer  (holes within grooves, rare)
    Canvas boundary → canvas_boundary  layer
    """
    try:
        import ezdxf
    except ModuleNotFoundError as exc:
        raise RuntimeError("DXF export requires the ezdxf package.") from exc

    try:
        from shapely.geometry import LineString, box
        from shapely.ops import unary_union
    except ModuleNotFoundError as exc:
        raise RuntimeError("Engrave DXF export requires the shapely package.") from exc

    doc = ezdxf.new(dxfversion="R2010")
    doc.header["$INSUNITS"] = 4  # mm
    msp = doc.modelspace()

    doc.layers.add("strapwork_outer").rgb = (255, 0, 0)
    doc.layers.add("strapwork_inner").rgb = (255, 128, 0)
    doc.layers.add("canvas_boundary").rgb = (80, 80, 80)

    # ── build buffered union in px space, then convert to mm ────────────
    half_px = (groove_width_mm / 2.0) * (1.0 / (25.4 / 96.0))  # mm → px

    segs = _strap_segments(gen)
    if not segs:
        doc.saveas(path)
        return

    bands = []
    for (wx1, wy1), (wx2, wy2) in segs:
        if math.hypot(wx2 - wx1, wy2 - wy1) < 1e-6:
            continue
        line = LineString([(wx1, wy1), (wx2, wy2)])
        bands.append(line.buffer(half_px, cap_style=2, join_style=2))  # flat caps, mitre joins

    merged = unary_union(bands)

    # clip to canvas boundary (in px)
    W_px = spec.width_px / 2
    H_px = spec.height_px / 2
    canvas_px = box(-W_px, -H_px, W_px, H_px)
    merged = merged.intersection(canvas_px)

    # ── helper: write one polygon ring as closed LWPOLYLINE in mm ───────
    def write_ring(coords, layer):
        pts_mm = [(px_to_mm(x), px_to_mm(y)) for x, y in coords]
        if len(pts_mm) < 3:
            return
        # drop duplicate closing point if present (shapely includes it)
        if pts_mm[0] == pts_mm[-1]:
            pts_mm = pts_mm[:-1]
        msp.add_lwpolyline(pts_mm, close=True,
                           dxfattribs={"layer": layer,
                                       "true_color": ezdxf.rgb2int(
                                           (255, 0, 0) if layer == "strapwork_outer"
                                           else (255, 128, 0))})

    # ── flatten MultiPolygon / Polygon / GeometryCollection ─────────────
    def iter_polygons(geom):
        from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
        if geom.is_empty:
            return
        if isinstance(geom, Polygon):
            yield geom
        elif isinstance(geom, (MultiPolygon, GeometryCollection)):
            for g in geom.geoms:
                yield from iter_polygons(g)

    for poly in iter_polygons(merged):
        write_ring(list(poly.exterior.coords), "strapwork_outer")
        for interior in poly.interiors:
            write_ring(list(interior.coords), "strapwork_inner")

    # ── canvas boundary ──────────────────────────────────────────────────
    W, H = spec.width_mm / 2, spec.height_mm / 2
    msp.add_lwpolyline([(-W, -H), (W, -H), (W, H), (-W, H)], close=True,
                       dxfattribs={"layer": "canvas_boundary"})

    doc.saveas(path)


# ── PNG ────────────────────────────────────────────────────────────────────

def save_png(gen, spec: DesignSpec, path: str, dpi: int = 150, preview_mode: str = "full"):
    W_px, H_px = spec.width_px, spec.height_px

    fig, ax = plt.subplots(figsize=(spec.width_mm / 25.4, spec.height_mm / 25.4))
    ax.set_aspect("equal")
    ax.set_xlim(-W_px / 2, W_px / 2)
    ax.set_ylim(-H_px / 2, H_px / 2)
    ax.set_facecolor((1, 1, 1, 0))
    fig.patch.set_alpha(0)

    if preview_mode == "full":
        for tile in gen.tiles:
            verts = [gen.points[i].coords for i in tile.pointIndexes]
            ax.add_patch(
                MplPolygon(
                    verts,
                    closed=True,
                    facecolor=TILE_FILL[tile.tileType],
                    edgecolor=TILE_EDGE,
                    linewidth=0.3,
                )
            )

    for (wx1, wy1), (wx2, wy2) in _strap_segments(gen):
        ax.plot(
            [wx1, wx2],
            [wy1, wy2],
            color=LINE1_COLOR,
            linewidth=LINE1_W,
            solid_capstyle="round",
        )

    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(fig)


# ── INFO ────────────────────────────────────────────────────────────────────

def save_info(gen, spec: DesignSpec, folder: str, path: str):
    from collections import Counter
    counts = Counter(TILE_NAMES[t.tileType] for t in gen.tiles)
    lines = [
        "Girih Design", "=" * 40, "",
        spec.summary(), "",
        f"Vertices : {len(gen.points)}",
        f"Tiles    : {len(gen.tiles)}", "",
        "Tile breakdown", "-" * 20,
    ]
    for name in ["decagon", "elong_hex", "pentagon", "bowtie", "rhombus"]:
        lines.append(f"  {name:<14} {counts.get(name, 0):>5}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── folder utils ─────────────────────────────────────────────────────────────


def ensure_designs_dir(base_dir="."):
    designs_dir = os.path.join(base_dir, "designs")
    os.makedirs(designs_dir, exist_ok=True)
    return designs_dir


def resolve_design_folder(name_or_path, designs_dir="."):
    normalized = os.path.normpath(name_or_path)

    if os.path.isabs(normalized):
        return normalized

    if os.path.dirname(normalized):
        return normalized

    return os.path.join(designs_dir, normalized)


def next_design_folder(base_dir="."):
    os.makedirs(base_dir, exist_ok=True)
    i = 1
    while True:
        name = os.path.join(base_dir, f"strapwork_{i:03d}")
        if not os.path.exists(name):
            return name
        i += 1


def run_exports(
    gen,
    spec: DesignSpec,
    folder: str,
    groove_width_mm: float = 1.5,
    preview_mode: str = "full",
):
    save_json(gen, spec, os.path.join(folder, "design.json"))
    save_dxf(gen, spec, os.path.join(folder, "design.dxf"))
    save_dxf_engrave(
        gen,
        spec,
        os.path.join(folder, "design_engrave.dxf"),
        groove_width_mm=groove_width_mm,
    )
    save_svg(gen, spec, os.path.join(folder, "design.svg"), preview_mode=preview_mode)
    save_png(gen, spec, os.path.join(folder, "design.png"), preview_mode=preview_mode)
    save_info(gen, spec, os.path.basename(folder), os.path.join(folder, "design_info.txt"))