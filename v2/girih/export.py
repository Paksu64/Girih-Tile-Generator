"""
Export pipeline — JSON, DXF, SVG, PNG, info txt.
Strapwork uses DLENS from tiles.py exactly as renderer.py does.
"""

import json, math, os
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from girih.units import DesignSpec, px_to_mm
from girih.tiles import DLENS

if TYPE_CHECKING:
    from girih.generator import GirihGenerator

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
LINE1_W     = 0.7
LINE2_W     = 0.7

DXF_RGB = {
    0:(176,196,222), 1:(144,238,144),
    2:(255,160,122), 3:(147,112,219), 4:(240,230,140),
}


# ── strapwork — exact copy of renderer._mid + DLENS lookup ─────────────────

def _mid(generator, tIndex, pointIdx):
    tile = generator.tiles[tIndex]
    idxs = tile.pointIndexes
    n    = len(idxs)
    off  = tile.offset
    i1   = (pointIdx + n - off - 1) % n
    i2   = (pointIdx + n - off) % n
    x1,y1 = generator.points[idxs[i1]].coords
    x2,y2 = generator.points[idxs[i2]].coords
    alfa  = math.atan2((x2-x1),(y2-y1))
    beta  = (math.pi/2 - alfa) + (math.pi*54/180)
    gamma = (math.pi/2 - alfa) + (math.pi*126/180)
    return (x1+x2)/2, (y1+y2)/2, beta, gamma


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
            segs.append(((xm,ym),(x2,y2)))
            segs.append(((xm,ym),(x3,y3)))
    return segs


# ── JSON ────────────────────────────────────────────────────────────────────

def save_json(gen, spec: DesignSpec, path: str):
    data = {
        "spec": {"width_mm":spec.width_mm,"height_mm":spec.height_mm,"side_mm":spec.side_mm,"side_px":spec.side_px,"symmetry_mode": getattr(gen, "symmetry_mode", "none")},
        "vertices": [
            {"id":i,"x_mm":px_to_mm(p.coords[0]),"y_mm":px_to_mm(p.coords[1]),
             "x_px":p.coords[0],"y_px":p.coords[1],"mask":p.mask}
            for i,p in enumerate(gen.points)
        ],
        "tiles": [
            {"id":i,"type_id":t.tileType,"type_name":TILE_NAMES[t.tileType],
             "vertex_ids":t.pointIndexes,"side_mm":px_to_mm(t.sideLength),"side_px":t.sideLength,"offset":t.offset,"rotation_deg": t.rotationDeg}
            for i,t in enumerate(gen.tiles)
        ],
    }
    with open(path,"w",encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str):
    with open(path,encoding="utf-8") as f:
        return json.load(f)


# ── SVG ────────────────────────────────────────────────────────────────────

def save_svg(gen, spec: DesignSpec, path: str):
    W, H = spec.width_mm, spec.height_mm

    def to_svg(px, py):
        return px_to_mm(px) + W/2,  H/2 - px_to_mm(py)

    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg"',
        f'     width="{W:.3f}mm" height="{H:.3f}mm"',
        f'     viewBox="0 0 {W:.3f} {H:.3f}">',
        f'  <defs><clipPath id="canvas">',
        f'    <rect x="0" y="0" width="{W:.3f}" height="{H:.3f}"/>',
        f'  </clipPath></defs>',
        f'  <rect width="100%" height="100%" fill="black"/>',
        f'  <g clip-path="url(#canvas)">',
    ]

    for tile in gen.tiles:
        pts = [to_svg(*gen.points[i].coords) for i in tile.pointIndexes]
        ps  = " ".join(f"{x:.3f},{y:.3f}" for x,y in pts)
        o.append(f'    <polygon points="{ps}" fill="{TILE_FILL[tile.tileType]}"'
                 f' stroke="white" stroke-width="0.3"/>')

    for (wx1,wy1),(wx2,wy2) in _strap_segments(gen):
        sx1,sy1 = to_svg(wx1,wy1)
        sx2,sy2 = to_svg(wx2,wy2)
        o.append(f'    <line x1="{sx1:.3f}" y1="{sy1:.3f}"'
                 f' x2="{sx2:.3f}" y2="{sy2:.3f}"'
                 f' stroke="{LINE1_COLOR}" stroke-width="0.5" stroke-linecap="round"/>')

    o += ['  </g>', '</svg>']
    with open(path,"w",encoding="utf-8") as f:
        f.write("\n".join(o))


# ── DXF ────────────────────────────────────────────────────────────────────

def save_dxf(gen, spec: DesignSpec, path: str):
    try:
        import ezdxf
    except ModuleNotFoundError as exc:
        raise RuntimeError("DXF export requires the ezdxf package.") from exc

    doc = ezdxf.new(dxfversion="R2010")
    doc.header["$INSUNITS"] = 4
    msp = doc.modelspace()

    for tid, name in TILE_NAMES.items():
        doc.layers.add(name).rgb = DXF_RGB[tid]
    doc.layers.add("strapwork").rgb = (255,0,0)
    doc.layers.add("canvas_boundary").rgb = (80,80,80)

    def to_dxf(px, py):
        return px_to_mm(px), px_to_mm(py)

    for tile in gen.tiles:
        pts = [to_dxf(*gen.points[i].coords) for i in tile.pointIndexes]
        r,g,b = DXF_RGB[tile.tileType]
        msp.add_lwpolyline(pts, close=True, dxfattribs={
            "layer": TILE_NAMES[tile.tileType],
            "true_color": ezdxf.rgb2int((r,g,b))
        })

    for (wx1,wy1),(wx2,wy2) in _strap_segments(gen):
        dx1,dy1 = to_dxf(wx1,wy1)
        dx2,dy2 = to_dxf(wx2,wy2)
        msp.add_line((dx1,dy1,0),(dx2,dy2,0),
                     dxfattribs={"layer":"strapwork","true_color":ezdxf.rgb2int((255,0,0))})

    W,H = spec.width_mm/2, spec.height_mm/2
    msp.add_lwpolyline([(-W,-H),(W,-H),(W,H),(-W,H)], close=True,
                       dxfattribs={"layer":"canvas_boundary"})
    doc.saveas(path)


# ── PNG ────────────────────────────────────────────────────────────────────

def save_png(gen, spec: DesignSpec, path: str, dpi: int = 150):
    W_px, H_px = spec.width_px, spec.height_px

    fig, ax = plt.subplots(figsize=(spec.width_mm/25.4, spec.height_mm/25.4))
    ax.set_aspect("equal")
    ax.set_xlim(-W_px/2, W_px/2)
    ax.set_ylim(-H_px/2, H_px/2)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")

    for tile in gen.tiles:
        verts = [gen.points[i].coords for i in tile.pointIndexes]
        ax.add_patch(MplPolygon(verts, closed=True,
                                facecolor=TILE_FILL[tile.tileType],
                                edgecolor="white", linewidth=0.3))

    for (wx1,wy1),(wx2,wy2) in _strap_segments(gen):
        ax.plot([wx1,wx2],[wy1,wy2], color=LINE1_COLOR,
                linewidth=LINE1_W, solid_capstyle="round")

    ax.axis("off")
    plt.tight_layout(pad=0)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="black")
    plt.close(fig)


# ── INFO ────────────────────────────────────────────────────────────────────

def save_info(gen, spec: DesignSpec, folder: str, path: str):
    from collections import Counter
    counts = Counter(TILE_NAMES[t.tileType] for t in gen.tiles)
    lines = [
        "Girih Design", "="*40, "",
        spec.summary(), "",
        f"Vertices : {len(gen.points)}",
        f"Tiles    : {len(gen.tiles)}", "",
        "Tile breakdown", "-"*20,
    ]
    for name in ["decagon","elong_hex","pentagon","bowtie","rhombus"]:
        lines.append(f"  {name:<14} {counts.get(name,0):>5}")
    with open(path,"w",encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── folder utils ─────────────────────────────────────────────────────────────

def next_design_folder(base_dir="."):
    i = 1
    while True:
        name = os.path.join(base_dir, f"girih_design_{i:03d}")
        if not os.path.exists(name):
            return name
        i += 1


def run_exports(gen, spec: DesignSpec, folder: str):
    save_json(gen, spec, os.path.join(folder, "design.json"))
    save_dxf (gen, spec, os.path.join(folder, "design.dxf"))
    save_svg (gen, spec, os.path.join(folder, "design.svg"))
    save_png (gen, spec, os.path.join(folder, "design.png"))
    save_info(gen, spec, os.path.basename(folder),
              os.path.join(folder, "design_info.txt"))