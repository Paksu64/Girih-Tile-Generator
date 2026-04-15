import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import matplotlib.animation as animation
from girih.tiles import DLENS

MAIN_STYLE = {
    "fillColors": ["white"]*5,
    "edgeColors": ["white"]*5,
    "edgeWidth": 0.2,
    "lineColor1": "red",
    "lineWidth1": 0.7,
    "lineColor2": "blue",
    "lineWidth2": 0.7,
}

WIKI_STYLE = {
    "fillColors": [
    "lightsteelblue",  # Ongen
    "lightgreen",      # Altıgen
    "lightsalmon",     # Concave Altıgen
    "mediumpurple",    # Yamuk
    "khaki"            # Beşgen
    ],
    "edgeColors": ["red"]*5,
    "edgeWidth": 1.0,
    "lineColor1": "black",
    "lineWidth1": 1.5,
    "lineColor2": "grey",
    "lineWidth2": 1.5,
}

class RendererMatplotlib:
    def render(self, generator, style=WIKI_STYLE, drawPoints=False, figsize=(8,8)):
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.set_xlim(-generator.width/2, generator.width/2)
        ax.set_ylim(-generator.height/2, generator.height/2)
        ax.set_facecolor("black")

        fillColors = style["fillColors"]
        edgeColors = style["edgeColors"]

        for tile in generator.tiles:
            verts = [generator.points[i].coords for i in tile.pointIndexes]
            poly = Polygon(verts, closed=True,
                           facecolor=fillColors[tile.tileType],
                           edgecolor=edgeColors[tile.tileType],
                           linewidth=style["edgeWidth"])
            ax.add_patch(poly)

        for tIndex, tile in enumerate(generator.tiles):
            mtype = tile.tileType
            offset = tile.offset
            length = tile.sideLength
            idxs = tile.pointIndexes
            n = len(idxs)
            for i in range(n):
                xm, ym, beta, gamma = self._mid(generator, tIndex, i)
                d1, d2 = DLENS[mtype][(i+offset)%n]

                x2 = xm + math.cos(beta)*length*d1
                y2 = ym + math.sin(beta)*length*d1
                x3 = xm + math.cos(gamma)*length*d2
                y3 = ym + math.sin(gamma)*length*d2

                ax.plot([xm,x2],[ym,y2],
                        color=style["lineColor1"],
                        linewidth=style["lineWidth1"])
                ax.plot([xm,x3],[ym,y3],
                        color=style["lineColor2"],
                        linewidth=style["lineWidth2"])

        plt.axis('off')
        plt.tight_layout()
        plt.title("Girih Jenerasyonu")
        plt.show()

    def _tile_centroid(self, generator, tile):
        verts = [generator.points[i].coords for i in tile.pointIndexes]
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        return cx, cy, verts

    def _clamp01(self, x):
        return 0.0 if x < 0 else (1.0 if x > 1 else x)


    def _mid(self, generator, tIndex, pointIdx):
        tile = generator.tiles[tIndex]
        idxs = tile.pointIndexes
        n = len(idxs)
        off = tile.offset

        i1 = (pointIdx + n - off - 1) % n
        i2 = (pointIdx + n - off) % n

        x1,y1 = generator.points[idxs[i1]].coords
        x2,y2 = generator.points[idxs[i2]].coords

        alfa = math.atan2((x2-x1),(y2-y1))
        beta = (math.pi/2 - alfa) + (math.pi*54/180)
        gamma = (math.pi/2 - alfa) + (math.pi*126/180)

        return ( (x1+x2)/2, (y1+y2)/2, beta, gamma )
    
    def collect_edges(self, generator):
        edges = []

        for tIndex, tile in enumerate(generator.tiles):
            mtype = tile.tileType
            offset = tile.offset
            length = tile.sideLength
            idxs = tile.pointIndexes
            n = len(idxs)

            for i in range(n):
                xm, ym, beta, gamma = self._mid(generator, tIndex, i)
                d1, d2 = DLENS[mtype][(i + offset) % n]

                # ray 1
                edges.append({
                    "mid": (xm, ym),
                    "ang": beta,
                    "L": length * d1,
                    "dist": math.hypot(xm, ym),
                    "color": "lineColor1",
                    "lw": "lineWidth1"
                })
                # ray 2
                edges.append({
                    "mid": (xm, ym),
                    "ang": gamma,
                    "L": length * d2,
                    "dist": math.hypot(xm, ym),
                    "color": "lineColor2",
                    "lw": "lineWidth2"
                })

        # inward -> outward
        edges.sort(key=lambda e: e["dist"])
        return edges
    
    def renderAnimated(self, generator, style=WIKI_STYLE, figsize=(8,8),
                                frames_per_tile=2, fill_frames=1,
                                pen_frames_per_line=1, interval=0.1, radial_order=True):
        import matplotlib.animation as animation

        fig, ax = plt.subplots(figsize=figsize)
        ax.set_aspect("equal")
        ax.set_xlim(-generator.width/2, generator.width/2)
        ax.set_ylim(-generator.height/2, generator.height/2)
        ax.set_facecolor("black")
        plt.axis("off")

        fillColors = style["fillColors"]
        edgeColors = style["edgeColors"]

        # ---------- TILE ORDER (center -> outward) ----------
        tile_infos = []
        for tIndex, tile in enumerate(generator.tiles):
            verts = [generator.points[i].coords for i in tile.pointIndexes]
            cx = sum(v[0] for v in verts) / len(verts)
            cy = sum(v[1] for v in verts) / len(verts)
            d = math.hypot(cx, cy)
            tile_infos.append((d, tIndex, tile, verts))

        tile_infos.sort(key=lambda x: x[0] if radial_order else x[1])

        # ---------- POLY FILLS (invisible first, NO outline) ----------
        polys = []
        for _, _, tile, verts in tile_infos:
            poly = Polygon(
                verts, closed=True,
                facecolor=fillColors[tile.tileType],
                edgecolor="none",
                alpha=0.0
            )
            ax.add_patch(poly)
            polys.append(poly)

        # ---------- OUTLINE LINES (tile-by-tile, edge grows from midpoint) ----------
        outline_lines, outline_meta = [], []
        for _, _, tile, verts in tile_infos:
            lines, metas = [], []
            n = len(verts)
            for i in range(n):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % n]

                mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                vx, vy = (x2 - x1), (y2 - y1)
                L = math.hypot(vx, vy)
                if L < 1e-9:
                    continue

                ux, uy = vx / L, vy / L
                half = L / 2

                ln, = ax.plot([], [],
                            color=edgeColors[tile.tileType],
                            linewidth=style["edgeWidth"],
                            solid_capstyle="round")
                lines.append(ln)
                metas.append((mx, my, ux, uy, half))
            outline_lines.append(lines)
            outline_meta.append(metas)

        # ---------- PEN SEGMENTS (NO helper fn, directly from _mid()) ----------
        segs = []
        for tIndex, tile in enumerate(generator.tiles):
            mtype = tile.tileType
            offset = tile.offset
            length = tile.sideLength
            n = len(tile.pointIndexes)

            for i in range(n):
                xm, ym, beta, gamma = self._mid(generator, tIndex, i)
                d1, d2 = DLENS[mtype][(i + offset) % n]

                # line 1 (beta)
                x1 = xm + math.cos(beta) * length * d1
                y1 = ym + math.sin(beta) * length * d1
                segs.append({
                    "a": (xm, ym),
                    "b": (x1, y1),
                    "color": style["lineColor1"],
                    "lw": style["lineWidth1"],
                    "dist": math.hypot(xm, ym)
                })

                # line 2 (gamma)
                x2 = xm + math.cos(gamma) * length * d2
                y2 = ym + math.sin(gamma) * length * d2
                segs.append({
                    "a": (xm, ym),
                    "b": (x2, y2),
                    "color": style["lineColor2"],
                    "lw": style["lineWidth2"],
                    "dist": math.hypot(xm, ym)
                })

        # center -> outward pen order (remove if you don’t want it)
        segs.sort(key=lambda s: s["dist"])

        pen_lines = []
        for s in segs:
            ln, = ax.plot([], [],
                        color=s["color"],
                        linewidth=s["lw"],
                        solid_capstyle="round")
            pen_lines.append(ln)

        # ---------- TIMELINE ----------
        tiles_n = len(tile_infos)
        block = frames_per_tile + fill_frames
        base_frames = tiles_n * block

        pen_frames = len(segs) * pen_frames_per_line
        total_frames = base_frames + pen_frames

        def ease(t):
            # fast, pen-like
            return t**0.35

        def update(frame):
            # --- PHASE A: tile outlines then fill ---
            if frame < base_frames:
                ti = min(frame // block, tiles_n - 1)
                local = frame % block

                # finalize previous tiles
                for j in range(ti):
                    for ln, meta in zip(outline_lines[j], outline_meta[j]):
                        mx, my, ux, uy, half = meta
                        ln.set_data([mx - ux*half, mx + ux*half],
                                    [my - uy*half, my + uy*half])
                    polys[j].set_alpha(1.0)

                # hide future tiles
                for j in range(ti + 1, tiles_n):
                    for ln in outline_lines[j]:
                        ln.set_data([], [])
                    polys[j].set_alpha(0.0)

                # current tile animation
                if local < frames_per_tile:
                    t = ease(local / max(frames_per_tile - 1, 1))
                    for ln, meta in zip(outline_lines[ti], outline_meta[ti]):
                        mx, my, ux, uy, half = meta
                        seglen = half * t
                        ln.set_data([mx - ux*seglen, mx + ux*seglen],
                                    [my - uy*seglen, my + uy*seglen])
                    polys[ti].set_alpha(0.0)
                else:
                    # outline full, fill fade in
                    for ln, meta in zip(outline_lines[ti], outline_meta[ti]):
                        mx, my, ux, uy, half = meta
                        ln.set_data([mx - ux*half, mx + ux*half],
                                    [my - uy*half, my + uy*half])

                    ft = (local - frames_per_tile) / max(fill_frames - 1, 1)
                    polys[ti].set_alpha(ft)

                # pen lines hidden during phase A
                for ln in pen_lines:
                    ln.set_data([], [])

            # --- PHASE B: pen draws interior lines one-by-one ---
            else:
                # lock everything: outlines + fills complete
                for j in range(tiles_n):
                    for ln, meta in zip(outline_lines[j], outline_meta[j]):
                        mx, my, ux, uy, half = meta
                        ln.set_data([mx - ux*half, mx + ux*half],
                                    [my - uy*half, my + uy*half])
                    polys[j].set_alpha(1.0)

                pf = frame - base_frames
                li = min(pf // pen_frames_per_line, len(segs) - 1)
                llocal = pf % pen_frames_per_line
                t = ease(llocal / max(pen_frames_per_line - 1, 1))

                # draw completed lines fully
                for i in range(li):
                    ax0, ay0 = segs[i]["a"]
                    bx0, by0 = segs[i]["b"]
                    pen_lines[i].set_data([ax0, bx0], [ay0, by0])

                # active line partial
                ax0, ay0 = segs[li]["a"]
                bx0, by0 = segs[li]["b"]
                cx = ax0 + (bx0 - ax0) * t
                cy = ay0 + (by0 - ay0) * t
                pen_lines[li].set_data([ax0, cx], [ay0, cy])

                # hide future lines
                for i in range(li + 1, len(segs)):
                    pen_lines[i].set_data([], [])

            # artists for blit
            artists = []
            for j in range(tiles_n):
                artists.extend(outline_lines[j])
            artists.extend(polys)
            artists.extend(pen_lines)
            return artists

        ani = animation.FuncAnimation(
            fig, update,
            frames=total_frames,
            interval=interval,
            blit=True
        )

        plt.tight_layout()
        plt.title("Girih Animasyonu")
        ani.save("girih.gif", writer="pillow", fps=120)
        plt.show()




