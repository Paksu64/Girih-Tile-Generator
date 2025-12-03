import math
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from tiles import DLENS

STYLE = {
    "fillColors": ["lightsteelblue","lightgreen","lightsalmon",
                   "mediumpurple","khaki"],
    "edgeColors": ["white"]*5,
    "edgeWidth": 2.0,
    "lineColor1": "gray",
    "lineWidth1": 2.0,
    "lineColor2": "black",
    "lineWidth2": 0.8,
}


class RendererMatplotlib:
    def render(self, generator, style=STYLE, drawPoints=False, figsize=(8,8)):
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
