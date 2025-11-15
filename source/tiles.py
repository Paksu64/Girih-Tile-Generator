import math
from geometry import TileBase, rotateVector

sideLength = 60.0

class PolygonTile(TileBase):
    def __init__(self, edgeCount):
        angleStep = 360.0 / edgeCount
        radius = sideLength / (2.0 * math.sin(math.pi / edgeCount))
        verts = []
        for i in range(edgeCount):
            a = math.radians(i * angleStep)
            verts.append((radius * math.cos(a), radius * math.sin(a)))
        super().__init__("polygon", verts)

class PentagonTile(PolygonTile):
    def __init__(self):
        super().__init__(5)

class DecagonTile(PolygonTile):
    def __init__(self):
        super().__init__(10)

class RhombusTile(TileBase):
    def __init__(self, acuteDeg=72.0):
        s = sideLength
        A = (0.0, 0.0)
        B = (s,   0.0)

        turnSeq = [acuteDeg, 180.0 - acuteDeg]
        angle = 0.0
        x, y = B
        verts = [A, B]

        for t in turnSeq:
            angle += t
            x += s * math.cos(math.radians(angle))
            y += s * math.sin(math.radians(angle))
            verts.append((x, y))

        super().__init__("rhombus", verts)

class ElongatedHexTile(TileBase):
    def __init__(self):
        s = sideLength
        A = (0.0, 0.0)
        B = (s,   0.0)

        turnSeq = [36.0, 108.0, 36.0, 36.0]
        angle = 0.0
        x, y = B
        verts = [A, B]

        for t in turnSeq:
            angle += t
            x += s * math.cos(math.radians(angle))
            y += s * math.sin(math.radians(angle))
            verts.append((x, y))

        super().__init__("elongatedHex", verts)

class BowtieTile(TileBase):
    def __init__(self):
        s = sideLength
        A = (0.0, 0.0)
        B = (s,   0.0)

        turnSeq = [108.0, 324.0, 108.0, 108.0, 324.0]
        angle = 0.0
        x, y = B
        verts = [A, B]

        for t in turnSeq:
            angle += t
            x += s * math.cos(math.radians(angle))
            y += s * math.sin(math.radians(angle))
            verts.append((x, y))

        super().__init__("bowtie", verts)