import math
import turtle

def rotateVector(vx, vy, angleDeg):
    rad = math.radians(angleDeg)
    c, s = math.cos(rad), math.sin(rad)
    return c * vx - s * vy, s * vx + c * vy


# -----------------------
# Tile base class
# -----------------------

class TileBase:
    def __init__(self, tileType, localVertices, x=0.0, y=0.0, angleDeg=0.0):
        self.tileType = tileType
        self.localVertices = [tuple(v) for v in localVertices]
        self.posX = x
        self.posY = y
        self.angleDeg = angleDeg
        self.updateWorldVertices()

    def updateWorldVertices(self):
        rad = math.radians(self.angleDeg)
        cosA, sinA = math.cos(rad), math.sin(rad)
        world = []
        for lx, ly in self.localVertices:
            wx = self.posX + (cosA * lx - sinA * ly)
            wy = self.posY + (sinA * lx + cosA * ly)
            world.append((wx, wy))
        self.worldVertices = world

    def rotate(self, deltaDeg):
        self.angleDeg += deltaDeg
        self.updateWorldVertices()

    def translate(self, dx, dy):
        self.posX += dx
        self.posY += dy
        self.updateWorldVertices()

    def edge(self, edgeIndex):
        v = self.worldVertices
        return v[edgeIndex], v[(edgeIndex + 1) % len(v)]

    def getEdgeAngle(self, edgeIndex):
        A, B = self.edge(edgeIndex)
        dx = B[0] - A[0]
        dy = B[1] - A[1]
        return math.degrees(math.atan2(dy, dx))

    def clone(self):
        newTile = self.__class__.__new__(self.__class__)
        newTile.tileType = self.tileType
        newTile.localVertices = [tuple(v) for v in self.localVertices]
        newTile.posX = self.posX
        newTile.posY = self.posY
        newTile.angleDeg = self.angleDeg
        newTile.updateWorldVertices()
        return newTile

    # --- attach logic (CW/CCW based) ---

    def attachToEdge(self, targetTile, myEdgeIndex, targetEdgeIndex, outward=True):
        """
        Attach this tile's edge `myEdgeIndex` to `targetTile`'s edge `targetEdgeIndex`.

        - The two edges will coincide with opposite directions (shared border).
        - If outward=True  -> this tile ends up on the opposite side of the edge
                              compared to the target's centroid.
        - If outward=False -> this tile ends up on the same side as the target's centroid.
        """

        # Target Edge
        tgtA, tgtB = targetTile.edge(targetEdgeIndex)
        bx = tgtB[0] - tgtA[0]
        by = tgtB[1] - tgtA[1]
        base_len = math.hypot(bx, by)
        if base_len == 0:
            return

        ux = bx / base_len
        uy = by / base_len  # unit direction along target edge A > B

        # Aligning my edge to target edge
        myA, myB = self.edge(myEdgeIndex)
        sx = myB[0] - myA[0]
        sy = myB[1] - myA[1]

        self_angle  = math.atan2(sy, sx)
        target_angle = math.atan2(-uy, -ux)   # opposite to target edge

        rot_deg = math.degrees(target_angle - self_angle)
        self.rotate(rot_deg)

        # Snapping myA to tgtA
        myA2, _ = self.edge(myEdgeIndex)
        dx = tgtA[0] - myA2[0]
        dy = tgtA[1] - myA2[1]
        self.translate(dx, dy)

        # inside check
        def centroid(tile):
            verts = tile.worldVertices
            n = len(verts)
            return (
                sum(v[0] for v in verts) / n,
                sum(v[1] for v in verts) / n,
            )

        ct_x, ct_y = centroid(targetTile)
        cs_x, cs_y = centroid(self)

        # Vectors from edge start to centroids
        vtx = ct_x - tgtA[0]
        vty = ct_y - tgtA[1]
        vsx = cs_x - tgtA[0]
        vsy = cs_y - tgtA[1]

        # cross(u, v) = ux*vy - uy*vx  → sign tells which side of the edge
        side_target = ux * vty - uy * vtx
        side_self   = ux * vsy - uy * vsx

        if outward:
            ok = side_target * side_self < 0
        else:
            ok = side_target * side_self > 0

        # Flip if needed
        if not ok:
            self.rotate(180)

        # Snap again to minimize error
        myA2, myB2 = self.edge(myEdgeIndex)
        my_mid = ((myA2[0] + myB2[0]) * 0.5,
                (myA2[1] + myB2[1]) * 0.5)

        tgt_mid = ((tgtA[0] + tgtB[0]) * 0.5,
                (tgtA[1] + tgtB[1]) * 0.5)

        dx = tgt_mid[0] - my_mid[0]
        dy = tgt_mid[1] - my_mid[1]

        self.translate(dx, dy)


    def draw(self, color="white"):
        verts = self.worldVertices
        turtle.color(color)
        turtle.penup()
        turtle.goto(verts[0])
        turtle.pendown()
        for vx, vy in verts[1:]:
            turtle.goto(vx, vy)
        turtle.goto(verts[0])
