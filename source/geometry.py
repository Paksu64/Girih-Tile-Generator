import turtle
import math

# --------------------------------------------------------------
# Utilities
# --------------------------------------------------------------

def rotateVector(vx, vy, angleDeg):
    # Rotate a 2D vector by angleDeg degrees.
    radians = math.radians(angleDeg)
    cosA, sinA = math.cos(radians), math.sin(radians)
    rotatedX = cosA * vx - sinA * vy
    rotatedY = sinA * vx + cosA * vy
    return rotatedX, rotatedY


def signedArea(vertices):
    # Signed polygon area used to detect orientation (CW/CCW).
    area = 0.0
    for i in range(len(vertices)):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % len(vertices)]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def getEdgeAngle(tile, edgeIndex):
    # Return world angle of the tile's edgeIndex in degrees.
    pointA = tile.worldVertices[edgeIndex]
    pointB = tile.worldVertices[(edgeIndex + 1) % len(tile.worldVertices)]
    dx = pointB[0] - pointA[0]
    dy = pointB[1] - pointA[1]
    return math.degrees(math.atan2(dy, dx))


def rotateEdgeToAngle(tile, edgeIndex, targetAngleDeg):
    # Rotate entire tile so chosen edge points toward targetAngleDeg.
    currentAngle = getEdgeAngle(tile, edgeIndex)
    deltaAngle = targetAngleDeg - currentAngle
    tile.angleDeg += deltaAngle
    tile.updateWorldVertices()


def moveEdgeStartTo(tile, edgeIndex, targetX, targetY):
    # Tile translation so that edgeIndex's start vertex is (targetX, targetY).
    pointA = tile.worldVertices[edgeIndex]
    dx = targetX - pointA[0]
    dy = targetY - pointA[1]
    tile.posX += dx
    tile.posY += dy
    tile.updateWorldVertices()

class TileBase:
    """
    Base class for all Girih tiles.
    Handles:
      - localVertices
      - world transformation (translation + rotation)
      - freeEdges
      - turtle rendering
    """

    def __init__(self, tileType, localVertices, x=0, y=0, angle=0):
        self.tileType = tileType
        self.localVertices = localVertices
        self.posX = x
        self.posY = y
        self.angleDeg = angle

        # Computed world vertices
        self.worldVertices = []
        self.updateWorldVertices()

        vertexCount = len(self.localVertices)
        self.freeEdges = [(i, (i + 1) % vertexCount) for i in range(vertexCount)]

    def updateWorldVertices(self):
        # Compute world coordinates from local vertices.
        newVertices = []
        rad = math.radians(self.angleDeg)
        cosA, sinA = math.cos(rad), math.sin(rad)

        for (localX, localY) in self.localVertices:
            rotatedX = cosA * localX - sinA * localY
            rotatedY = sinA * localX + cosA * localY
            newVertices.append((self.posX + rotatedX, self.posY + rotatedY))

        self.worldVertices = newVertices

    def getEdgeMidpoint(self, edgeIndex):
        # Return midpoint of world edge edgeIndex.
        A = self.worldVertices[edgeIndex]
        B = self.worldVertices[(edgeIndex + 1) % len(self.worldVertices)]
        midX = (A[0] + B[0]) / 2
        midY = (A[1] + B[1]) / 2
        return (midX, midY)

    def drawGirihStar(self, step=2, color="blue", penSize=2):
        turtle.color(color)
        turtle.pensize(penSize)

        vertexCount = len(self.worldVertices)

        # Edge midpoints
        midpoints = [self.getEdgeMidpoint(i) for i in range(vertexCount)]

        # Draw path
        turtle.penup()
        turtle.goto(midpoints[0])
        turtle.pendown()

        # Number of jumps
        jumps = vertexCount // math.gcd(vertexCount, step)
        index = 0

        for _ in range(jumps):
            index = (index + step) % vertexCount
            turtle.goto(midpoints[index])

        # Close
        turtle.goto(midpoints[0])

    def draw(self, color="white", animate=False, delayMs=2):
        vertices = self.worldVertices

        turtle.color(color)
        turtle.penup()
        turtle.goto(vertices[0])
        turtle.pendown()

        if animate:
            for v in vertices[1:]:
                turtle.goto(v)
                turtle.update()
                turtle.delay(delayMs)
            turtle.goto(vertices[0])
            turtle.update()
        else:
            for v in vertices[1:]:
                turtle.goto(v)
            turtle.goto(vertices[0])

    # ----------------------------------------------------------
    def getEdgeVector(self, edgeIndex):
        A = self.worldVertices[edgeIndex]
        B = self.worldVertices[(edgeIndex + 1) % len(self.worldVertices)]
        return B[0] - A[0], B[1] - A[1]

    def attachToEdge(self, targetTile, myEdgeIndex, targetEdgeIndex):
        desiredDirection = getEdgeAngle(targetTile, targetEdgeIndex)
        rotateEdgeToAngle(self, myEdgeIndex, desiredDirection)

        attachPoint = targetTile.worldVertices[targetEdgeIndex]
        moveEdgeStartTo(self, myEdgeIndex, attachPoint[0], attachPoint[1])