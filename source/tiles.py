import math
import turtle
from geometry import TileBase, rotateVector, signedArea


class PolygonTile(TileBase):
    def __init__(self, edgeCount, radius=1.0, x=0, y=0, angle=0):
        self.vertexCount = edgeCount
        localVerts = [
            (
                radius * math.cos(math.radians(i * (360 / edgeCount))),
                radius * math.sin(math.radians(i * (360 / edgeCount)))
            )
            for i in range(edgeCount)
        ]

        super().__init__("polygon", localVerts, x, y, angle)

    @property
    def vertices(self):
        return self.localVertices

    def edge(self, i):
        return self.vertices[i], self.vertices[(i + 1) % self.vertexCount]

    def edgeUnitVectors(self, i):
        (Ax, Ay), (Bx, By) = self.edge(i)
        dx, dy = Bx - Ax, By - Ay
        length = math.hypot(dx, dy)
        unitTangent = (dx / length, dy / length)
        unitNormal = (-unitTangent[1], unitTangent[0])
        return unitTangent, unitNormal, length


class RhombusTile(TileBase):
    def __init__(self, basePolygon, edgeIndex,
                 acuteAngleDeg=72, acuteAtA=True, isOutward=True,
                 x=0, y=0, angle=0):

        vertexA = basePolygon.vertices[edgeIndex]
        nextIndex = (edgeIndex + 1) % basePolygon.vertexCount
        vertexB = basePolygon.vertices[nextIndex]

        unitTangent, _, edgeLength = basePolygon.edgeUnitVectors(edgeIndex)

        if acuteAtA:
            turnSequence = [acuteAngleDeg, 180 - acuteAngleDeg, acuteAngleDeg]
        else:
            turnSequence = [180 - acuteAngleDeg, acuteAngleDeg, 180 - acuteAngleDeg]

        isCounterClockwise = signedArea(basePolygon.vertices) > 0
        turnSign = -1 if isOutward and isCounterClockwise else 1

        vertices = [vertexA, vertexB]
        directionX, directionY = unitTangent
        currentX, currentY = vertexB

        for turnAngleDeg in turnSequence[:2]:
            directionX, directionY = rotateVector(
                directionX, directionY, turnSign * turnAngleDeg)
            currentX += edgeLength * directionX
            currentY += edgeLength * directionY
            vertices.append((currentX, currentY))

        super().__init__("rhombus", vertices, x, y, angle)



class ElongatedHexTile(TileBase):
    def __init__(self, basePolygon, edgeIndex, acuteAtA=True, isOutward=True,
                 x=0, y=0, angle=0):

        vertexA, vertexB = basePolygon.edge(edgeIndex)
        unitTangent, _, edgeLength = basePolygon.edgeUnitVectors(edgeIndex)

        if acuteAtA:
            turnSequence = [108, 36, 108, 36, 36, 36]
        else:
            turnSequence = [36, 36, 36, 108, 36, 108]

        isCounterClockwise = signedArea(basePolygon.vertices) > 0
        turnSign = -1 if isOutward and isCounterClockwise else 1

        vertices = [vertexA, vertexB]
        directionX, directionY = unitTangent
        currentX, currentY = vertexB

        for turnAngleDeg in turnSequence[1:5]:
            directionX, directionY = rotateVector(
                directionX, directionY, turnSign * turnAngleDeg)
            currentX += edgeLength * directionX
            currentY += edgeLength * directionY
            vertices.append((currentX, currentY))

        super().__init__("elongatedHexagon", vertices, x, y, angle)


class BowtieTile(TileBase):
    def __init__(self, basePolygon, edgeIndex, isOutward=True,
                 x=0, y=0, angle=0):

        vertexA, vertexB = basePolygon.edge(edgeIndex)
        unitTangent, _, edgeLength = basePolygon.edgeUnitVectors(edgeIndex)

        turnSequence = [108, 324, 108, 108, 324]

        isCounterClockwise = signedArea(basePolygon.vertices) > 0
        turnSign = -1 if isOutward and isCounterClockwise else 1

        vertices = [vertexA, vertexB]
        directionX, directionY = unitTangent
        currentX, currentY = vertexB

        for turnAngleDeg in turnSequence:
            directionX, directionY = rotateVector(
                directionX, directionY, turnSign * turnAngleDeg)
            currentX += edgeLength * directionX
            currentY += edgeLength * directionY
            vertices.append((currentX, currentY))

        super().__init__("bowtie", vertices, x, y, angle)
