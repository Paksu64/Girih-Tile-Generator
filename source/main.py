import turtle
from tiles import (
    PolygonTile,
    RhombusTile,
    ElongatedHexTile,
    BowtieTile
)

def setupScreen():
    screen = turtle.Screen()
    screen.setup(900, 900)
    screen.bgcolor("black")
    screen.title("Girih Tile Generator Demo")
    screen.tracer(0)
    turtle.speed(0)
    turtle.hideturtle()
    turtle.pensize(2)
    return screen

def drawDemoTiles():
    dec = PolygonTile(10, radius=200, x=0, y=0, angle=0)
    dec.draw(color="#98dde9")

    rhombus = RhombusTile(dec, edgeIndex=0)
    hexTile = ElongatedHexTile(dec, edgeIndex=4)
    bowtie = BowtieTile(dec, edgeIndex=7)

    rhombus.attachToEdge(dec, myEdgeIndex=0, targetEdgeIndex=0)
    hexTile.attachToEdge(dec, myEdgeIndex=0, targetEdgeIndex=4)
    bowtie.attachToEdge(dec, myEdgeIndex=0, targetEdgeIndex=7)

    rhombus.draw(color="#989ce9")
    hexTile.draw(color="#bde998")
    bowtie.draw(color="#e99899")

def main():
    setupScreen()
    drawDemoTiles()
    turtle.update()
    turtle.done()

if __name__ == "__main__":
    main()