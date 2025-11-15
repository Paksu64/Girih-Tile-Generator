import turtle
from tiles import (
    DecagonTile,
    PentagonTile,
    RhombusTile,
    ElongatedHexTile,
    BowtieTile,
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
    decProto = DecagonTile()
    dec = decProto.clone()
    dec.draw(color="#98dde9")

    pentProto = PentagonTile()
    rhombusProto = RhombusTile()
    hexProto = ElongatedHexTile()
    bowtieProto = BowtieTile()

    pent = pentProto.clone()
    rhom = rhombusProto.clone()
    hext = hexProto.clone()
    bow  = bowtieProto.clone()

    pent.attachToEdge(dec, myEdgeIndex=0, targetEdgeIndex=1, outward=True)
    rhom.attachToEdge(dec, myEdgeIndex=0, targetEdgeIndex=5, outward=True)
    hext.attachToEdge(dec, myEdgeIndex=0, targetEdgeIndex=8, outward=True)
    bow.attachToEdge(dec,  myEdgeIndex=1, targetEdgeIndex=0, outward=True)

    pent.draw(color="#ffee77")
    rhom.draw(color="#8894ff")
    hext.draw(color="#88ff99")
    bow.draw(color="#ff8899")

def main():
    setupScreen()
    drawDemoTiles()
    turtle.update()
    turtle.done()

if __name__ == "__main__":
    main()
