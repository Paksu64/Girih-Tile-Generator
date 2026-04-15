from generator import GirihGenerator
from renderer import RendererMatplotlib

if __name__ == "__main__":
    gen = GirihGenerator(width=1000, height=1000,
                         sideLength=70, iterations=40, retries=5)

    # pick any%
    gen.runGen1()
    #gen.runGen2()
    #gen.runGen3()

    renderer = RendererMatplotlib()
    renderer.renderAnimated(gen)
