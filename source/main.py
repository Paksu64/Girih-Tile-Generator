from generator import GirihGenerator
from renderer import RendererMatplotlib

if __name__ == "__main__":
    gen = GirihGenerator(width=500, height=500,
                         sideLength=20, iterations=500, retries=10)

    # pick any
    gen.runGen1()
    # gen.runGen2()
    # gen.runGen3()

    renderer = RendererMatplotlib()
    renderer.render(gen)
