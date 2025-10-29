import matplotlib.pyplot as plt
from geometry import Polygon, Rhombus, draw_poly
from animator import animator

def main():
    plt.style.use("dark_background")

    dec = Polygon(10, radius=1.0)
    edge_index = 0
    rh = Rhombus(dec, edge_index=edge_index)

    anim = animator(figsize=(6, 6))
    anim._fig.patch.set_facecolor("black")
    anim._ax.set_facecolor("black")

    anim.add_polygon(dec.points, color="#98dde9", linewidth=2.0)
    anim.add_polygon(rh.points, color="#989ce9", linewidth=2.0)

    ani = anim.animate(steps_per_seg=20, interval_ms=5)
    plt.show()

def test():
    return

if __name__ == "__main__":
    main()
