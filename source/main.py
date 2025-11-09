import matplotlib.pyplot as plt
from geometry import Polygon, Rhombus, ElongatedHexagon, Bowtie, draw_poly
from animator import animator

def signed_area(points):
    a = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        a += x1 * y2 - x2 * y1
    return 0.5 * a

def draw_edge_frame(ax, poly, edge_index, scale=0.35, show_outward=True):
    (Ax, Ay), (Bx, By) = poly.edge(edge_index)
    u_hat, nL_hat, L = poly.edge_unit_vectors(edge_index)
    alen = L * scale
    Mx, My = (Ax + Bx) * 0.5, (Ay + By) * 0.5
    ax.arrow(Mx, My, alen * u_hat[0], alen * u_hat[1],
             head_width=alen * 0.08, head_length=alen * 0.12,
             length_includes_head=True, color="#55ffff", linewidth=1.8)
    ax.arrow(Mx, My, alen * nL_hat[0], alen * nL_hat[1],
             head_width=alen * 0.08, head_length=alen * 0.12,
             length_includes_head=True, color="#ffb74d", linewidth=1.8)
    if show_outward:
        ccw = signed_area(poly.points) > 0
        n_out = (-nL_hat[0], -nL_hat[1]) if ccw else nL_hat
        ax.arrow(Mx, My, alen * n_out[0], alen * n_out[1],
                 head_width=alen * 0.08, head_length=alen * 0.12,
                 length_includes_head=True, color="#b9f18c", linewidth=2.2)
    ax.plot([Ax, Bx], [Ay, By], color="#ffffff", linewidth=2.0, alpha=0.6)

def test():
    plt.style.use("dark_background")

    dec = Polygon(10, radius=1.0)
    rh = Rhombus(dec, edge_index=0)
    rh2 = Rhombus(dec, edge_index=1, acute_at_A=False)
    eh = ElongatedHexagon(dec, edge_index=7)
    eh2 = ElongatedHexagon(dec, edge_index=9, acute_at_A=False)
    eh3 = ElongatedHexagon(dec, edge_index=8, acute_at_A=False)
    bt = Bowtie(dec, edge_index=3)

    anim = animator(figsize=(6, 6))
    anim._fig.patch.set_facecolor("black")
    anim._ax.set_facecolor("black")

    anim.add_polygon(dec.points, color="#98dde9", linewidth=2.0)
    anim.add_polygon(rh.points, color="#989ce9", linewidth=2.0)
    anim.add_polygon(eh.points, color="#bde998", linewidth=2.0)
    anim.add_polygon(eh2.points, color="#bde998", linewidth=2.0)
    anim.add_polygon(rh2.points, color="#989ce9", linewidth=2.0)
    anim.add_polygon(eh3.points, color="#bde998", linewidth=2.0)
    anim.add_polygon(bt.points, color="#e998b5", linewidth=2.0)
    # Debug edge frames
    # draw_edge_frame(anim._ax, dec, edge_index=0, scale=0.35, show_outward=True)
    # draw_edge_frame(anim._ax, dec, edge_index=7, scale=0.35, show_outward=True)

    anim._ax.set_aspect("equal", adjustable="box")
    anim._ax.axis("off")

    ani = anim.animate(steps_per_seg=20, interval_ms=2)
    ani.save("girih.gif", writer="pillow", fps=30)
    plt.show()

def main():
    return

if __name__ == "__main__":
    test()
