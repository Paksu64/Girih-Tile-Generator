import math
import matplotlib.pyplot as plt

phi = (1 + 5**0.5) / 2

class Polygon:
    def __init__(self, n, radius=1.0):
        self.n = n
        self.radius = radius
        self.points = []
        self.generate()

    def generate(self):
        self.points = []
        for i in range(self.n):
            angle = math.radians(i * (360 / self.n))
            x = self.radius * math.cos(angle)
            y = self.radius * math.sin(angle)
            self.points.append((x,y))

    def edge(self, i):
        A = self.points[i % self.n]
        B = self.points[(i + 1) % self.n]
        return A, B

    def edge_unit_vectors(self, i):
            (Ax, Ay), (Bx, By) = self.edge(i)

            # Direction vector from A to B
            dx = Bx - Ax
            dy = By - Ay

            # Edge length (magnitude of direction vector)
            length = math.hypot(dx, dy)

            # Unit vector along the edge
            u_hat = (dx / length, dy / length)

            # Left normal (rotate +90°)
            n_hat = (u_hat[1], u_hat[0])

            return u_hat, n_hat, length

    def draw(self):
        xs = [x for x, y in self.points] + [self.points[0][0]]
        ys = [y for x, y in self.points] + [self.points[0][1]]
        plt.plot(xs, ys, '-o')
        plt.axis('equal')
        plt.show()


class Rhombus(Polygon):
    def __init__(self, base_polygon, edge_index, acute_deg=72.0):
        # Edge geometry from the base polygon
        A, B = base_polygon.edge(edge_index)
        u_hat, n_hat, length = base_polygon.edge_unit_vectors(edge_index)

        # Compute height from the edge length
        h = length * math.sin(math.radians(acute_deg))

        # Four points of the rhombus
        Ax, Ay = A
        Bx, By = B
        nx, ny = n_hat
        C = (Bx + nx * h, By + ny * h)
        D = (Ax + nx * h, Ay + ny * h)
        points = [A, B, C, D]

        self.n = 4
        self.radius = length
        self.points = points


def draw_poly(points, ax, color):
    xs = [p[0] for p in points] + [points[0][0]]
    ys = [p[1] for p in points] + [points[0][1]]
    ax.plot(xs, ys, color=color)

def draw_ngon(n, radius):
    polygon = Polygon(n, radius)
    fig, ax = plt.subplots()
    draw_poly(polygon.points, ax, color='blue')
    plt.axis('equal')
    plt.show()
    return polygon