import math
import matplotlib.pyplot as plt

phi = (1 + 5**0.5) / 2

def rot(vx, vy, deg):
    r = math.radians(deg)
    c, s_ = math.cos(r), math.sin(r)
    return (c*vx - s_*vy, s_*vx + c*vy)

def signed_area(pts):
    a = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        a += x1 * y2 - x2 * y1
    return 0.5 * a

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
            n_hat = (-u_hat[1], u_hat[0])

            return u_hat, n_hat, length

    def draw(self):
        xs = [x for x, y in self.points] + [self.points[0][0]]
        ys = [y for x, y in self.points] + [self.points[0][1]]
        plt.plot(xs, ys, '-o')
        plt.axis('equal')
        plt.show()

class Rhombus(Polygon):
    def __init__(self, base_polygon, edge_index, acute_deg=72.0, acute_at_A=True, outward=True):
        import math

        A, B = base_polygon.edge(edge_index)
        u_hat, _, s = base_polygon.edge_unit_vectors(edge_index)

        alpha = float(acute_deg)

        # exterior turns at B, C, D (degrees)
        # interiors: [α, 180-α, α, 180-α] if acute_at_A else flipped
        if acute_at_A:
            turns = [alpha, 180.0 - alpha, alpha]
        else:
            turns = [180.0 - alpha, alpha, 180.0 - alpha]

        ccw = signed_area(base_polygon.points) > 0
        turn_sign = -1 if outward and ccw else (1 if outward and not ccw else (1 if ccw else -1))

        pts = [A, B]
        dirx, diry = u_hat
        curx, cury = B

        # make C and D; last edge (D->A) closes the rhombus
        for t in turns[:2]:
            dirx, diry = rot(dirx, diry, turn_sign * t)
            curx, cury = curx + s*dirx, cury + s*diry
            pts.append((curx, cury))

        self.n = 4
        self.radius = s
        self.points = pts

class ElongatedHexagon(Polygon):
    def __init__(self, base_polygon, edge_index, acute_at_A=True, outward=True):
        import math

        A, B = base_polygon.edge(edge_index)
        u_hat, _, s = base_polygon.edge_unit_vectors(edge_index)

        # interior pattern starting at A
        left_turns = ([108.0, 36.0, 108.0, 36.0, 36.0, 36.0]
                     if acute_at_A else
                     [36.0, 36.0, 36.0, 108.0, 36.0, 108.0])

        # choose sign so that "outward=True" goes outside regardless of CW/CCW
        ccw = signed_area(base_polygon.points) > 0
        turn_sign = -1 if outward and ccw else (1 if outward and not ccw else 1 if ccw else -1)

        pts = [A, B]
        dirx, diry = u_hat
        curx, cury = B

        # apply turns at B, C, D, E (F->A closes)
        for t in left_turns[1:5]:
            dirx, diry = rot(dirx, diry, turn_sign * t)
            curx, cury = curx + s*dirx, cury + s*diry
            pts.append((curx, cury))

        self.n = 6
        self.radius = s
        self.points = pts

class Bowtie(Polygon):
    def __init__(self, base_polygon, edge_index, outward=True):
        A, B = base_polygon.edge(edge_index)
        u_hat, _, s = base_polygon.edge_unit_vectors(edge_index)

        turns = [108.0, 324.0, 108.0, 108.0, 324.0]

        ccw = signed_area(base_polygon.points) > 0
        turn_sign = -1 if outward and ccw else (1 if outward and not ccw else (1 if ccw else -1))

        pts = [A, B]
        dirx, diry = u_hat
        curx, cury = B

        for t in turns:
            dirx, diry = rot(dirx, diry, turn_sign * t)
            curx, cury = curx + s*dirx, cury + s*diry
            pts.append((curx, cury))

        self.n = 6
        self.radius = s
        self.points = pts


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