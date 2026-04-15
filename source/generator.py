# girih/generator.py
from collections import deque
import math, random
from tiles import Point, Tile, TILE_ANGLES, DLENS
from masks import masksCompatible, addMasks, removeMask

EPS = 1e-6

class GirihGenerator:

    def __init__(self, width=1000, height=1000, sideLength=50,
                 iterations=500, retries=10):
        self.width = width
        self.height = height
        self.sideLength = sideLength
        self.iterations = iterations
        self.retries = retries

        self.points = []     # list[Point]
        self.tiles = []      # list[Tile]
        self.pointDistances = []  # (index, dist)


    def tryAddPoint(self, length, coords, mask):
        x, y = coords
        for i, p in enumerate(self.points):
            if abs(p.coords[0]-x) < EPS*length and abs(p.coords[1]-y) < EPS*length:
                # Attempt merge
                if masksCompatible(p.mask, mask):
                    p.mask = addMasks(p.mask, mask)
                    return i
                return -1

        idx = len(self.points)
        self.points.append(Point(coords, mask[:]))
        dist = math.hypot(x, y)
        self._registerDistance(idx, dist)
        return idx

    def _registerDistance(self, idx, dist):
        inserted = False
        for i, (pid, d) in enumerate(self.pointDistances):
            if dist < d:
                self.pointDistances.insert(i, (idx, dist))
                inserted = True
                break
        if not inserted:
            self.pointDistances.append((idx, dist))


    def testTileOverlap(self, newIdxs, length):
        mind = (1 - EPS) * length
        for ni in newIdxs:
            px, py = self.points[ni].coords
            for i, p in enumerate(self.points):
                if i == ni or i in newIdxs:
                    continue
                if math.hypot(p.coords[0]-px, p.coords[1]-py) < mind:
                    return False
        return True


    def addTile(self, tileType, offset, length, x, y, angleDeg):
        angles = TILE_ANGLES[tileType]
        n = len(angles)

        rollback = len(self.points)
        idxs = []
        maskRoll = {}

        cx, cy = x, y
        alfa = angleDeg

        for i in range(n):
            ang = angles[(i + offset) % n]
            prev = angles[(i + offset + n - 1) % n]

            mask = [0]*10
            betaSector = alfa / 36
            for j in range(prev):
                mask[int((j + betaSector + 10) % 10)] = 1

            pIdx = self.tryAddPoint(length, (cx, cy), mask)
            if pIdx < 0:
                # rollback
                for pid, m in maskRoll.items():
                    removeMask(self.points[pid], m)
                self.points = self.points[:rollback]
                return -1

            maskRoll[pIdx] = mask[:]
            idxs.append(pIdx)

            rad = math.radians(alfa)
            cx += math.cos(rad) * length
            cy += math.sin(rad) * length
            alfa += 180 - ang*36

        newIdxs = list(range(rollback, len(self.points)))
        if not self.testTileOverlap(newIdxs, length):
            for pid, m in maskRoll.items():
                removeMask(self.points[pid], m)
            self.points = self.points[:rollback]
            return -1

        self.tiles.append(Tile(tileType, idxs, offset, length, alfa, maskRoll))
        return len(self.tiles)-1


    def fillSlot(self, length, x, y, ff, n):
        if n < 2 or n > 8:
            return

        def place(t, ofs):
            self.addTile(t, ofs, length, x, y, ff*36)

        if n == 2:
            opts = [(1,1),(1,4),(2,1),(2,2),(3,1)]
            place(*random.choice(opts))
            return

        if n == 3:
            place(*random.choice([(3,2),(4,1)]))
            return

        if n == 4:
            if random.random() > 0.5:
                place(*random.choice([(0,1),(1,2),(1,3)]))
            else:
                self.fillSlot(length, x, y, ff, 2)
                self.fillSlot(length, x, y, ff+2, 2)
            return

        if n == 5:
            if random.random() > 0.5:
                self.fillSlot(length,x,y,ff,2)
                self.fillSlot(length,x,y,ff+2,3)
            else:
                self.fillSlot(length,x,y,ff,3)
                self.fillSlot(length,x,y,ff+3,2)
            return

        if n == 6:
            r = random.randrange(5)
            if r == 0:
                self.addTile(2,3,length,x,y,(ff+3)*36)
            elif r == 1:
                self.fillSlot(length,x,y,ff,2)
                self.fillSlot(length,x,y,ff+2,2)
                self.fillSlot(length,x,y,ff+4,2)
            elif r == 2:
                self.fillSlot(length,x,y,ff,3)
                self.fillSlot(length,x,y,ff+3,3)
            elif r == 3:
                self.fillSlot(length,x,y,ff,2)
                self.fillSlot(length,x,y,ff+2,4)
            else:
                self.fillSlot(length,x,y,ff,4)
                self.fillSlot(length,x,y,ff+4,2)
            return

        if n == 7:
            r = random.randrange(5)
            if r == 0:
                self.fillSlot(length,x,y,ff,2)
                self.fillSlot(length,x,y,ff+2,2)
                self.fillSlot(length,x,y,ff+4,3)
            elif r == 1:
                self.fillSlot(length,x,y,ff,2)
                self.fillSlot(length,x,y,ff+2,3)
                self.fillSlot(length,x,y,ff+5,2)
            elif r == 2:
                self.fillSlot(length,x,y,ff,3)
                self.fillSlot(length,x,y,ff+3,2)
                self.fillSlot(length,x,y,ff+5,2)
            elif r == 3:
                self.fillSlot(length,x,y,ff,3)
                self.fillSlot(length,x,y,ff+3,4)
            else:
                self.fillSlot(length,x,y,ff,4)
                self.fillSlot(length,x,y,ff+4,3)
            return

        if n == 8:
            r = random.randrange(10)
            if r == 0:
                self.fillSlot(length,x,y,ff,2)
                self.fillSlot(length,x,y,ff+2,2)
                self.fillSlot(length,x,y,ff+4,2)
                self.fillSlot(length,x,y,ff+6,2)
                return
            if r == 1:
                self.fillSlot(length,x,y,ff,2)
                self.fillSlot(length,x,y,ff+2,2)
                self.fillSlot(length,x,y,ff+4,4)
                return
            if r == 2:
                self.fillSlot(length,x,y,ff,2)
                self.fillSlot(length,x,y,ff+2,4)
                self.fillSlot(length,x,y,ff+6,2)
                return
            if r == 3:
                self.fillSlot(length,x,y,ff,4)
                self.fillSlot(length,x,y,ff+4,2)
                self.fillSlot(length,x,y,ff+6,2)
                return
            if r == 4:
                self.fillSlot(length,x,y,ff,2)
                self.fillSlot(length,x,y,ff+2,3)
                self.fillSlot(length,x,y,ff+5,3)
                return
            if r == 5:
                self.fillSlot(length,x,y,ff,3)
                self.fillSlot(length,x,y,ff+3,2)
                self.fillSlot(length,x,y,ff+5,3)
                return
            if r == 6:
                self.fillSlot(length,x,y,ff,3)
                self.fillSlot(length,x,y,ff+3,3)
                self.fillSlot(length,x,y,ff+6,2)
                return
            if r == 7:
                self.fillSlot(length,x,y,ff,4)
                self.fillSlot(length,x,y,ff+4,4)
                return
            if r == 8:
                self.fillSlot(length,x,y,ff,2)
                self.fillSlot(length,x,y,ff+2,6)
                return
            if r == 9:
                self.fillSlot(length,x,y,ff,6)
                self.fillSlot(length,x,y,ff+6,2)
                return

    def fillPoint(self, idx, length):
        if idx < 0 or idx >= len(self.points):
            return -1
        p = self.points[idx]

        first = -2
        freeCnt = 0

        for i in range(11):
            s = p.mask[i%10]
            if s == 1 and first == -2:
                first = -1
            if s == 0 and first == -1:
                first = i%10

        if first < 0:
            return -1

        while p.mask[first%10] == 0 and freeCnt < 12:
            freeCnt += 1
            first += 1
        first -= freeCnt

        if freeCnt > 10:
            return -1

        x, y = p.coords
        self.fillSlot(length, x, y, first, freeCnt)
        return 0

    def addInitialDecagon(self):
        self.addTile(0, 0, self.sideLength,
                     -self.sideLength/2,
                     -1.5388*self.sideLength,
                     0)

    def isVisiblePoint(self, i):
        if 0 <= i < len(self.points):
            x,y = self.points[i].coords
            return (-self.width/2 <= x <= self.width/2 and
                    -self.height/2 <= y <= self.height/2)
        return False
    
    def _pickClosestVisiblePoint(self):
        """
        self.pointDistances: [(pointIndex, distance), ...]
        En küçük distance'a sahip ve görünür olan noktayı seçer.
        Bulamazsa None döner.
        """
        if not self.pointDistances:
            return None

        # Mesafeye göre küçükten büyüğe (stabil) sırala
        # (pointDistances zaten sıralıysa bu satırı kaldırabilirsin)
        sorted_pd = sorted(self.pointDistances, key=lambda x: x[1])

        for idx, _dist in sorted_pd:
            if self.isVisiblePoint(idx):
                return idx

        return None

    def runGen1(self):
        if not self.tiles:
            self.addInitialDecagon()

        i = 0
        steps = 0
        while i < len(self.points) and steps < self.iterations:
            if self.isVisiblePoint(i):
                for _ in range(self.retries):
                    self.fillPoint(i, self.sideLength)
            i += 1
            steps += 1

    def runGen2(self):
        if not self.tiles:
            self.addInitialDecagon()

        for _ in range(self.iterations):
            if not self.pointDistances:
                break

            placed_any = False

            # Bu turda en fazla L kez aday dene (tümü görünmez olabilir)
            L = len(self.pointDistances)
            tried = 0

            while tried < L and not placed_any:
                idx = self._pickClosestVisiblePoint()
                tried += 1

                if idx is None:
                    break

                tiles_before = len(self.tiles)

                for _ in range(self.retries):
                    self.fillPoint(idx, self.sideLength)

                if len(self.tiles) > tiles_before:
                    placed_any = True
                else:
                    # Bu idx başarısızsa, aynı idx tekrar seçilmesin diye listeden geçici çıkar
                    # (pointDistances içinde tekil idx varsayımıyla)
                    self.pointDistances = [(i, d) for (i, d) in self.pointDistances if i != idx]

            if not placed_any:
                break


    def runGen3(self):
        if not self.tiles:
            self.addInitialDecagon()

        q = deque([0])      # frontier queue
        in_q = {0}          # tekrar eklemeyi önle
        steps = 0

        while q and steps < self.iterations:
            idx = q.popleft()
            in_q.discard(idx)

            if not self.isVisiblePoint(idx):
                steps += 1
                continue

            tiles_before = len(self.tiles)
            points_before = len(self.points)

            for _ in range(self.retries):
                self.fillPoint(idx, self.sideLength)

            # Eğer bu nokta gerçekten büyüme ürettiyse, yeni noktaları frontier'e kat
            if len(self.tiles) > tiles_before:
                for ni in range(points_before, len(self.points)):
                    if ni not in in_q:
                        q.append(ni)
                        in_q.add(ni)

            steps += 1
