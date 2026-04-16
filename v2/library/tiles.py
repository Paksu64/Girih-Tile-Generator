from dataclasses import dataclass, field
from typing import List, Tuple, Dict

Vec2 = Tuple[float, float]

# (36° units)
TILE_ANGLES = [
    [4,4,4,4,4,4,4,4,4,4],      # decagon
    [2,4,4,2,4,4],              # elongated hex
    [2,2,6,2,2,6],              # bowtie
    [2,3,2,3],                  # rhombus
    [3,3,3,3,3],                # pentagon
]

DLENS = [
    [[0.96,0.96]] * 10,
    [[0.61,0.61]] * 6,
    [[0.38,0.38]] * 6,
    [[0.44,0.44]] * 4,
    [[0.44,0.44]] * 5,
]


@dataclass
class Point:
    coords: Vec2
    mask: List[int] = field(default_factory=lambda: [0]*10)

    def maskSum(self) -> int:
        return sum(self.mask)


@dataclass
class Tile:
    tileType: int                  # 0..4
    pointIndexes: List[int]        # indices into points list
    offset: int
    sideLength: float
    rotationDeg: float
    masks: Dict[int, List[int]]    # pointIndex -> mask
