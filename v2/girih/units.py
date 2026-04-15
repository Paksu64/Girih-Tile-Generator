"""
Unit conversion and canvas specification.

Everything inside the generator runs in abstract "world units" where
1 world unit = 1 pixel at 96 dpi.  This module converts between
millimetres and world units so the rest of the pipeline stays clean.

96 dpi means 1 inch = 96 px, 1 mm = 96/25.4 ≈ 3.7795 px.
"""

MM_PER_INCH = 25.4
PX_PER_INCH = 96.0          # SVG reference pixel density
PX_PER_MM   = PX_PER_INCH / MM_PER_INCH   # ≈ 3.7795


def mm_to_px(mm: float) -> float:
    return mm * PX_PER_MM


def px_to_mm(px: float) -> float:
    return px / PX_PER_MM


def parse_mm(value: str) -> float:
    """Parse a string like '120mm' or '120' (bare number also treated as mm)."""
    v = value.strip().lower()
    if v.endswith("mm"):
        return float(v[:-2])
    return float(v)          # bare number → mm


class DesignSpec:
    """
    The single source of truth for a design's physical dimensions.

    All public attributes are in mm.
    Use .to_px() to get the world-unit values the generator needs.
    """

    def __init__(self, width_mm: float, height_mm: float, side_mm: float):
        self.width_mm  = width_mm
        self.height_mm = height_mm
        self.side_mm   = side_mm

    # ── pixel equivalents (what the generator sees) ────────────────────

    @property
    def width_px(self) -> float:
        return mm_to_px(self.width_mm)

    @property
    def height_px(self) -> float:
        return mm_to_px(self.height_mm)

    @property
    def side_px(self) -> float:
        return mm_to_px(self.side_mm)

    # ── derived metrics ────────────────────────────────────────────────

    @property
    def tiles_wide(self) -> float:
        """Approximate number of tile-widths across the canvas."""
        return self.width_mm / self.side_mm

    @property
    def tiles_tall(self) -> float:
        return self.height_mm / self.side_mm

    def summary(self) -> str:
        return (
            f"Canvas : {self.width_mm:.1f} × {self.height_mm:.1f} mm\n"
            f"Tile   : {self.side_mm:.1f} mm side length\n"
            f"Grid   : ~{self.tiles_wide:.1f} × {self.tiles_tall:.1f} tiles"
        )
