
"""
animator.py — Quick line-by-line path animator using matplotlib.

Usage (basic):
    from animator import animator
    import numpy as np

    anim = animator(figsize=(6, 6))
    square = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    anim.add_polygon(square, linewidth=2.0)

    theta = np.linspace(0, 2*np.pi, 6)[:-1]
    R1, R2 = 1.4, 0.55
    star = []
    for i in range(5):
        star.append((R1*np.cos(theta[i]), R1*np.sin(theta[i])))
        star.append((R2*np.cos(theta[i] + np.pi/5), R2*np.sin(theta[i] + np.pi/5)))
    star.append(star[0])
    anim.add_polyline(star, closed=False, linewidth=2.0)

    # Preview animation (in notebooks) or save:
    ani = anim.animate(steps_per_seg=20, interval_ms=20)
    # anim.save_gif("demo.gif", fps=50, steps_per_seg=20, interval_ms=20)

Notes:
  • No seaborn dependency; pure matplotlib.
  • Each path is broken into segments and drawn incrementally frame-by-frame.
  • You can pass per-path kwargs (e.g., color, linewidth, alpha) and they
    will be applied to that path's segments.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

Point = Tuple[float, float]


@dataclass
class _Segment:
    x1: float
    y1: float
    x2: float
    y2: float
    style: Dict[str, Any]


class animator:
    def __init__(
        self,
        figsize: Tuple[float, float] = (6, 6),
        axis_off: bool = True,
        equal_aspect: bool = True,
    ) -> None:
        self._segments: List[_Segment] = []
        self._fig, self._ax = plt.subplots(figsize=figsize)
        self._lines = []  # matplotlib Line2D instances, one per segment
        if equal_aspect:
            self._ax.set_aspect("equal", adjustable="box")
        if axis_off:
            self._ax.axis("off")

    # ------------------------------- Building paths -------------------------------
    def add_polyline(
        self,
        pts: Iterable[Point],
        closed: bool = False,
        **style: Any,
    ) -> "animator":
        """
        Add a polyline (open by default). If closed=True, last point connects to first.
        Style kwargs are forwarded to Line2D (e.g., color, linewidth, alpha, linestyle).
        """
        pts = list(pts)
        if len(pts) < 2:
            return self
        if closed and (pts[0] != pts[-1]):
            pts = pts + [pts[0]]
        for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
            self._segments.append(_Segment(x1, y1, x2, y2, dict(style)))
        return self

    def add_polygon(self, pts: Iterable[Point], **style: Any) -> "animator":
        """Convenience for a closed polyline."""
        return self.add_polyline(pts, closed=True, **style)

    def add_paths(
        self,
        paths: Sequence[Iterable[Point]],
        closed: bool | Sequence[bool] = False,
        styles: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> "animator":
        """
        Add many polylines in one call.
        - closed: bool or sequence matching paths
        - styles: per-path style dicts; missing entries fallback to {}
        """
        if isinstance(closed, bool):
            closed_flags = [closed] * len(paths)
        else:
            closed_flags = list(closed)
            if len(closed_flags) != len(paths):
                raise ValueError("len(closed) must match len(paths)")

        styles = styles or [{} for _ in paths]
        if len(styles) != len(paths):
            raise ValueError("len(styles) must match len(paths)")

        for p, c, st in zip(paths, closed_flags, styles):
            self.add_polyline(p, closed=c, **(st or {}))
        return self

    # ------------------------------- Animation core -------------------------------
    def _autoscale(self, xlim: Optional[Tuple[float, float]], ylim: Optional[Tuple[float, float]]) -> None:
        xs = [v for s in self._segments for v in (s.x1, s.x2)]
        ys = [v for s in self._segments for v in (s.y1, s.y2)]
        if xs and ys:
            if xlim is None:
                pad = (max(xs) - min(xs)) * 0.1
                self._ax.set_xlim(min(xs) - pad, max(xs) + pad)
            else:
                self._ax.set_xlim(*xlim)
            if ylim is None:
                pad = (max(ys) - min(ys)) * 0.1
                self._ax.set_ylim(min(ys) - pad, max(ys) + pad)
            else:
                self._ax.set_ylim(*ylim)

    def animate(
        self,
        steps_per_seg: int = 20,
        interval_ms: int = 20,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        blit: bool = False,
        repeat: bool = False,
        default_linewidth: float = 2.0,
    ):
        """
        Create the matplotlib.animation.FuncAnimation.
        Return value is the FuncAnimation instance.
        """
        if not self._segments:
            raise ValueError("No segments added. Use add_polyline/add_polygon first.")

        # Create one Line2D per segment with its own style.
        self._lines = []
        for seg in self._segments:
            kwargs = dict(seg.style)
            if "linewidth" not in kwargs:
                kwargs["linewidth"] = default_linewidth
            line, = self._ax.plot([], [], **kwargs)
            self._lines.append(line)

        # Scale axes
        self._autoscale(xlim, ylim)

        total_frames = steps_per_seg * len(self._segments)

        def init():
            for ln in self._lines:
                ln.set_data([], [])
            return self._lines

        def update(frame: int):
            f = min(frame, total_frames - 1)
            seg_idx = f // steps_per_seg
            t = (f % steps_per_seg) / float(steps_per_seg)

            for i, seg in enumerate(self._segments):
                if i < seg_idx:
                    self._lines[i].set_data([seg.x1, seg.x2], [seg.y1, seg.y2])
                elif i == seg_idx:
                    xt = seg.x1 + t * (seg.x2 - seg.x1)
                    yt = seg.y1 + t * (seg.y2 - seg.y1)
                    self._lines[i].set_data([seg.x1, xt], [seg.y1, yt])
                else:
                    self._lines[i].set_data([], [])
            return self._lines

        ani = FuncAnimation(
            self._fig,
            update,
            frames=total_frames,
            init_func=init,
            interval=interval_ms,
            blit=blit,
            repeat=repeat,
        )
        return ani

    # ------------------------------- Export helpers -------------------------------
    def save_gif(self, path: str, fps: int = 50, **animate_kwargs: Any) -> None:
        """
        Convenience: build animation and save as GIF.
        Example:
            anim.save_gif("out.gif", fps=50, steps_per_seg=25, interval_ms=20)
        """
        ani = self.animate(**animate_kwargs)
        writer = PillowWriter(fps=fps)
        ani.save(path, writer=writer)

    def save_mp4(self, path: str, fps: int = 50, **animate_kwargs: Any) -> None:
        """
        Save as MP4 using ffmpeg if available.
        """
        from matplotlib.animation import FFMpegWriter
        ani = self.animate(**animate_kwargs)
        writer = FFMpegWriter(fps=fps)
        ani.save(path, writer=writer)


# ------------------------------- CLI Demo -------------------------------
if __name__ == "__main__":
    # Example: draw a square and a star and save as GIF
    import numpy as np
    sq = [(-1.0, -1.0), (1.0, -1.0), (1.0,  1.0), (-1.0,  1.0)]
    theta = np.linspace(0, 2*np.pi, 6)[:-1]
    R1, R2 = 1.4, 0.55
    star = []
    for i in range(5):
        star.append((R1*np.cos(theta[i]), R1*np.sin(theta[i])))
        star.append((R2*np.cos(theta[i] + np.pi/5), R2*np.sin(theta[i] + np.pi/5)))
    star.append(star[0])

    q = animator(figsize=(6, 6))
    q.add_polygon(sq, linewidth=2.0)
    q.add_polyline(star, closed=False, linewidth=2.0)
    q.save_gif("animator_demo.gif", fps=50, steps_per_seg=20, interval_ms=20)
    print("Saved animator_demo.gif")
