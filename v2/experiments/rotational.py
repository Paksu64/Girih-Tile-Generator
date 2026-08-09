"""
Rotational-symmetry reconstruction: detect a rotation center and fold order
(5 or 10, matching the generator's own `symmetry_mode="c5"/"c10"`) via
`symmetry.detect_rotational_symmetry`, then fill in sparse/damaged angular
sectors by rotating a well-reconstructed sector into their place -- the
rotational analog of `periodic.py`'s translational attach.

Unlike `periodic.py` (which crops and re-detects one unit in isolation),
this detects the whole image *once* with the standard tile-edges pipeline,
then buckets the resulting vertices by which angular sector (relative to
the detected center) they fall in. Good sectors are trusted as detected;
sectors with much less content than the best sector are assumed damaged/
missing and replaced by rotating the best sector's own vertex/edge
subgraph into that sector's position. Everything is then merged (same
tolerance-based position clustering used everywhere else in this pipeline)
and classified into tiles once, so the center and every sector-to-sector
boundary get properly attached rather than left as separate islands.
"""
import math
from collections import defaultdict

from . import assemble, faces, preprocess, symmetry
from . import lines as lines_mod
from . import vertices as vertices_mod


def _to_world(cx, cy):
    """Image pixel space (y down) -> world space (y up), matching
    vertices.build_graph's convention."""
    return cx, -cy


def _sector_of(pt, center, n):
    cx, cy = center
    x, y = pt
    ang = math.degrees(math.atan2(y - cy, x - cx)) % 360.0
    return int(ang // (360.0 / n)) % n


def _rotate_point(pt, center, angle_deg):
    cx, cy = center
    x, y = pt
    rad = math.radians(angle_deg)
    dx, dy = x - cx, y - cy
    c, s = math.cos(rad), math.sin(rad)
    return (cx + dx * c - dy * s, cy + dx * s + dy * c)


def reconstruct_and_fill_sectors(graph, center_world, n, damage_frac=0.5):
    """
    graph: {"vertices", "edges", "unit_length"} in world space, e.g. from
    `vertices.build_graph`.
    Returns a new {"vertices", "edges", "unit_length"} with damaged sectors
    replaced, seams merged -- ready for a single `faces.classify_faces`
    call. Also returns per-sector vertex counts for reporting.
    """
    vertices = graph["vertices"]
    edges = graph["edges"]

    sector_of_vertex = {vid: _sector_of(pos, center_world, n) for vid, pos in vertices.items()}
    counts = defaultdict(int)
    for s in sector_of_vertex.values():
        counts[s] += 1
    best_count = max(counts.values(), default=0)
    best_sector = max(counts, key=counts.get) if counts else 0

    damaged = {
        s for s in range(n)
        if counts.get(s, 0) < damage_frac * best_count
    }

    # Template: the best sector's own vertices/edges, kept in place (angle
    # delta 0 rotates it onto itself) -- reused for every damaged sector by
    # rotating by that sector's angular offset from the template.
    template_vids = {vid for vid, s in sector_of_vertex.items() if s == best_sector}
    template_edges = [
        (u, v) for u, v in edges
        if u in template_vids or v in template_vids
    ]

    raw_points = []
    raw_pairs = []

    def add_point(pos):
        raw_points.append(pos)
        return len(raw_points) - 1

    # Kept (non-damaged) sectors: original detected geometry, verbatim.
    kept_index = {}
    for vid, pos in vertices.items():
        if sector_of_vertex[vid] not in damaged:
            kept_index[vid] = add_point(pos)
    for u, v in edges:
        if u in kept_index and v in kept_index:
            raw_pairs.append((kept_index[u], kept_index[v]))

    # Damaged sectors: rotate the template subgraph into place. A template
    # edge is included if *either* endpoint is a template-sector vertex,
    # so edges reaching toward a neighboring sector still connect at the
    # (shared, later-merged) boundary.
    for s in sorted(damaged):
        delta = (s - best_sector) * (360.0 / n)
        local_index = {}
        for vid in template_vids:
            local_index[vid] = add_point(_rotate_point(vertices[vid], center_world, delta))
        # Also carry over the *other* endpoint of any boundary edge (not
        # itself in the template sector) so the rotated copy still has
        # somewhere to attach -- rotate it too; it'll merge with whatever
        # real geometry already occupies that neighboring position.
        for u, v in template_edges:
            for a, b in ((u, v), (v, u)):
                if a in template_vids and b not in local_index:
                    local_index[b] = add_point(_rotate_point(vertices[b], center_world, delta))
        for u, v in template_edges:
            raw_pairs.append((local_index[u], local_index[v]))

    unit_length = graph["unit_length"]
    tol = max(3.0, unit_length * 0.15)
    centers, cluster_of = vertices_mod._cluster_points(raw_points, tol)

    merged_edges = set()
    for i0, i1 in raw_pairs:
        c0, c1 = cluster_of[i0], cluster_of[i1]
        if c0 != c1:
            merged_edges.add((min(c0, c1), max(c0, c1)))

    out_vertices = {i: centers[i] for i in range(len(centers))}
    stats = {
        "n": n,
        "best_sector": best_sector,
        "damaged_sectors": sorted(damaged),
        "sector_vertex_counts": dict(counts),
    }
    return {"vertices": out_vertices, "edges": list(merged_edges), "unit_length": unit_length}, stats


def run(image_path, side_mm=25.0, folder=None, groove_width_mm=1.5, damage_frac=0.5):
    img = preprocess.load_image(image_path)
    img = preprocess.strip_saturated_ink(img)
    img = preprocess.normalize_contrast(img)
    img, rectified = preprocess.rectify(img, corners=None)

    sym = symmetry.detect_rotational_symmetry(img)
    if sym is None:
        raise ValueError("No confident rotational symmetry detected in this image")

    line_info = lines_mod.extract_lines(img)
    graph = vertices_mod.build_graph(line_info["segments"], phase_deg=line_info["phase_deg"])

    center_world = _to_world(*sym["center"])
    full_graph, sector_stats = reconstruct_and_fill_sectors(
        graph, center_world, sym["n"], damage_frac=damage_frac
    )
    tiles, unresolved = faces.classify_faces(
        full_graph["vertices"], full_graph["edges"], full_graph["unit_length"]
    )

    result = {
        "rectified": rectified,
        "center_px": sym["center"],
        "n": sym["n"],
        "symmetry_score": sym["score"],
        "sector_stats": sector_stats,
        "unit_length_px": full_graph["unit_length"],
        "vertices": full_graph["vertices"],
        "tiles": tiles,
        "unresolved": unresolved,
    }

    if folder is not None:
        gen, spec, stats = assemble.reconstruct_to_folder(
            full_graph["vertices"], full_graph["edges"], tiles, unresolved,
            full_graph["unit_length"], folder, side_mm=side_mm, groove_width_mm=groove_width_mm,
        )
        result["gen"] = gen
        result["spec"] = spec
        result["stats"] = stats

    return result
