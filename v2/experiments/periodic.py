"""
Symmetry-based reconstruction: detect the pattern's repeating period
(`symmetry.py`), reconstruct one clean unit cell's raw vertex/edge graph
through the existing, validated tile-edges pipeline, then replicate and
*attach* that graph across the full detected extent using the period
vector -- adjacent copies get stitched into one connected tiling, not
placed side by side as independent islands.

Why attach at the graph level, not the tile level: a tile straddling a
period boundary gets cut in half by the unit crop -- copy k sees the right
half (typically left as "unresolved", since a half-tile isn't a closed
polygon `faces.py` can classify), copy k+1 sees the left half. Replicating
already-classified tiles (the first version of this module) discards both
halves forever. Replicating the raw graph and merging first means those two
halves become one closed polygon in the combined graph, which face
classification can then actually recognize as a whole tile -- for a
genuinely periodic source this isn't just cosmetic seam-fixing, it recovers
real tiles neither copy could resolve alone.

Replication happens on the pipeline's raw (pre-mm-scaled) vertex graph --
`assemble.build_generator` is the only place image-pixel coordinates get
converted to real-world mm, so doing the replication before that means one
copy of that scale conversion handles the whole assembled pattern, and the
period vector (measured directly in the source image's pixel space) can be
added to vertex coordinates with no unit conversion of its own.
"""
from . import assemble, faces, preprocess, symmetry
from . import lines as lines_mod
from . import vertices as vertices_mod


def reconstruct_unit(unit_img):
    line_info = lines_mod.extract_lines(unit_img)
    graph = vertices_mod.build_graph(line_info["segments"], phase_deg=line_info["phase_deg"])
    tiles, unresolved = faces.classify_faces(graph["vertices"], graph["edges"], graph["unit_length"])
    return graph, tiles, unresolved


def replicate_and_attach(graph, period_px, n_copies, merge_tol_frac=0.15):
    """
    Replicate the unit's raw vertex/edge graph across `n_copies`, then merge
    vertices that land on (approximately) the same point across a copy
    boundary -- the same tolerance-based clustering used everywhere else in
    this pipeline (`vertices._cluster_points`), not exact-equality, since
    detection noise means seam-matching vertices coincide closely but not
    bit-for-bit.

    period_px: (dx, dy) in image pixel space (y down), as returned by
    `symmetry.detect_period`. `graph["vertices"]` are already in world space
    (y up, see `vertices.build_graph`), so the y component of the period
    flips sign to match.

    Returns {"vertices", "edges", "unit_length"} -- same shape as
    `vertices.build_graph`, ready for a single `faces.classify_faces` call
    over the whole attached pattern.
    """
    dx, dy = period_px
    n_unit_vertices = (max(graph["vertices"].keys()) + 1) if graph["vertices"] else 0

    raw_points = []
    raw_index_of = {}  # (copy_k, orig_vid) -> index into raw_points
    raw_edges = []
    for k in range(n_copies):
        ox, oy = k * dx, -k * dy
        for vid, (x, y) in graph["vertices"].items():
            raw_index_of[(k, vid)] = len(raw_points)
            raw_points.append((x + ox, y + oy))
        for u, v in graph["edges"]:
            raw_edges.append((raw_index_of[(k, u)], raw_index_of[(k, v)]))

    unit_length = graph["unit_length"]
    tol = max(3.0, unit_length * merge_tol_frac)
    centers, cluster_of = vertices_mod._cluster_points(raw_points, tol)

    merged_edges = set()
    for u, v in raw_edges:
        cu, cv = cluster_of[u], cluster_of[v]
        if cu != cv:
            merged_edges.add((min(cu, cv), max(cu, cv)))

    vertices_out = {i: centers[i] for i in range(len(centers))}
    return {"vertices": vertices_out, "edges": list(merged_edges), "unit_length": unit_length}


def run(image_path, side_mm=25.0, folder=None, groove_width_mm=1.5, n_copies=None):
    img = preprocess.load_image(image_path)
    img = preprocess.strip_saturated_ink(img)
    img = preprocess.normalize_contrast(img)
    rect, rectified = preprocess.rectify(img, corners=None)

    period = symmetry.detect_period(rect)
    if period is None:
        raise ValueError("No confident repeating period detected in this image")
    period_confidence = symmetry.confidence(rect, period)
    dx, dy = period
    p = int(round(dx))
    w = rect.shape[1]
    max_copies = w // p
    if max_copies < 1:
        raise ValueError(f"Detected period ({p}px) is wider than the image ({w}px)")
    n_copies = max_copies if n_copies is None else min(n_copies, max_copies)

    unit_img = rect[:, 0:p]
    unit_graph, unit_tiles, unit_unresolved = reconstruct_unit(unit_img)

    full_graph = replicate_and_attach(unit_graph, period, n_copies)
    all_tiles, all_unresolved = faces.classify_faces(
        full_graph["vertices"], full_graph["edges"], full_graph["unit_length"]
    )

    result = {
        "rectified": rectified,
        "period_px": period,
        "period_confidence": period_confidence,
        "n_copies": n_copies,
        "unit_length_px": unit_graph["unit_length"],
        "unit_vertices": unit_graph["vertices"],
        "unit_tiles": unit_tiles,
        "unit_unresolved": unit_unresolved,
        "vertices": full_graph["vertices"],
        "tiles": all_tiles,
        "unresolved": all_unresolved,
    }

    if folder is not None:
        gen, spec, stats = assemble.reconstruct_to_folder(
            full_graph["vertices"], full_graph["edges"], all_tiles, all_unresolved,
            full_graph["unit_length"], folder, side_mm=side_mm, groove_width_mm=groove_width_mm,
        )
        result["gen"] = gen
        result["spec"] = spec
        result["stats"] = stats

    return result
