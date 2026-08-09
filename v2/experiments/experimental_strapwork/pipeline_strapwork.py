"""
EXPERIMENTAL / PARKED -- see `crossings.py` in this same package.

Self-contained strapwork-mode orchestration, mirroring what the active
`experiments/pipeline.py` does for tile-edges mode. Split out here (not
called from the live `pipeline.py`) so the active package has no dependency
on parked code.
"""
from .. import assemble, faces, preprocess
from .. import lines as lines_mod
from .. import vertices as vertices_mod
from . import crossings as crossings_mod
from . import strap_invert


def run(image_path, corners=None, side_mm=25.0, folder=None, groove_width_mm=1.5):
    img = preprocess.load_image(image_path)
    img = preprocess.strip_saturated_ink(img)
    img = preprocess.normalize_contrast(img)
    img, rectified = preprocess.rectify(img, corners=corners)

    line_info = lines_mod.extract_lines(img)

    # edge_direction_deg is calibrated to _mid's absolute y-up convention,
    # unlike plain lattice-snapping which is orientation-agnostic -- so
    # crossing-detection and phase both need to run in that same world-space
    # (y up) convention, not raw pixel space (y down).
    world_segments = vertices_mod._flip_y(line_info["segments"])
    world_phase = lines_mod.find_dominant_phase(world_segments)
    found = crossings_mod.find_crossings(world_segments, world_phase)
    graph = strap_invert.build_graph_from_strapwork(found, phase_deg=world_phase)

    tiles, unresolved = faces.classify_faces(graph["vertices"], graph["edges"], graph["unit_length"])

    result = {
        "rectified": rectified,
        "phase_deg": line_info["phase_deg"],
        "unit_length_px": graph["unit_length"],
        "vertices": graph["vertices"],
        "edges": graph["edges"],
        "tiles": tiles,
        "unresolved": unresolved,
    }

    if folder is not None:
        gen, spec, stats = assemble.reconstruct_to_folder(
            graph["vertices"], graph["edges"], tiles, unresolved,
            graph["unit_length"], folder, side_mm=side_mm, groove_width_mm=groove_width_mm,
        )
        result["gen"] = gen
        result["spec"] = spec
        result["stats"] = stats

    return result
