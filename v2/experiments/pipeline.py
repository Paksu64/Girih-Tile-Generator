"""End-to-end orchestration of the reconstruction pipeline stages:
preprocess -> lines -> vertices -> faces -> assemble.

Strapwork input mode was tried and parked (see
`experiments/experimental_strapwork/`) -- its math is validated exact,
but real-photo crossing-matching recall plateaus around 50%, and that gap
needs its own follow-up work. This module only carries the working
tile-edges path now; `periodic.py` builds on it for symmetry-based
reconstruction.
"""
from . import assemble, faces, preprocess
from . import lines as lines_mod
from . import vertices as vertices_mod


def run(image_path, corners=None, side_mm=25.0, folder=None, groove_width_mm=1.5,
        input_mode="tile-edges"):
    if input_mode != "tile-edges":
        raise ValueError(
            f"unsupported input_mode: {input_mode!r} (only 'tile-edges' is active; "
            "strapwork mode is parked in experiments/experimental_strapwork/)"
        )

    img = preprocess.load_image(image_path)
    img = preprocess.strip_saturated_ink(img)
    img = preprocess.normalize_contrast(img)
    img, rectified = preprocess.rectify(img, corners=corners)

    line_info = lines_mod.extract_lines(img)
    graph = vertices_mod.build_graph(line_info["segments"], phase_deg=line_info["phase_deg"])

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
