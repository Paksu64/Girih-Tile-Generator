"""
Stage 1 of the reconstruction pipeline: image loading, contrast
normalization, and perspective rectification.

Automatic rectification looks for the largest quadrilateral contour in the
image (the frame/border a girih panel is usually mounted in) and warps it to
a fronto-parallel rectangle -- the same "largest quad" trick used for
document scanning. This is far more reliable on real architectural photos
than trying to find vanishing points from the girih strapwork/tile lines
themselves, since those never run in just two orthogonal directions (they
run in five, 36 degrees apart, by construction).

If no confident quad is found, `corners` can be supplied directly as a
same-outcome manual fallback.
"""
import cv2
import numpy as np


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def strip_saturated_ink(img, hue_low=150, hue_high=8, sat_min=40):
    """
    Inpaint over strongly-saturated colored content (annotation lines,
    colored ink, etc.) so line detection only sees the target black/dark-
    ink pattern. Traced directly to a real failure: a red tile-decomposition
    overlay drawn on a historical pattern scan was being detected as real
    pattern lines, bridging otherwise-separate regions of the true pattern
    into one spurious connected blob. Hue is restricted to the red end of
    the wheel (`hue_low`-180 or 0-`hue_high`) specifically so this doesn't
    also erase tan/brown paper staining, which sits at a distinctly
    different hue (~10-20) despite also having elevated saturation.

    Uses inpainting rather than a flat fill: painting a hard block color
    creates new sharp edges at the mask boundary that Canny then picks up
    as *more* spurious lines than it removes (confirmed empirically -- flat
    white fill increased detected segments here, it didn't reduce them).
    Inpainting blends with the surrounding paper texture instead.

    Paper and black/gray ink both have low saturation, so this is safe to
    apply unconditionally -- it only ever touches clearly non-target
    colored content, never the pattern itself.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, _ = cv2.split(hsv)
    mask = (((h > hue_low) | (h < hue_high)) & (s > sat_min)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)


def normalize_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l2, a, b]), cv2.COLOR_LAB2BGR)


def _order_corners(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return tl, tr, br, bl


def _four_point_transform(img, pts):
    tl, tr, br, bl = _order_corners(pts)
    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    W = int(max(w1, w2))
    H = int(max(h1, h2))
    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (W, H))


def _find_panel_quad(img, min_area_frac=0.15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    img_area = img.shape[0] * img.shape[1]
    best = None
    best_area = 0.0
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        area = cv2.contourArea(approx)
        if area < min_area_frac * img_area:
            continue
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2).astype(np.float32)
    return best


def rectify(img, corners=None):
    """
    Returns (rectified_img, used_rectification: bool).

    `corners` (4 (x, y) points, any order) forces the manual fallback.
    Otherwise attempts automatic panel-quad detection; if no confident quad
    is found the original image is returned unchanged.
    """
    if corners is not None:
        return _four_point_transform(img, np.array(corners, dtype=np.float32)), True

    quad = _find_panel_quad(img)
    if quad is None:
        return img, False
    return _four_point_transform(img, quad), True
