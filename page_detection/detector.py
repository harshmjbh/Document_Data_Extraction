"""
Page/document detection using OpenCV.
Pipeline: Gaussian blur -> Canny -> morph close -> contours ->
  Convex Hull -> 4 Extreme Points (handling perspective).
Selection by score (area * rectangularity * border * edge strength).
Subpixel refinement + snap corners to strongest edge; optional document-aspect filter.
Bilateral and noise-reduced blur passes for cleaner edges.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np

# Minimum rectangularity (quad area / bbox area); 0.38 allows tilted/irregular pages
RECT_MIN = 0.38
# Document aspect range (bbox width/height); None = disabled. 0.35-2.8 = portrait/landscape, up to 4.5 for spreads
DOC_ASPECT_RANGE = (0.35, 4.5)


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """
    Order 4 points into TL, TR, BR, BL.
    This ensures the polygon is drawn in a smooth clockwise loop.
    """
    if pts.shape[0] != 4:
        return pts

    # Sum: TL is min(x+y), BR is max(x+y)
    sum_xy = pts[:, 0] + pts[:, 1]
    tl = pts[np.argmin(sum_xy)]
    br = pts[np.argmax(sum_xy)]

    # Diff: TR is max(x-y), BL is min(x-y)
    diff_xy = pts[:, 0] - pts[:, 1]
    tr = pts[np.argmax(diff_xy)]
    bl = pts[np.argmin(diff_xy)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def _hull_to_quad(hull: np.ndarray) -> np.ndarray:
    """
    Get 4 extreme points from convex hull (TL, TR, BR, BL) as a quad.
    This effectively fits a quadrilateral to any convex shape (including curved spines).
    """
    if len(hull) < 4:
        return hull.astype(np.float32)

    pts = hull.reshape(-1, 2)
    sum_xy = pts[:, 0] + pts[:, 1]
    diff_xy = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(sum_xy)]
    br = pts[np.argmax(sum_xy)]
    tr = pts[np.argmax(diff_xy)]
    bl = pts[np.argmin(diff_xy)]
    return _order_corners(np.array([tl, tr, br, bl], dtype=np.float32))


def _contour_to_quad(contour: np.ndarray) -> Optional[np.ndarray]:
    """
    Convert a contour to a 4-point quad using Convex Hull extremes.
    We avoid minAreaRect because it forces 90-degree corners, which causes
    misalignment on perspective-distorted (trapezoidal) documents.
    """
    hull = cv2.convexHull(contour)
    if len(hull) < 4:
        return None
    # Always use hull extremes to allow for perspective/trapezoids
    return _hull_to_quad(hull)


def _border_proximity(quad: np.ndarray, image_shape: tuple) -> float:
    """Fraction of bbox edges touching/near image border. 0-1."""
    h, w = image_shape[:2]
    margin = max(2, 0.05 * min(w, h))
    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    touch_left = 1.0 if x <= margin else 0.0
    touch_right = 1.0 if (x + bw) >= (w - margin) else 0.0
    touch_top = 1.0 if y <= margin else 0.0
    touch_bottom = 1.0 if (y + bh) >= (h - margin) else 0.0
    return (touch_left + touch_right + touch_top + touch_bottom) / 4.0


def _quad_edge_strength(quad: np.ndarray, gray: np.ndarray) -> float:
    """Average edge strength along the quad boundary (0-1)."""
    h, w = gray.shape[:2]
    if h < 3 or w < 3:
        return 0.5
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    mag = np.clip(mag, 0, 255).astype(np.float32)

    pts = np.array(quad, dtype=np.float32)
    n_sample = 20
    total = 0.0
    count = 0

    for i in range(4):
        a, b = pts[i], pts[(i + 1) % 4]
        for j in range(n_sample):
            t = j / max(n_sample - 1, 1)
            x = a[0] * (1 - t) + b[0] * t
            y = a[1] * (1 - t) + b[1] * t
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                total += mag[yi, xi]
                count += 1

    if count == 0:
        return 0.5
    mean_mag = total / count
    return float(np.clip(mean_mag / 128.0, 0.0, 1.0))


def _quad_score(quad: np.ndarray, image_shape: tuple, gray: np.ndarray = None) -> float:
    """Score: area * rectangularity * border * edge."""
    area = cv2.contourArea(quad)
    if area <= 0:
        return 0.0
    h, w = image_shape[:2]
    image_area = w * h

    # 1. Area Ratio (larger is usually better)
    area_ratio = area / max(image_area, 1.0)

    # 2. Rectangularity (convex vs bbox)
    x, y, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    bbox_area = bw * bh
    rectangularity = (area / bbox_area) if bbox_area > 0 else 0.0

    # 3. Border Proximity (penalty for touching edges)
    border = _border_proximity(quad, image_shape)

    base = area_ratio * (0.4 + 0.6 * rectangularity) * (1.0 + 0.2 * border)

    # 4. Edge Strength (verify lines match image edges) — weight increased so edge-aligned quads win
    if gray is not None:
        edge = _quad_edge_strength(quad, gray)
        base *= 0.7 + 0.3 * edge

    return base


def _refine_quad_corners(quad: np.ndarray, gray: np.ndarray, image_shape: tuple) -> np.ndarray:
    """Refine corners to subpixel precision. Window size reduced to avoid overshoot."""
    h, w = image_shape[:2]
    if h < 11 or w < 11:
        return quad

    pts = np.array(quad, dtype=np.float32).reshape(4, 2)

    # Clip to be safely inside
    pts[:, 0] = np.clip(pts[:, 0], 5, w - 6)
    pts[:, 1] = np.clip(pts[:, 1], 5, h - 6)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.001)
    refined = cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), criteria)

    refined[:, 0] = np.clip(refined[:, 0], 0, w - 1)
    refined[:, 1] = np.clip(refined[:, 1], 0, h - 1)

    return _order_corners(refined.astype(np.float32))


def _snap_corners_to_edges(
    quad: np.ndarray, gray: np.ndarray, image_shape: tuple, window: int = 5, max_move: float = 4.0
) -> np.ndarray:
    """
    Move each corner to the strongest edge pixel within a small window.
    Reduces overshoot/crop by snapping to the actual page boundary.
    """
    h, w = image_shape[:2]
    if h < 2 * window + 1 or w < 2 * window + 1:
        return quad
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    pts = np.array(quad, dtype=np.float32).reshape(4, 2)
    out = np.empty_like(pts)
    for i in range(4):
        cx, cy = pts[i, 0], pts[i, 1]
        ix, iy = int(round(cx)), int(round(cy))
        best_val = -1.0
        best_x, best_y = cx, cy
        for dy in range(-window, window + 1):
            for dx in range(-window, window + 1):
                nx, ny = ix + dx, iy + dy
                if 0 <= nx < w and 0 <= ny < h:
                    dist = np.hypot(nx - cx, ny - cy)
                    if dist <= max_move and mag[ny, nx] > best_val:
                        best_val = mag[ny, nx]
                        best_x, best_y = float(nx), float(ny)
        # Only snap if we found a strong edge (avoid snapping to noise)
        if best_val >= 25.0:
            out[i, 0], out[i, 1] = best_x, best_y
        else:
            out[i, 0], out[i, 1] = cx, cy
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    return _order_corners(out.astype(np.float32))


def _reasonable_aspect(quad: np.ndarray) -> bool:
    """Reject extremely thin line-like quads."""
    _, _, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    if bw < 2 or bh < 2:
        return False
    aspect = max(bw, bh) / min(bw, bh)
    return aspect <= 8.0


def _document_aspect(quad: np.ndarray, image_area: float = 0) -> bool:
    """Keep quads that look like a document (portrait/landscape or spread). Disabled if DOC_ASPECT_RANGE is None."""
    if DOC_ASPECT_RANGE is None:
        return True
    _, _, bw, bh = cv2.boundingRect(quad.astype(np.int32))
    if bw < 2 or bh < 2:
        return False
    aspect = bw / bh
    lo, hi = DOC_ASPECT_RANGE
    if image_area > 0:
        qarea = cv2.contourArea(quad)
        if qarea >= image_area * 0.25 and lo <= aspect <= hi:
            return True
    return lo <= aspect <= hi


def _detect_from_edges(
    blurred: np.ndarray,
    image_shape: tuple,
    canny_low: int,
    canny_high: int,
    morph_size: int,
    min_area_ratio: float,
) -> Optional[np.ndarray]:
    """Run contour + quad selection on an edge map. Returns best quad found."""
    edges = cv2.Canny(blurred, canny_low, canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_size, morph_size))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    h, w = image_shape[:2]
    image_area = w * h
    min_area = image_area * min_area_ratio
    best_quad = None
    best_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # 1. Try ApproxPolyDP
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        quad = None
        # IMPROVEMENT: If approx is 4 points, use it.
        # If > 4 points (curved spine), use Hull Extremes instead of discarding.
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(np.float32)
        else:
            # Fallback: Extract the 4 corners from the hull of the contour
            quad = _contour_to_quad(contour)

        if quad is None:
            continue

        if not _reasonable_aspect(quad) or not _document_aspect(quad, image_area):
            continue

        # Basic rectangularity check (looser than minAreaRect)
        rect_area = cv2.contourArea(quad)
        _, _, bw, bh = cv2.boundingRect(quad.astype(np.int32))
        bbox_area = bw * bh
        if bbox_area > 0 and (rect_area / bbox_area) < RECT_MIN:
            continue

        if rect_area > best_area:
            best_area = rect_area
            best_quad = quad

    return _order_corners(best_quad) if best_quad is not None else None


def _detect_from_mask(mask: np.ndarray, image_shape: tuple) -> Optional[np.ndarray]:
    """
    Get document quad from a binary mask (e.g. from rembg or U-Net).
    Mask: (H, W) uint8, 255 = foreground (document). Finds largest document-like contour.
    """
    if mask is None or mask.size == 0:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.shape[2] == 3 else mask[:, :, 0]
    _, binary = cv2.threshold(mask.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image_shape[:2]
    image_area = w * h
    min_area = image_area * 0.08
    best_quad = None
    best_area = 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        quad = _contour_to_quad(contour)
        if quad is None or not _reasonable_aspect(quad) or not _document_aspect(quad, image_area):
            continue
        qa = cv2.contourArea(quad)
        _, _, qw, qh = cv2.boundingRect(quad.astype(np.int32))
        if qw * qh > 0 and (qa / (qw * qh)) < RECT_MIN:
            continue
        if qa > best_area:
            best_area = qa
            best_quad = quad
    return _order_corners(best_quad) if best_quad is not None else None


# Lazy-loaded rembg session (u2netp = faster/smaller model); reused across calls
_rembg_session = None


def _get_rembg_session():
    global _rembg_session
    if _rembg_session is None:
        try:
            from rembg import new_session
            _rembg_session = new_session("u2netp")
        except Exception:
            pass
    return _rembg_session


def detect_page_rembg(
    bgr_frame: np.ndarray,
    max_dim: int = 640,
    refine: bool = True,
) -> Optional[np.ndarray]:
    """
    Foolproof detection using saliency (rembg). Ignores lighting/shadows and similar background.
    Uses u2netp (lighter model) and optional resize for faster inference; session reused across calls.
    """
    try:
        from rembg import remove
        from PIL import Image
    except ImportError:
        return None
    if bgr_frame is None or bgr_frame.size == 0:
        return None
    h, w = bgr_frame.shape[:2]

    # Run rembg on resized image when large (big speedup)
    if max_dim and max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small_w, small_h = int(w * scale), int(h * scale)
        if small_w < 1 or small_h < 1:
            small_w, small_h = max(1, small_w), max(1, small_h)
        bgr_small = cv2.resize(bgr_frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
        rgb_small = cv2.cvtColor(bgr_small, cv2.COLOR_BGR2RGB)
        input_pil = Image.fromarray(rgb_small)
    else:
        rgb_small = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        input_pil = Image.fromarray(rgb_small)

    session = _get_rembg_session()
    try:
        if session is not None:
            output = remove(input_pil, only_mask=True, session=session)
        else:
            output = remove(input_pil, only_mask=True)
    except TypeError:
        try:
            output = remove(input_pil, only_mask=True)
        except TypeError:
            output = remove(input_pil)

    mask_np = np.array(output)
    if mask_np.ndim == 3:
        if mask_np.shape[2] == 4:
            mask_np = mask_np[:, :, 3]
        else:
            mask_np = mask_np[:, :, 0] if mask_np.shape[2] >= 1 else np.max(mask_np, axis=2)
    mask_u8 = (np.clip(mask_np, 0, 255)).astype(np.uint8)

    # Resize mask back to original size if we ran on a smaller image
    if max_dim and max(h, w) > max_dim and (mask_u8.shape[0] != h or mask_u8.shape[1] != w):
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_u8 = (mask_u8 > 127).astype(np.uint8) * 255

    quad = _detect_from_mask(mask_u8, (h, w))
    if quad is None:
        return None
    if refine:
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        quad = _refine_quad_corners(quad, gray, (h, w))
        quad = _snap_corners_to_edges(quad, gray, (h, w), window=5, max_move=4.0)
    return quad


def detect_page(
    bgr_frame: np.ndarray,
    method: str = "unet",
    fast: bool = False,
) -> Optional[np.ndarray]:
    """
    Detect a document quad.

    method:
      - "unet": U-Net-style segmentation (rembg/U²-Net) → mask → contours → quad. Default; robust.
      - "rembg": Same as unet (saliency/rembg).
      - "canny": OpenCV Canny + contours only (fast, can fail on low contrast).
      - "auto": Try canny first; if no quad, try unet/rembg.
    fast: if True and method is unet/rembg, skip corner refinement (faster, slightly less accurate).
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return None

    if method in ("unet", "rembg"):
        return detect_page_rembg(bgr_frame, max_dim=640, refine=not fast)

    # canny path (and first attempt for auto)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = bgr_frame.shape[:2]
    shape = (h, w)

    candidates: List[Tuple[float, np.ndarray]] = []

    # IMPROVEMENT: Reduced morph sizes (e.g. 5->3, 9->5) to reduce overshoot ("bloating")
    # Tighter parameters mean the mask hugs the page edge closer.
    params_list = [
        (50, 150, 3, 0.02),   # High contrast, very tight
        (25, 80, 5, 0.01),    # Med contrast, tight
        (30, 90, 5, 0.005),   # Lower contrast
        (25, 70, 9, 0.005),   # Fallback for blurry/large gaps
    ]

    for (canny_lo, canny_hi, morph, min_ratio) in params_list:
        q = _detect_from_edges(blurred, shape, canny_lo, canny_hi, morph, min_ratio)
        if q is not None:
            candidates.append((_quad_score(q, shape, gray), q))

    # Noise-reduced pass (strong blur for patterned backgrounds)
    blurred_strong = cv2.GaussianBlur(gray, (7, 7), 0)
    for (canny_lo, canny_hi, morph, min_ratio) in [(50, 150, 3, 0.02), (25, 80, 5, 0.01)]:
        q = _detect_from_edges(blurred_strong, shape, canny_lo, canny_hi, morph, min_ratio)
        if q is not None:
            candidates.append((_quad_score(q, shape, gray), q))

    # Bilateral pass: preserves edges while smoothing texture (cleaner page boundary)
    bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
    for (canny_lo, canny_hi, morph, min_ratio) in [(50, 150, 3, 0.02), (25, 80, 5, 0.01)]:
        q = _detect_from_edges(bilateral, shape, canny_lo, canny_hi, morph, min_ratio)
        if q is not None:
            candidates.append((_quad_score(q, shape, gray), q))

    # Adaptive Threshold pass (good for uneven lighting)
    # Re-using strict logic for consistency
    binary = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
    )
    # Simple contour finding on binary
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    adapt_best = None
    adapt_max_area = 0
    for c in contours:
        if cv2.contourArea(c) < (w * h * 0.15):
            continue
        q = _contour_to_quad(c)
        if q is not None and _reasonable_aspect(q) and _document_aspect(q, w * h):
            qa = cv2.contourArea(q)
            _, _, qw, qh = cv2.boundingRect(q.astype(np.int32))
            if qw * qh > 0 and (qa / (qw * qh)) >= RECT_MIN and qa > adapt_max_area:
                adapt_max_area = qa
                adapt_best = q
    if adapt_best is not None:
        candidates.append((_quad_score(adapt_best, shape, gray), adapt_best))

    if not candidates:
        # auto: fallback to rembg for foolproof behavior
        if method == "auto":
            return detect_page_rembg(bgr_frame, max_dim=640, refine=not fast)
        return None

    # Select best candidate by score
    best = max(candidates, key=lambda x: x[0])
    quad = best[1]

    # Refine corners (subpixel then snap to strongest edge in small window)
    quad = _refine_quad_corners(quad, gray, shape)
    quad = _snap_corners_to_edges(quad, gray, shape, window=5, max_move=4.0)

    return quad
