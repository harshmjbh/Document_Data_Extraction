"""Draw document quad (bounding box) on a BGR frame."""

import numpy as np
import cv2


def draw_quad_on_frame(
    frame: np.ndarray,
    quad: np.ndarray,
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    in_place: bool = True,
) -> np.ndarray:
    """
    Draw the document quad as a closed polygon on the frame.

    Args:
        frame: BGR image (H, W, 3).
        quad: (4, 2) array TL, TR, BR, BL.
        color: BGR tuple, default green.
        thickness: line thickness in pixels.
        in_place: if True, modify frame; else copy first.

    Returns:
        Frame with quad drawn (same object as frame if in_place, else copy).
    """
    if not in_place:
        frame = frame.copy()
    pts = np.array(quad, dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)
    return frame
