"""Page detection: detect document quad and draw bounding box (OpenCV + optional rembg)."""

from .detector import detect_page, detect_page_rembg
from .draw import draw_quad_on_frame

__all__ = ["detect_page", "detect_page_rembg", "draw_quad_on_frame"]
