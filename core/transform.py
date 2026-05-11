from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Helper nội bộ
# ──────────────────────────────────────────────────────────────────────────────

def _to_float32_cv(pts: List[List[float]]) -> np.ndarray:
    """Chuyển list-of-[u,v] → np.float32 shape (N, 1, 2) cho cv2."""
    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


def _default_foot(bbox: List[float]) -> Tuple[float, float]:
    """Bottom-center của bounding box [x1, y1, x2, y2]."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


# ──────────────────────────────────────────────────────────────────────────────
# Main class
# ──────────────────────────────────────────────────────────────────────────────

class HomographyTransformer:

    def __init__(
        self,
        src_points: List[List[float]],
        dst_points: List[List[float]],
        canvas_w: int = 600,
        canvas_h: int = 800,
        real_w_m: float = 8.0,
        real_h_m: float = 10.0,
    ) -> None:
        if len(src_points) < 4 or len(dst_points) < 4:
            raise ValueError(
                "Cần ít nhất 4 cặp điểm (src, dst) để tính Homography."
            )

        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.real_w_m = real_w_m
        self.real_h_m = real_h_m

        # Scale pixel - mét
        self.px_per_m_x: float = canvas_w / real_w_m
        self.px_per_m_y: float = canvas_h / real_h_m

        # ── Tính ma trận H ────────────────────────────────────────────────────
        src = _to_float32_cv(src_points)
        dst = _to_float32_cv(dst_points)

        self._H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if self._H is None:
            raise RuntimeError(
                "cv2.findHomography() trả về None. "
                "Kiểm tra lại src_points / dst_points trong settings.yaml."
            )

        self._H_inv: np.ndarray = np.linalg.inv(self._H)
        self._src_points_raw: List[List[float]] = src_points  # lưu để draw

        n_inliers = int(status.sum()) if status is not None else "?"
        logger.info(
            "[HomographyTransformer] Khởi tạo OK — inliers %s/%d, "
            "canvas %dx%d px, phòng %.1f×%.1fm",
            n_inliers, len(src_points), canvas_w, canvas_h, real_w_m, real_h_m,
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: dict) -> "HomographyTransformer":

        return cls(
            src_points=cfg["src_points"],
            dst_points=cfg["dst_points"],
            canvas_w=int(cfg.get("floor_canvas_w", 600)),
            canvas_h=int(cfg.get("floor_canvas_h", 800)),
            real_w_m=float(cfg.get("real_width_m", 8.0)),
            real_h_m=float(cfg.get("real_height_m", 10.0)),
        )

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def H(self) -> np.ndarray:
        """Ma trận Homography 3×3 (copy, read-only)."""
        return self._H.copy()

    @property
    def H_inv(self) -> np.ndarray:
        """Inverse Homography 3×3 (BEV → camera)."""
        return self._H_inv.copy()

    # ── Biến đổi toạ độ ───────────────────────────────────────────────────────

    def transform_point(
        self,
        pixel_xy: Tuple[float, float],
    ) -> Tuple[int, int]:
        
        pt = np.array([[[pixel_xy[0], pixel_xy[1]]]], dtype=np.float32)
        res = cv2.perspectiveTransform(pt, self._H)  # (1, 1, 2)
        x = int(np.clip(res[0, 0, 0], 0, self.canvas_w - 1))
        y = int(np.clip(res[0, 0, 1], 0, self.canvas_h - 1))
        return x, y

    def transform_points_batch(
        self,
        points: np.ndarray,
    ) -> np.ndarray:
        
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("points phải có shape (N, 2).")

        pts = points.astype(np.float32).reshape(-1, 1, 2)
        res = cv2.perspectiveTransform(pts, self._H)  # (N, 1, 2)
        res = res.reshape(-1, 2)
        res[:, 0] = np.clip(res[:, 0], 0, self.canvas_w - 1)
        res[:, 1] = np.clip(res[:, 1], 0, self.canvas_h - 1)
        return res.astype(np.int32)

    def warp_frame(self, frame: np.ndarray) -> np.ndarray:
        
        return cv2.warpPerspective(
            frame,
            self._H,
            (self.canvas_w, self.canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(30, 30, 30),
        )

    def pixel_to_meters(
        self,
        bev_xy: Tuple[int, int],
    ) -> Tuple[float, float]:

        return (
            bev_xy[0] / self.px_per_m_x,
            bev_xy[1] / self.px_per_m_y,
        )

    # ── Tích hợp với tracker ──────────────────────────────────────────────────

    def annotate_tracks(
        self,
        tracks: Dict,
        foot_fn=None,
    ) -> None:
        
        if foot_fn is None:
            foot_fn = _default_foot

        for frame_data in tracks.get("person", []):
            if not frame_data:
                continue

            ids: list = list(frame_data.keys())
            feet = np.array(
                [foot_fn(frame_data[tid]["bbox"]) for tid in ids],
                dtype=np.float32,
            )  # shape (N, 2)

            bev_pts = self.transform_points_batch(feet)  # (N, 2)

            for i, tid in enumerate(ids):
                frame_data[tid]["position_transformed"] = (
                    int(bev_pts[i, 0]),
                    int(bev_pts[i, 1]),
                )

    # ── Debug / Calibration ───────────────────────────────────────────────────

    def draw_calibration(
        self,
        frame: np.ndarray,
        src_points: Optional[List[List[float]]] = None,
        color: Tuple[int, int, int] = (0, 255, 80),
        radius: int = 8,
    ) -> np.ndarray:
        
        out = frame.copy()
        pts = src_points if src_points is not None else self._src_points_raw
        labels = ["TL", "TR", "BR", "BL"]

        for i, pt in enumerate(pts):
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(out, (cx, cy), radius, color, -1)
            cv2.putText(
                out,
                labels[i] if i < 4 else str(i),
                (cx + radius + 2, cy - radius),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
        return out

    def draw_bev_grid(
        self,
        canvas: Optional[np.ndarray] = None,
        grid_m: float = 1.0,
        line_color: Tuple[int, int, int] = (60, 60, 60),
        text_color: Tuple[int, int, int] = (120, 120, 120),
    ) -> np.ndarray:
       
        if canvas is None:
            canvas = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        else:
            canvas = canvas.copy()

        # Vẽ đường dọc
        x_m = grid_m
        while x_m < self.real_w_m:
            px = int(x_m * self.px_per_m_x)
            cv2.line(canvas, (px, 0), (px, self.canvas_h - 1), line_color, 1)
            cv2.putText(
                canvas, f"{x_m:.0f}m", (px + 2, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA,
            )
            x_m += grid_m

        # Vẽ đường ngang
        y_m = grid_m
        while y_m < self.real_h_m:
            py = int(y_m * self.px_per_m_y)
            cv2.line(canvas, (0, py), (self.canvas_w - 1, py), line_color, 1)
            cv2.putText(
                canvas, f"{y_m:.0f}m", (2, py - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA,
            )
            y_m += grid_m

        return canvas