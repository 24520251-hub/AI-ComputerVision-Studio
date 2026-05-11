from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Map tên colormap (từ settings.yaml) → hằng số cv2
_COLORMAP_LOOKUP: dict = {
    "JET":     cv2.COLORMAP_JET,
    "HOT":     cv2.COLORMAP_HOT,
    "INFERNO": cv2.COLORMAP_INFERNO,
    "PLASMA":  cv2.COLORMAP_PLASMA,
    "VIRIDIS": cv2.COLORMAP_VIRIDIS,
    "TURBO":   cv2.COLORMAP_TURBO,
}


# ──────────────────────────────────────────────────────────────────────────────
# Core class
# ──────────────────────────────────────────────────────────────────────────────

class HeatmapGenerator:

    def __init__(
        self,
        canvas_w: int = 600,
        canvas_h: int = 800,
        gaussian_ksize: int = 51,
        gaussian_sigma: float = 20.0,
        alpha: float = 0.60,
        decay_rate: float = 0.0,
        colormap: str = "JET",
        increment: float = 1.0,
    ) -> None:
        self.canvas_w = canvas_w
        self.canvas_h = canvas_h
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.decay_rate = float(np.clip(decay_rate, 0.0, 1.0))
        self.increment = float(increment)

        # Gaussian kernel – đảm bảo ksize là số lẻ ≥ 1
        ksize = int(gaussian_ksize)
        if ksize % 2 == 0:
            ksize += 1
        self.gaussian_ksize = ksize
        self.gaussian_sigma = float(gaussian_sigma)

        # Colormap
        cmap_key = str(colormap).upper()
        if cmap_key not in _COLORMAP_LOOKUP:
            logger.warning(
                "Colormap '%s' không được hỗ trợ, dùng JET.", colormap
            )
            cmap_key = "JET"
        self._colormap_cv2 = _COLORMAP_LOOKUP[cmap_key]

        # Accumulation map: float32 để không bị overflow khi cộng dồn
        self._accum: np.ndarray = np.zeros(
            (canvas_h, canvas_w), dtype=np.float32
        )

        # Thống kê
        self._total_points: int = 0
        self._frame_count: int = 0

        logger.info(
            "[HeatmapGenerator] canvas=%dx%d, ksize=%d, sigma=%.1f, "
            "alpha=%.2f, decay=%.4f, colormap=%s",
            canvas_w, canvas_h, ksize, gaussian_sigma,
            alpha, decay_rate, cmap_key,
        )

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        cfg: dict,
        canvas_w: int = 600,
        canvas_h: int = 800,
    ) -> "HeatmapGenerator":
        
        return cls(
            canvas_w=canvas_w,
            canvas_h=canvas_h,
            gaussian_ksize=int(cfg.get("gaussian_ksize", 51)),
            gaussian_sigma=float(cfg.get("gaussian_sigma", 20.0)),
            alpha=float(cfg.get("alpha", 0.60)),
            decay_rate=float(cfg.get("decay_rate", 0.0)),
            colormap=str(cfg.get("colormap", "JET")),
        )

    # ── Tích luỹ ──────────────────────────────────────────────────────────────

    def update(
        self,
        bev_points: Sequence[Tuple[int, int]],
    ) -> None:
        
        self._frame_count += 1

        # ── Decay (nếu bật) ────────────────────────────────────────────────
        if self.decay_rate > 0.0:
            self._accum *= (1.0 - self.decay_rate)

        # ── Tích luỹ các điểm mới ─────────────────────────────────────────
        for x, y in bev_points:
            cx = int(np.clip(x, 0, self.canvas_w - 1))
            cy = int(np.clip(y, 0, self.canvas_h - 1))
            self._accum[cy, cx] += self.increment
            self._total_points += 1

    def update_from_tracks(
        self,
        frame_data: dict,
    ) -> None:
        
        pts = [
            info["position_transformed"]
            for info in frame_data.values()
            if "position_transformed" in info
        ]
        self.update(pts)

    # ── Render ────────────────────────────────────────────────────────────────

    def _build_colored_heatmap(self) -> np.ndarray:
       
        # 1. Gaussian Blur để loang điểm dữ liệu
        blurred = cv2.GaussianBlur(
            self._accum,
            (self.gaussian_ksize, self.gaussian_ksize),
            self.gaussian_sigma,
        )

        # 2. Normalize về [0, 255]
        max_val = blurred.max()
        if max_val < 1e-6:
            # Chưa có dữ liệu gì
            return np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)

        normalized = (blurred / max_val * 255.0).astype(np.uint8)

        # 3. Áp colormap
        colored = cv2.applyColorMap(normalized, self._colormap_cv2)

        # 4. Mask vùng bằng 0 (không có ai) → trong suốt khi blend
        # Tạo mask: pixel nào có intensity > 0 mới giữ lại
        mask = normalized > 0
        colored[~mask] = 0  # vùng không có người → đen (sẽ không hiện khi blend)

        return colored

    def render(
        self,
        background: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        
        if background is None:
            bg = np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        else:
            # Resize an toàn nếu background không khớp canvas
            if background.shape[:2] != (self.canvas_h, self.canvas_w):
                bg = cv2.resize(background, (self.canvas_w, self.canvas_h))
            else:
                bg = background.copy()

        colored = self._build_colored_heatmap()

        # Alpha blend chỉ tại vùng có heatmap (mask > 0)
        mask_bool = colored.sum(axis=2) > 0  # (H, W) bool

        out = bg.copy()
        if mask_bool.any():
            out[mask_bool] = cv2.addWeighted(
                bg,      1.0 - self.alpha,
                colored, self.alpha,
                0,
            )[mask_bool]

        return out

    def render_standalone(self) -> np.ndarray:
        
        return self._build_colored_heatmap()

    def render_with_grid(
        self,
        background: Optional[np.ndarray] = None,
        transformer=None,
        grid_m: float = 1.0,
    ) -> np.ndarray:
        
        out = self.render(background)
        if transformer is not None:
            out = transformer.draw_bev_grid(out, grid_m=grid_m)
        return out

    # ── Snapshot / Persistence ────────────────────────────────────────────────

    def get_snapshot(self) -> np.ndarray:
        
        return self.render_standalone()

    def save_snapshot(self, path: str) -> None:
        """Lưu snapshot ra file ảnh."""
        img = self.get_snapshot()
        cv2.imwrite(path, img)
        logger.info("[HeatmapGenerator] Snapshot đã lưu → %s", path)

    def get_accumulation_map(self) -> np.ndarray:
        """Trả về bản copy của accumulation map (float32) để analytics."""
        return self._accum.copy()

    def get_hotspots(
        self,
        top_n: int = 5,
        min_distance_px: int = 30,
    ) -> List[Tuple[int, int, float]]:
        
        blurred = cv2.GaussianBlur(
            self._accum,
            (self.gaussian_ksize, self.gaussian_ksize),
            self.gaussian_sigma,
        )

        flat_idx = np.argsort(blurred.ravel())[::-1]  # giảm dần
        results: List[Tuple[int, int, float]] = []
        taken: List[Tuple[int, int]] = []

        for idx in flat_idx:
            if len(results) >= top_n:
                break
            val = float(blurred.ravel()[idx])
            if val < 1e-6:
                break
            y, x = divmod(int(idx), self.canvas_w)

            # NMS: bỏ nếu quá gần điểm đã chọn
            too_close = any(
                abs(x - tx) + abs(y - ty) < min_distance_px
                for tx, ty in taken
            )
            if not too_close:
                results.append((x, y, val))
                taken.append((x, y))

        return results

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Xoá toàn bộ dữ liệu tích luỹ (dùng đầu ca / session mới)."""
        self._accum[:] = 0.0
        self._total_points = 0
        self._frame_count = 0
        logger.info("[HeatmapGenerator] Accumulation map đã reset.")

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def total_points(self) -> int:
        """Tổng số điểm đã tích luỹ từ khi reset."""
        return self._total_points

    @property
    def frame_count(self) -> int:
        """Số frame đã gọi update()."""
        return self._frame_count

    @property
    def max_intensity(self) -> float:
        """Giá trị cực đại hiện tại trong accumulation map."""
        return float(self._accum.max())

    def __repr__(self) -> str:
        return (
            f"HeatmapGenerator("
            f"canvas={self.canvas_w}×{self.canvas_h}, "
            f"frames={self._frame_count}, "
            f"points={self._total_points}, "
            f"max_intensity={self.max_intensity:.1f})"
        )