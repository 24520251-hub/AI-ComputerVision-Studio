"""
processors/heatmap_gen.py
--------------------------
Tạo bản đồ nhiệt (Heatmap) từ dữ liệu vị trí Bird's-eye View.

Pipeline:
  1. Tích lũy điểm → accumulation_map (float32)
  2. Gaussian Blur làm mịn
  3. Normalize [0, 255]
  4. COLORMAP_JET phủ màu
  5. Alpha blend lên canvas BEV

Tối ưu CPU:
  - Dùng numpy vectorized thay vì vòng lặp for
  - Gaussian Blur chỉ chạy khi render (không mỗi frame)
  - Accumulation map tích lũy liên tục, render theo yêu cầu
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Optional
import yaml


class HeatmapGenerator:
    """
    Quản lý accumulation map và render heatmap Bird's-eye View.

    Parameters
    ----------
    floor_w, floor_h : int
        Kích thước canvas (phải khớp với HomographyTransformer).
    gaussian_ksize : int
        Kích thước kernel Gaussian Blur (phải là số lẻ, ví dụ 51).
    gaussian_sigma : float
        Độ lệch chuẩn Gaussian (kiểm soát độ loang).
    alpha : float
        Độ trong suốt khi blend heatmap lên nền BEV (0=trong suốt, 1=đục).
    decay_rate : float
        Tỉ lệ giảm dần theo thời gian (0 = không decay, 0.01 = giảm chậm).
        Hữu ích khi muốn heatmap phản ánh hoạt động gần đây hơn.
    """

    def __init__(
        self,
        floor_w: int = 800,
        floor_h: int = 600,
        gaussian_ksize: int = 51,
        gaussian_sigma: float = 20.0,
        alpha: float = 0.6,
        decay_rate: float = 0.0,
    ):
        self.floor_w = floor_w
        self.floor_h = floor_h
        self.alpha = alpha
        self.decay_rate = decay_rate

        # Đảm bảo kernel size lẻ
        if gaussian_ksize % 2 == 0:
            gaussian_ksize += 1
        self.gaussian_ksize = gaussian_ksize
        self.gaussian_sigma = gaussian_sigma

        # Accumulation map: float32 để tích lũy không bị tràn
        self.accumulation_map = np.zeros((floor_h, floor_w), dtype=np.float32)
        self._frame_count = 0

    @classmethod
    def from_config(cls, config_path: str = "config/settings.yaml") -> "HeatmapGenerator":
        """Khởi tạo từ file settings.yaml."""
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        hcfg = cfg.get("homography", {})
        heat_cfg = cfg.get("heatmap", {})
        return cls(
            floor_w=hcfg.get("floor_canvas_w", 800),
            floor_h=hcfg.get("floor_canvas_h", 600),
            gaussian_ksize=heat_cfg.get("gaussian_ksize", 51),
            gaussian_sigma=heat_cfg.get("gaussian_sigma", 20.0),
            alpha=heat_cfg.get("alpha", 0.6),
            decay_rate=heat_cfg.get("decay_rate", 0.0),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Tích lũy dữ liệu
    # ──────────────────────────────────────────────────────────────────────

    def accumulate(self, bev_points: np.ndarray, weights: Optional[np.ndarray] = None):
        """
        Tích lũy tọa độ BEV vào accumulation map.

        Parameters
        ----------
        bev_points : np.ndarray, shape (N, 2)
            Tọa độ (x, y) trên canvas BEV — output của HomographyTransformer.
        weights : np.ndarray, shape (N,) hoặc None
            Trọng số tùy chọn (ví dụ: stay_duration tính bằng giây).
            None → mỗi điểm cộng 1.
        """
        if len(bev_points) == 0:
            return

        bev_points = np.round(bev_points).astype(np.int32)

        # Clip vào biên canvas
        xs = np.clip(bev_points[:, 0], 0, self.floor_w - 1)
        ys = np.clip(bev_points[:, 1], 0, self.floor_h - 1)

        if weights is None:
            weights = np.ones(len(xs), dtype=np.float32)
        else:
            weights = np.asarray(weights, dtype=np.float32)

        # Vectorized: dùng np.add.at để tránh vòng for
        np.add.at(self.accumulation_map, (ys, xs), weights)

        # Decay (optional)
        if self.decay_rate > 0:
            self.accumulation_map *= (1.0 - self.decay_rate)

        self._frame_count += 1

    def accumulate_from_tracks(self, tracks: dict, frame_num: int):
        """
        Wrapper tiện lợi: nhận tracks dict từ hệ thống tracker.

        tracks["persons"][frame_num][track_id] = {
            "position_transformed": (x_bev, y_bev),
            "stay_duration": float,  # giây
            ...
        }
        """
        if "persons" not in tracks:
            return
        frame_data = tracks["persons"].get(frame_num, {})
        if not frame_data:
            return

        pts, weights = [], []
        for tid, tdata in frame_data.items():
            pos = tdata.get("position_transformed")
            if pos is not None:
                pts.append(pos)
                weights.append(tdata.get("stay_duration", 1.0))

        if pts:
            self.accumulate(np.array(pts, dtype=np.float32), np.array(weights))

    # ──────────────────────────────────────────────────────────────────────
    # Render heatmap
    # ──────────────────────────────────────────────────────────────────────

    def render(self, background: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render heatmap đầy đủ (Blur + ColorMap + Blend).

        Parameters
        ----------
        background : np.ndarray (H, W, 3) hoặc None
            Ảnh nền BEV (canvas sơ đồ studio). None → nền xám.

        Returns
        -------
        np.ndarray (H, W, 3) BGR — heatmap đã blend sẵn để hiển thị.
        """
        # ── Nền ────────────────────────────────────────────────────────────
        if background is None:
            bg = np.full((self.floor_h, self.floor_w, 3), 240, dtype=np.uint8)
        else:
            bg = cv2.resize(background, (self.floor_w, self.floor_h))

        # ── Gaussian Blur (làm loang điểm nhiệt) ──────────────────────────
        blurred = cv2.GaussianBlur(
            self.accumulation_map,
            (self.gaussian_ksize, self.gaussian_ksize),
            self.gaussian_sigma,
        )

        # ── Normalize về [0, 255] ──────────────────────────────────────────
        max_val = blurred.max()
        if max_val < 1e-6:
            # Chưa có dữ liệu → trả về nền
            return bg.copy()

        normalized = (blurred / max_val * 255).astype(np.uint8)

        # ── Phủ màu COLORMAP_JET ───────────────────────────────────────────
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # ── Mask: chỉ blend những vùng có dữ liệu ─────────────────────────
        mask = (normalized > 0).astype(np.float32)[:, :, np.newaxis]

        # ── Alpha blend ────────────────────────────────────────────────────
        result = bg.copy().astype(np.float32)
        result = result * (1 - mask * self.alpha) + colored.astype(np.float32) * (mask * self.alpha)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def render_with_stats(self, background: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Render heatmap + overlay thống kê cơ bản (peak zone, total frames).
        """
        canvas = self.render(background)
        stats = self.get_stats()

        # Overlay text
        y = 25
        lines = [
            f"Frames: {self._frame_count}",
            f"Peak intensity: {stats['peak_intensity']:.0f}",
            f"Active area: {stats['active_area_pct']:.1f}%",
        ]
        for line in lines:
            cv2.putText(canvas, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
            y += 22

        # Vẽ colorbar
        self._draw_colorbar(canvas)

        return canvas

    # ──────────────────────────────────────────────────────────────────────
    # Thống kê
    # ──────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Trả về dict thống kê cơ bản từ accumulation map."""
        total_pixels = self.floor_w * self.floor_h
        active_pixels = int((self.accumulation_map > 0).sum())
        peak_y, peak_x = np.unravel_index(
            self.accumulation_map.argmax(), self.accumulation_map.shape
        )
        return {
            "peak_intensity": float(self.accumulation_map.max()),
            "mean_intensity": float(self.accumulation_map[self.accumulation_map > 0].mean())
            if active_pixels > 0 else 0.0,
            "active_area_pct": active_pixels / total_pixels * 100,
            "peak_bev_xy": (int(peak_x), int(peak_y)),
            "total_frames": self._frame_count,
        }

    def get_top_zones(self, n: int = 3, zone_radius: int = 40) -> list[dict]:
        """
        Tìm N vùng nóng nhất (non-maximum suppression đơn giản).
        Dùng để xác định hotspot trong studio.
        """
        blurred = cv2.GaussianBlur(
            self.accumulation_map,
            (self.gaussian_ksize, self.gaussian_ksize),
            self.gaussian_sigma,
        )
        tmp = blurred.copy()
        zones = []
        for _ in range(n):
            peak_val = tmp.max()
            if peak_val < 1e-6:
                break
            peak_y, peak_x = np.unravel_index(tmp.argmax(), tmp.shape)
            zones.append({
                "rank": len(zones) + 1,
                "bev_xy": (int(peak_x), int(peak_y)),
                "intensity": float(peak_val),
            })
            # Suppress vùng xung quanh
            y1 = max(0, peak_y - zone_radius)
            y2 = min(self.floor_h, peak_y + zone_radius)
            x1 = max(0, peak_x - zone_radius)
            x2 = min(self.floor_w, peak_x + zone_radius)
            tmp[y1:y2, x1:x2] = 0

        return zones

    def reset(self):
        """Xóa accumulation map (dùng khi bắt đầu session mới)."""
        self.accumulation_map[:] = 0
        self._frame_count = 0

    def save_accumulation(self, path: str):
        """Lưu accumulation map dạng .npy để debug hoặc resume."""
        np.save(path, self.accumulation_map)

    def load_accumulation(self, path: str):
        """Load lại accumulation map đã lưu."""
        self.accumulation_map = np.load(path).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _draw_colorbar(self, canvas: np.ndarray):
        """Vẽ thanh màu chú giải (Low→High) góc phải dưới."""
        bar_h, bar_w = 120, 18
        x0 = self.floor_w - bar_w - 10
        y0 = self.floor_h - bar_h - 30

        # Tạo dải màu gradient
        gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
        colored_bar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        canvas[y0:y0+bar_h, x0:x0+bar_w] = colored_bar
        cv2.rectangle(canvas, (x0, y0), (x0+bar_w, y0+bar_h), (80, 80, 80), 1)

        cv2.putText(canvas, "High", (x0 - 5, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(canvas, "Low", (x0 - 3, y0 + bar_h + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# ──────────────────────────────────────────────────────────────────────────────
# Quick test (chạy: python processors/heatmap_gen.py)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    gen = HeatmapGenerator(floor_w=800, floor_h=600,
                           gaussian_ksize=61, gaussian_sigma=25.0, alpha=0.65)

    # Giả lập người đứng ở các vị trí khác nhau
    rng = np.random.default_rng(42)
    
    # Zone A: góc trái (quầy lễ tân) — nhiều người
    zone_a = rng.normal(loc=[150, 120], scale=[30, 25], size=(300, 2))
    # Zone B: giữa phòng — trung bình
    zone_b = rng.normal(loc=[400, 300], scale=[50, 40], size=(150, 2))
    # Zone C: góc phải (khu thiết bị) — ít người
    zone_c = rng.normal(loc=[650, 480], scale=[40, 30], size=(60, 2))

    all_pts = np.vstack([zone_a, zone_b, zone_c]).astype(np.float32)
    gen.accumulate(all_pts)

    # Render với stats
    result = gen.render_with_stats()

    # Top zones
    zones = gen.get_top_zones(n=3)
    for z in zones:
        cx, cy = z["bev_xy"]
        cv2.circle(result, (cx, cy), 15, (255, 255, 255), 2)
        cv2.putText(result, f"#{z['rank']}", (cx+17, cy+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        print(f"  Zone #{z['rank']}: BEV({cx},{cy}) | Intensity={z['intensity']:.0f}")

    import os
    os.makedirs("assets", exist_ok=True)
    out = "assets/debug_heatmap_test.png"
    cv2.imwrite(out, result)
    print(f"\n[OK] Heatmap test saved: {out}")
    print(f"Stats: {gen.get_stats()}")
