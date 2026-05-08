#==== Thêm ======
"""
core/transform.py
-----------------
Xử lý Homography tĩnh: chuyển đổi tọa độ pixel (camera góc nghiêng)
sang tọa độ thực tế Bird's-eye View (mặt bằng 2D studio).

Sử dụng: khởi tạo một lần với H matrix từ settings.yaml, 
gọi transform() cho mỗi batch điểm chân (foot positions).
"""

import numpy as np
import cv2
import yaml
from pathlib import Path
from typing import Union


class HomographyTransformer:
    """
    Chuyển đổi tọa độ từ image plane → Bird's-eye view plane.

    Attributes
    ----------
    H : np.ndarray (3x3)
        Ma trận Homography (image → floor).
    H_inv : np.ndarray (3x3)
        Ma trận Homography ngược (floor → image), dùng để vẽ overlay.
    floor_w, floor_h : int
        Kích thước canvas Bird's-eye view (pixel đại diện cho m² thực tế).
    scale_x, scale_y : float
        Tỉ lệ pixel/m trên canvas Bird's-eye view.
    """

    def __init__(self, config_path: Union[str, Path] = "config/settings.yaml"):
        config_path = Path(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        hcfg = cfg["homography"]

        # ── Tọa độ 4 góc trên ảnh camera (pixel) ──────────────────────────
        self.src_points = np.array(hcfg["src_points"], dtype=np.float32)

        # ── Tọa độ 4 góc tương ứng trên mặt bằng 2D (pixel canvas) ────────
        self.dst_points = np.array(hcfg["dst_points"], dtype=np.float32)

        # ── Kích thước canvas Bird's-eye view ──────────────────────────────
        self.floor_w = hcfg.get("floor_canvas_w", 800)
        self.floor_h = hcfg.get("floor_canvas_h", 600)

        # ── Kích thước phòng thực tế (m) để tính scale ─────────────────────
        self.real_w_m = hcfg.get("real_width_m", 10.0)   # chiều ngang studio
        self.real_h_m = hcfg.get("real_height_m", 8.0)   # chiều sâu studio
        self.scale_x  = self.floor_w / self.real_w_m      # pixel / m
        self.scale_y  = self.floor_h / self.real_h_m

        # ── Tính ma trận H ─────────────────────────────────────────────────
        self.H, status = cv2.findHomography(self.src_points, self.dst_points)
        if self.H is None:
            raise ValueError("cv2.findHomography thất bại. Kiểm tra lại src/dst_points.")

        self.H_inv = np.linalg.inv(self.H)

        print(f"[HomographyTransformer] H matrix:\n{self.H}")
        print(f"[HomographyTransformer] Canvas: {self.floor_w}x{self.floor_h}px "
              f"| Studio: {self.real_w_m}m x {self.real_h_m}m")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def transform(self, points: np.ndarray) -> np.ndarray:
        """
        Chuyển nhiều điểm image-plane → Bird's-eye view (vectorized).

        Parameters
        ----------
        points : np.ndarray, shape (N, 2)
            Mảng tọa độ pixel (u, v) — thường là foot-center của bbox.

        Returns
        -------
        np.ndarray, shape (N, 2)
            Tọa độ tương ứng trên canvas Bird's-eye view (x_bev, y_bev).
        """
        if points.ndim == 1:
            points = points[np.newaxis, :]   # (1, 2)

        # Homogeneous: (N, 2) → (N, 3)
        ones = np.ones((len(points), 1), dtype=np.float64)
        pts_h = np.hstack([points.astype(np.float64), ones])  # (N, 3)

        # P_real = H · P_image  (broadcast)
        transformed = (self.H @ pts_h.T).T                     # (N, 3)

        # Chia cho w (homogeneous → Cartesian)
        w = transformed[:, 2:3]
        w = np.where(np.abs(w) < 1e-8, 1e-8, w)               # tránh /0
        result = transformed[:, :2] / w

        # Clip về trong canvas
        result[:, 0] = np.clip(result[:, 0], 0, self.floor_w - 1)
        result[:, 1] = np.clip(result[:, 1], 0, self.floor_h - 1)

        return result.astype(np.float32)

    def transform_single(self, u: float, v: float) -> tuple[float, float]:
        """Shortcut cho 1 điểm duy nhất."""
        res = self.transform(np.array([[u, v]]))
        return float(res[0, 0]), float(res[0, 1])

    def to_real_meters(self, bev_point: np.ndarray) -> np.ndarray:
        """
        Chuyển tọa độ BEV (pixel canvas) → mét thực tế.
        Hữu ích khi lưu DB hoặc debug.
        """
        bev_point = np.atleast_2d(bev_point).astype(np.float32)
        real = np.empty_like(bev_point)
        real[:, 0] = bev_point[:, 0] / self.scale_x
        real[:, 1] = bev_point[:, 1] / self.scale_y
        return real

    def get_floor_canvas(self, floor_img: np.ndarray = None) -> np.ndarray:
        """
        Trả về canvas BEV trắng (hoặc ảnh sơ đồ studio nếu truyền vào).
        Canvas này dùng làm nền vẽ heatmap / overlay.
        """
        if floor_img is not None:
            return cv2.resize(floor_img, (self.floor_w, self.floor_h))
        canvas = np.zeros((self.floor_h, self.floor_w, 3), dtype=np.uint8)
        canvas[:] = (240, 240, 240)  # xám nhạt
        self._draw_grid(canvas)
        return canvas

    def warp_frame_to_bev(self, frame: np.ndarray) -> np.ndarray:
        """
        Warp toàn bộ frame camera → Bird's-eye view (dùng để debug / demo).
        Không dùng trong production (chậm, chỉ cần transform tọa độ điểm).
        """
        return cv2.warpPerspective(frame, self.H, (self.floor_w, self.floor_h))

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _draw_grid(self, canvas: np.ndarray, step_m: float = 1.0):
        """Vẽ lưới 1m×1m lên canvas BEV (debug)."""
        color = (200, 200, 200)
        # Dọc
        for x_m in np.arange(0, self.real_w_m + step_m, step_m):
            x_px = int(x_m * self.scale_x)
            cv2.line(canvas, (x_px, 0), (x_px, self.floor_h), color, 1)
        # Ngang
        for y_m in np.arange(0, self.real_h_m + step_m, step_m):
            y_px = int(y_m * self.scale_y)
            cv2.line(canvas, (0, y_px), (self.floor_w, y_px), color, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Quick test (chạy: python core/transform.py)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    transformer = HomographyTransformer("config/settings.yaml")

    # Giả lập 5 điểm chân từ bbox detector
    fake_feet = np.array([
        [320, 480],
        [640, 500],
        [200, 350],
        [900, 420],
        [500, 600],
    ], dtype=np.float32)

    bev_pts = transformer.transform(fake_feet)
    real_pts = transformer.to_real_meters(bev_pts)

    print("\n=== Kết quả transform ===")
    for i, (img_pt, bev_pt, real_pt) in enumerate(zip(fake_feet, bev_pts, real_pts)):
        print(f"  [{i}] Image({img_pt[0]:.0f},{img_pt[1]:.0f}) "
              f"→ BEV({bev_pt[0]:.1f},{bev_pt[1]:.1f}) "
              f"→ Real({real_pt[0]:.2f}m, {real_pt[1]:.2f}m)")

    # Vẽ canvas BEV và các điểm
    canvas = transformer.get_floor_canvas()
    for pt in bev_pts:
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 8, (0, 100, 255), -1)

    out_path = "assets/debug_bev_test.png"
    os.makedirs("assets", exist_ok=True)
    cv2.imwrite(out_path, canvas)
    print(f"\n[OK] Đã lưu ảnh BEV test: {out_path}")
