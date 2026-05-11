"""
quicktest.py
============
Test nhanh cho HomographyTransformer (transform.py) và HeatmapGenerator (heatmap_gen.py).
Không cần camera hay video thật — dùng dữ liệu giả lập.

Cách chạy:
    python quicktest.py

Output:
    - Console log từng bước
    - heatmap_snapshot.png  (heatmap thuần túy)
    - heatmap_blended.png   (blend lên nền BEV giả)
    - heatmap_grid.png      (có lưới mét)
"""

import sys
import os
import logging
import numpy as np
import cv2

# ── Đảm bảo import được 2 module cần test ────────────────────────────────────
# Nếu chạy từ thư mục khác, thêm đường dẫn tới nơi chứa file
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from core.transform import HomographyTransformer
from processors.heatmap_gen import HeatmapGenerator

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("quicktest")

# ═════════════════════════════════════════════════════════════════════════════
# Cấu hình giả lập
# ═════════════════════════════════════════════════════════════════════════════

# Giả sử camera nhìn vào phòng 8×10m, canvas BEV 600×800 px
CANVAS_W, CANVAS_H = 600, 800

# 4 góc tương ứng: góc ảnh camera (pixel) → góc canvas BEV
# Giả lập như camera hơi nghiêng: trapezoid → rectangle
SRC_POINTS = [
    [150, 100],   # TL (camera)
    [490, 100],   # TR
    [580, 460],   # BR
    [60,  460],   # BL
]
DST_POINTS = [
    [0,          0],           # TL (BEV)
    [CANVAS_W-1, 0],           # TR
    [CANVAS_W-1, CANVAS_H-1],  # BR
    [0,          CANVAS_H-1],  # BL
]

# Fake tracks: 3 người, 30 frame, mỗi người có quỹ đạo riêng
N_FRAMES  = 30
N_PERSONS = 3


def make_fake_tracks() -> dict:
    """Tạo tracks dict giả theo format của Tracker.get_object_track()."""
    rng = np.random.default_rng(42)

    # Điểm bắt đầu (pixel camera) cho mỗi người
    starts = [(200, 150), (320, 250), (450, 380)]
    # Tốc độ di chuyển mỗi frame
    speeds = [(3, 4), (-2, 5), (4, -3)]

    tracks: dict = {"person": []}
    for f in range(N_FRAMES):
        frame_data: dict = {}
        for tid in range(N_PERSONS):
            sx, sy = starts[tid]
            dx, dy = speeds[tid]
            x = sx + dx * f + rng.integers(-5, 5)
            y = sy + dy * f + rng.integers(-5, 5)
            # Clamp trong khung ảnh giả (640×480)
            x = int(np.clip(x, 50, 590))
            y = int(np.clip(y, 50, 430))
            frame_data[tid] = {
                "bbox": [x - 20, y - 60, x + 20, y],   # bbox người (giả)
                "score": 0.92,
            }
        tracks["person"].append(frame_data)
    return tracks


# ═════════════════════════════════════════════════════════════════════════════
# TEST 1 — HomographyTransformer
# ═════════════════════════════════════════════════════════════════════════════

def test_transformer() -> HomographyTransformer:
    logger.info("=" * 60)
    logger.info("TEST 1: HomographyTransformer")
    logger.info("=" * 60)

    # Khởi tạo
    tfm = HomographyTransformer(
        src_points=SRC_POINTS,
        dst_points=DST_POINTS,
        canvas_w=CANVAS_W,
        canvas_h=CANVAS_H,
        real_w_m=8.0,
        real_h_m=10.0,
    )
    logger.info("Khởi tạo OK: %s", tfm)

    # Test transform_point (1 điểm)
    cam_pt = (320, 280)
    bev_pt = tfm.transform_point(cam_pt)
    logger.info("transform_point(%s) → BEV %s", cam_pt, bev_pt)
    assert 0 <= bev_pt[0] < CANVAS_W, "BEV x ngoài canvas!"
    assert 0 <= bev_pt[1] < CANVAS_H, "BEV y ngoài canvas!"

    # Test transform_points_batch (nhiều điểm)
    pts = np.array([[150, 100], [320, 250], [490, 380]], dtype=np.float32)
    bev_batch = tfm.transform_points_batch(pts)
    logger.info("transform_points_batch:\n  camera=%s\n  bev   =%s", pts.tolist(), bev_batch.tolist())
    assert bev_batch.shape == (3, 2), "Shape batch sai!"

    # Test pixel_to_meters
    bev_m = tfm.pixel_to_meters(bev_pt)
    logger.info("pixel_to_meters(%s) → (%.2fm, %.2fm)", bev_pt, *bev_m)

    # Test warp_frame với ảnh giả
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(fake_frame, (150, 100), (490, 460), (0, 200, 100), 3)
    bev_frame = tfm.warp_frame(fake_frame)
    assert bev_frame.shape == (CANVAS_H, CANVAS_W, 3), "warp_frame shape sai!"
    logger.info("warp_frame OK: output shape = %s", bev_frame.shape)

    # Test draw_calibration
    cal_frame = tfm.draw_calibration(fake_frame)
    assert cal_frame.shape == fake_frame.shape, "draw_calibration shape sai!"
    logger.info("draw_calibration OK")

    # Test annotate_tracks
    tracks = make_fake_tracks()
    tfm.annotate_tracks(tracks)
    # Kiểm tra tất cả frame và track đều có position_transformed
    missing = 0
    for f_idx, frame_data in enumerate(tracks["person"]):
        for tid, info in frame_data.items():
            if "position_transformed" not in info:
                missing += 1
    assert missing == 0, f"{missing} track thiếu position_transformed!"
    sample = tracks["person"][0][0]["position_transformed"]
    logger.info(
        "annotate_tracks OK: %d frames × %d persons — sample[frame0, tid0] = %s",
        N_FRAMES, N_PERSONS, sample,
    )

    logger.info("✓ TEST 1 PASSED\n")
    return tfm


# ═════════════════════════════════════════════════════════════════════════════
# TEST 2 — HeatmapGenerator
# ═════════════════════════════════════════════════════════════════════════════

def test_heatmap(tfm: HomographyTransformer) -> HeatmapGenerator:
    logger.info("=" * 60)
    logger.info("TEST 2: HeatmapGenerator")
    logger.info("=" * 60)

    gen = HeatmapGenerator(
        canvas_w=CANVAS_W,
        canvas_h=CANVAS_H,
        gaussian_ksize=51,
        gaussian_sigma=20.0,
        alpha=0.60,
        decay_rate=0.0,
        colormap="JET",
        increment=1.0,
    )
    logger.info("Khởi tạo OK: %r", gen)

    # --- 2a. from_config factory ---
    cfg = {
        "gaussian_ksize": 41,
        "gaussian_sigma": 15.0,
        "alpha": 0.55,
        "decay_rate": 0.01,
        "colormap": "TURBO",
    }
    gen2 = HeatmapGenerator.from_config(cfg, CANVAS_W, CANVAS_H)
    logger.info("from_config OK: %r", gen2)

    # --- 2b. Đẩy tracks đã annotated vào heatmap ---
    tracks = make_fake_tracks()
    tfm.annotate_tracks(tracks)

    for frame_data in tracks["person"]:
        gen.update_from_tracks(frame_data)

    logger.info(
        "Sau %d frames: total_points=%d, max_intensity=%.1f",
        gen.frame_count, gen.total_points, gen.max_intensity,
    )
    assert gen.frame_count == N_FRAMES
    assert gen.total_points == N_FRAMES * N_PERSONS

    # --- 2c. Render standalone (heatmap thuần) ---
    snap = gen.render_standalone()
    assert snap.shape == (CANVAS_H, CANVAS_W, 3)
    assert snap.max() > 0, "Snapshot hoàn toàn đen — dữ liệu chưa tích luỹ?"
    logger.info("render_standalone OK: max pixel = %d", snap.max())

    # --- 2d. Render blend lên background giả ---
    bg = np.full((CANVAS_H, CANVAS_W, 3), 40, dtype=np.uint8)  # nền xám tối
    blended = gen.render(bg)
    assert blended.shape == (CANVAS_H, CANVAS_W, 3)
    logger.info("render(bg) OK")

    # --- 2e. Render với lưới mét ---
    grid_img = gen.render_with_grid(bg, transformer=tfm, grid_m=1.0)
    assert grid_img.shape == (CANVAS_H, CANVAS_W, 3)
    logger.info("render_with_grid OK")

    # --- 2f. get_accumulation_map ---
    accum = gen.get_accumulation_map()
    assert accum.shape == (CANVAS_H, CANVAS_W)
    assert accum.dtype == np.float32
    logger.info("get_accumulation_map OK: max=%.2f", accum.max())

    # --- 2g. Hotspots ---
    hotspots = gen.get_hotspots(top_n=3, min_distance_px=30)
    logger.info("get_hotspots(top_n=3) → %s", hotspots)
    assert len(hotspots) <= 3
    for x, y, val in hotspots:
        assert 0 <= x < CANVAS_W
        assert 0 <= y < CANVAS_H
        assert val > 0

    # --- 2h. Decay test ---
    gen2_decay = HeatmapGenerator(
        canvas_w=CANVAS_W, canvas_h=CANVAS_H, decay_rate=0.1
    )
    gen2_decay.update([(300, 400)] * 10)
    before = gen2_decay.max_intensity
    gen2_decay.update([])  # frame rỗng → chỉ decay
    after  = gen2_decay.max_intensity
    assert after < before, "Decay không hoạt động!"
    logger.info("Decay OK: %.4f → %.4f", before, after)

    # --- 2i. Reset ---
    gen.reset()
    assert gen.total_points == 0
    assert gen.frame_count  == 0
    assert gen.max_intensity == 0.0
    logger.info("reset OK")

    # ── Lưu ảnh output ───────────────────────────────────────────────────────
    # Tái tạo heatmap sau khi reset để lưu ảnh
    for frame_data in tracks["person"]:
        gen.update_from_tracks(frame_data)

    out_dir = os.path.dirname(os.path.abspath(__file__))

    cv2.imwrite(os.path.join(out_dir, "heatmap_snapshot.png"), gen.get_snapshot())
    logger.info("Đã lưu: heatmap_snapshot.png")

    cv2.imwrite(os.path.join(out_dir, "heatmap_blended.png"), gen.render(bg))
    logger.info("Đã lưu: heatmap_blended.png")

    cv2.imwrite(os.path.join(out_dir, "heatmap_grid.png"),
                gen.render_with_grid(bg, transformer=tfm, grid_m=1.0))
    logger.info("Đã lưu: heatmap_grid.png")

    logger.info("✓ TEST 2 PASSED\n")
    return gen


# ═════════════════════════════════════════════════════════════════════════════
# TEST 3 — Pipeline tích hợp (end-to-end giả lập 1 session)
# ═════════════════════════════════════════════════════════════════════════════

def test_pipeline():
    logger.info("=" * 60)
    logger.info("TEST 3: Pipeline end-to-end")
    logger.info("=" * 60)

    tfm = HomographyTransformer(
        src_points=SRC_POINTS, dst_points=DST_POINTS,
        canvas_w=CANVAS_W, canvas_h=CANVAS_H,
    )
    gen = HeatmapGenerator(
        canvas_w=CANVAS_W, canvas_h=CANVAS_H,
        decay_rate=0.005, alpha=0.65, colormap="INFERNO",
    )

    tracks = make_fake_tracks()
    tfm.annotate_tracks(tracks)

    for frame_num, frame_data in enumerate(tracks["person"]):
        # Lấy BEV points
        bev_pts = [
            info["position_transformed"]
            for info in frame_data.values()
            if "position_transformed" in info
        ]
        gen.update(bev_pts)

        # Render mỗi frame (như trong main.py)
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bev_bg     = tfm.warp_frame(fake_frame)
        output     = gen.render(bev_bg)
        assert output.shape == (CANVAS_H, CANVAS_W, 3)

    logger.info("Pipeline: %r", gen)
    assert gen.frame_count == N_FRAMES
    logger.info("✓ TEST 3 PASSED\n")


# ═════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═════════════════════════════════════════════════════════════════════════════

def test_edge_cases():
    logger.info("=" * 60)
    logger.info("TEST 4: Edge cases")
    logger.info("=" * 60)

    gen = HeatmapGenerator(canvas_w=CANVAS_W, canvas_h=CANVAS_H)

    # Render khi chưa có điểm nào
    empty = gen.render()
    assert empty.max() == 0, "Canvas rỗng phải tất cả đen"
    logger.info("Render khi chưa có điểm: OK (all zeros)")

    # update với list rỗng
    gen.update([])
    assert gen.total_points == 0
    logger.info("update([]) OK")

    # Điểm ngoài biên canvas → phải được clamp
    gen.update([(-100, -200), (9999, 9999)])
    assert gen.total_points == 2  # vẫn tích luỹ (sau khi clamp)
    logger.info("Clamp out-of-bound points OK")

    # Colormap không hợp lệ → fallback JET
    gen_bad = HeatmapGenerator(colormap="UNICORN")
    logger.info("Invalid colormap fallback OK")

    # transform_points_batch với 0 điểm
    tfm = HomographyTransformer(
        src_points=SRC_POINTS, dst_points=DST_POINTS,
        canvas_w=CANVAS_W, canvas_h=CANVAS_H,
    )
    result = tfm.transform_points_batch(np.empty((0, 2)))
    assert result.shape == (0, 2)
    logger.info("transform_points_batch(empty) OK")

    logger.info("✓ TEST 4 PASSED\n")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info(">>> Bắt đầu quicktest <<<\n")

    try:
        tfm = test_transformer()
        gen = test_heatmap(tfm)
        test_pipeline()
        test_edge_cases()
        logger.info("=" * 60)
        logger.info("✅  TẤT CẢ TEST PASSED")
        logger.info("=" * 60)
    except AssertionError as e:
        logger.error("❌ ASSERTION FAILED: %s", e)
        sys.exit(1)
    except Exception as e:
        logger.exception("❌ EXCEPTION: %s", e)
        sys.exit(1)