from core.tracker import Tracker
from core.reid import ReID
from utils.video_utils import read_video, save_video
from utils.visualize_utils import draw_annotations
import os

import sys
import logging
import numpy as np
import cv2
import yaml
from core.transform import HomographyTransformer
from processors.heatmap_gen import HeatmapGenerator

def main():
    # ==================== CONFIG =======================
    model_path = "models/best.pt"
    video_path = "assets/input_videos/test1.avi"
    stub_path = "assets/stubs/track_stub.pkl"
    
    # ==================== Read_video ==========================
    frames = read_video(video_path)
    if not frames: return

    # ===================== TRACKER ===========================
    tracker = Tracker(model_path)
    tracks = tracker.get_object_track(frames, read_from_stub=True, stub_path=stub_path)
    
    # ====================== ReID ===============================
    reid_manager = ReID(
        similarity_threshold=0.75,
        jacket_dist_thresh=25, # Pixel
        device='cuda'
    )
    refined_tracks = reid_manager.merge_tracks_offline(frames, tracks)
    
    # 5. Vẽ kết quả và lưu
    output_frames = draw_annotations(frames, refined_tracks)
    save_video(output_frames, output_dir="assets/output_videos")

    # ===================== HOMOGRAPHY ===========================
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    transformer = HomographyTransformer.from_config(
        cfg["homography"]
    )
 
    for frame_num, frame_data in enumerate(
        refined_tracks.get("person", [])
    ):
        for track_id, tdata in frame_data.items():
            bbox = tdata.get("bbox")
            if bbox is not None:
                # Foot position: bottom-center của bounding box
                foot_x = (bbox[0] + bbox[2]) / 2.0
                foot_y = float(bbox[3])
                bev_x, bev_y = transformer.transform_point(
                    (foot_x, foot_y)
                )
                tdata["position_transformed"] = (bev_x, bev_y)
 
    # ====================== HEATMAP ============================
    heatmap_gen = HeatmapGenerator.from_config(
        cfg["heatmap"]
    )
 
    for frame_num, frame_data in enumerate(
        refined_tracks.get("person", [])
    ):
        heatmap_gen.update_from_tracks(frame_data)
 
    bev_canvas = transformer.draw_bev_grid()
    heatmap_frame = heatmap_gen.render_with_grid(background=bev_canvas)
 
    os.makedirs("assets/output_videos", exist_ok=True)
    cv2.imwrite("assets/output_videos/heatmap_bev.png", heatmap_frame)
    print("✅ Heatmap BEV đã lưu: assets/output_videos/heatmap_bev.png")
 
    print("✅ Xử lý hoàn tất!")
    
if __name__ == "__main__":
    main()