from core.tracker import Tracker
from core.reid import ReID
from utils.video_utils import read_video, save_video
from utils.visualize_utils import draw_annotations
import os

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
    print("✅ Xử lý hoàn tất!")

if __name__ == "__main__":
    main()