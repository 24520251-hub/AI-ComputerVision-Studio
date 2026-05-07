studio_analytics/
├── assets/                 # Chứa video test, ảnh sơ đồ studio (bird-view)
│   ├── input_videos/
│   ├── output_videos/
│   └── stubs/              # chứa thông tin tracker, perspective_tranform để chạy nhanh hơn dễ debug
├── model/                  # Chứa file .pt của YOLOv11s và OSNet
│   └── best.pt             # yolo11s
├── config/
│   ├── settings.yaml       # Cấu hình ngưỡng IoU, Confidence, Ma trận Homography
│   └── zone_config.json    # Định nghĩa các vùng ROI trong studio
├── core/                   # "Trái tim" của hệ thống
│   ├── detector.py         # Wrapper cho YOLOv11
│   ├── tracker.py          # Cấu hình ByteTrack
│   ├── reid.py             # Logic trích xuất feature & Gallery matching
│   └── transform.py        # Xử lý Homography (Perspective to Bird-view)
├── database/
│   ├── db_manager.py       # Kết nối PostgreSQL/TimescaleDB
│   └── models.py           # Định nghĩa bảng (ID, Entry, Exit, Trajectory)
├── processors/
│   ├── heatmap_gen.py      # Thuật toán vẽ Heatmap (Gaussian Blur)
│   └── analytics.py        # Tính toán thời gian ở lại (Stay duration)
├── app/                    # Web Backend (FastAPI)
│   ├── main.py
│   └── api/                # Các endpoints trả về JSON heatmap & stats
├── utils/                  # Các hàm bổ trợ (vẽ box, xử lý video frame)
│   ├── visualization.py    # vẽ
│   ├── bbox_utils.py 
│   └── video_utils.py      # Đọc video, lưu video
│
├── main.py                 # Script chính chạy pipeline từ Camera/Video
└── requirements.txt

tracks["persons"][frame_num][track_id] = {
    "bbox": [x1, y1, x2, y2],
    "position": (x, y),                # Pixel (foot position) trên ảnh gốc
    "position_transformed": (x_2d, y_2d), # Tọa độ trên mặt bằng Studio (Bird-view)
    
    # --- Thông tin bổ sung quan trọng cho Studio ---
    "entry_timestamp": float,          # Thời điểm bắt đầu xuất hiện (giây thứ mấy)
    "stay_duration": float,            # Thời gian đã ở lại tính đến frame này (giây)
    "features": np.array,              # Feature vector từ OSNet (dùng để ReID)
    "is_active": bool                  # Người này còn trong khung hình hay không
}