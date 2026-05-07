import torch
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
import torchreid
from utils.bbox_utils import get_foot_position, measure_distance, get_cosine_similarity

class ReID:
    def __init__(self, 
                 model_name='osnet_x1_0',
                 similarity_threshold=0.75,
                 jacket_dist_thresh=30,      # Ngưỡng pixel cho logic cởi áo
                 jacket_time_thresh=48,      # 2 giây (24fps)
                 long_term_thresh=18000,     # 10 phút
                 device='cuda'):

        self.similarity_threshold = similarity_threshold
        self.jacket_dist_thresh = jacket_dist_thresh
        self.jacket_time_thresh = jacket_time_thresh
        self.long_term_thresh = long_term_thresh
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        # Load OSNet
        self.model = torchreid.models.build_model(
            name=model_name, num_classes=1000, pretrained=True
        ).to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Gallery: {id: {'features': [], 'last_pos': (x,y), 'last_frame': idx}}
        self.gallery = {}

    def extract_feature(self, frame, bbox):#Trích xuất feature vector từ bbox
        
        x1, y1, x2, y2 = map(int, bbox)
        if x2 - x1 < 20 or y2 - y1 < 40: return None # Bỏ qua box quá nhỏ
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0: return None
        
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feature = self.model(input_tensor)
            feature = F.normalize(feature, p=2, dim=1)
        return feature.cpu().numpy().flatten()

    def _match_with_gallery(self, current_feat, current_pos, current_frame):
        best_id = None
        max_sim = -1

        for track_id, data in self.gallery.items():
            time_gap = current_frame - data['last_frame']
            dist = measure_distance(current_pos, data['last_pos'])

            # 1. LOGIC CỞI ÁO KHOÁC (Jacket Logic)
            # Nếu vị trí chân gần như không đổi và biến mất rất ngắn
            if dist < self.jacket_dist_thresh and time_gap < self.jacket_time_thresh:
                return track_id, 1.0

            # 2. SO KHỚP NGOẠI HÌNH (ReID)
            sims = [get_cosine_similarity(current_feat, f) for f in data['features']]
            current_max_sim = max(sims) if sims else 0

            # Điều chỉnh ngưỡng cho Long-term (đi vệ sinh)
            effective_thresh = self.similarity_threshold
            if time_gap > 300: # Nếu đi quá 10 giây, siết chặt ngưỡng ReID
                effective_thresh += 0.05

            if current_max_sim > effective_thresh and current_max_sim > max_sim:
                # Kiểm tra ràng buộc không gian nếu không phải long-term
                if time_gap < 150 or dist < 500: 
                    max_sim = current_max_sim
                    best_id = track_id

        return best_id, max_sim

    def merge_tracks_offline(self, frames, tracks):
        """Xử lý gộp ID offline để debug"""
        person_tracks = tracks.get("person", [])
        new_person_tracks = [{} for _ in range(len(frames))]
        id_map = {} # Ánh xạ ID của tracker sang ID thực tế

        for frame_idx, frame_data in enumerate(person_tracks):
            frame = frames[frame_idx]
            for old_id, track_info in frame_data.items():
                bbox = track_info['bbox']
                foot_pos = get_foot_position(bbox)
                
                # Trích xuất đặc trưng
                feature = self.extract_feature(frame, bbox)
                
                actual_id = id_map.get(old_id)
                if actual_id is None:
                    # Thử khớp với gallery
                    matched_id, sim = self._match_with_gallery(feature, foot_pos, frame_idx)
                    if matched_id:
                        actual_id = matched_id
                        id_map[old_id] = actual_id
                    else:
                        actual_id = old_id
                        id_map[old_id] = actual_id
                
                # Cập nhật Gallery (lưu tối đa 10 đặc trưng/ID)
                if actual_id not in self.gallery:
                    self.gallery[actual_id] = {'features': [], 'last_pos': foot_pos, 'last_frame': frame_idx}
                
                if feature is not None and len(self.gallery[actual_id]['features']) < 10:
                    self.gallery[actual_id]['features'].append(feature)
                
                self.gallery[actual_id]['last_pos'] = foot_pos
                self.gallery[actual_id]['last_frame'] = frame_idx
                
                # Lưu kết quả mới
                new_person_tracks[frame_idx][actual_id] = {'bbox': bbox}

        return {"person": new_person_tracks}