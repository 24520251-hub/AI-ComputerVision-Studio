import numpy as np
def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_width_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return abs(x2-x1)

def measure_distance(bbox1,bbox2):
    return ((bbox1[0]-bbox2[0])**2+(bbox1[1]-bbox2[1])**2)**0.5

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)

def get_cosine_similarity(feat1, feat2):
    return np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))

def is_static(pos_history, threshold=20):
    if len(pos_history) < 10: return False
    # Tính độ lệch chuẩn của các vị trí gần đây
    std_dev = np.std([p[0] for p in pos_history]) + np.std([p[1] for p in pos_history])
    return std_dev < threshold