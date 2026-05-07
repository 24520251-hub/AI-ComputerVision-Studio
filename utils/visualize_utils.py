import pickle
import os
import sys
sys.path.append("../")
import cv2
from .bbox_utils import get_foot_position , get_width_of_bbox

def draw_ellipse(frame,bbox,id):
    x,y = get_foot_position(bbox)
    width = get_width_of_bbox(bbox)
    cv2.ellipse(
        frame,
        center=(x,y),
        axes=(int(width),int(0.35*width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=(0,255,0),
        thickness=2,
        lineType=cv2.LINE_4)
    
    rectangle_width = 40
    rectangle_height = 20
    x1_rect = x - rectangle_width//2
    x2_rect = x + rectangle_width//2
    y1_rect = (y - rectangle_height//2) + 15
    y2_rect = (y + rectangle_height//2) + 15
        
    if id is not None:
        cv2.rectangle(frame,
                      (int(x1_rect),int(y1_rect)),
                      (int(x2_rect),int(y2_rect)),
                      (255,0,255),
                      cv2.FILLED)
        
        x1_text = x1_rect + 12
        if id > 99:
            x1_text-=10
            cv2.putText(frame,
                        f"{id}",
                        (int(x1_text),int(y1_rect+15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0,0,0),
                        2
            )
    return frame

import cv2

def draw_track_id(frame, bbox, track_id):
    x1, y1, x2, y2 = map(int, bbox)

    # Trung tâm phía trên đầu
    x_center = (x1 + x2) // 2
    y_top = y1

    rectangle_width = 40
    rectangle_height = 20

    # Vẽ box phía trên đầu
    rect_x1 = x_center - rectangle_width // 2
    rect_y1 = y_top - 30

    rect_x2 = x_center + rectangle_width // 2
    rect_y2 = y_top - 10

    cv2.rectangle(
        frame,
        (rect_x1, rect_y1),
        (rect_x2, rect_y2),
        (255, 0, 255),
        cv2.FILLED
    )

    # Hiển thị track id
    text_x = rect_x1 + 10
    text_y = rect_y1 + 15

    cv2.putText(
        frame,
        f"{track_id}",
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2
    )

    return frame

def draw_annotations(frames,tracks):
    output_video_frame = []
    person_tracks = tracks.get("person", [])
    for frame_num, frame in enumerate(frames):
        frame = frame.copy()

        if frame_num >= len(person_tracks):
            output_video_frame.append(frame)
            continue

        person_dict = tracks["person"][frame_num]

        for track_id, person in person_dict.items():
            frame = draw_track_id(frame,person['bbox'],track_id)
        output_video_frame.append(frame)
    return output_video_frame
    

