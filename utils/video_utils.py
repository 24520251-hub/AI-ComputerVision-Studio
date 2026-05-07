import cv2, os

#Read Video
def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame =cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

#Save Next Video with
def get_next_video_name(output_dir="assets/output_videos", prefix="run", ext=".avi"):
    os.makedirs(output_dir, exist_ok=True)

    files = os.listdir(output_dir)
    nums = []

    for f in files:
        if f.startswith(prefix) and f.endswith(ext):
            try:
                num = int(f[len(prefix):-len(ext)])
                nums.append(num)
            except:
                pass

    next_num = max(nums) + 1 if nums else 1

    return os.path.join(output_dir, f"{prefix}{next_num}{ext}")

#Save Video
def save_video(output_video_frames, output_dir="assets/output_videos"):
     
    if len(output_video_frames) == 0:
        print("No frames!")
        return

    if output_video_frames[0] is None:
        print("Invalid frame!")
        return

    height, width = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    output_video_path = get_next_video_name(output_dir=output_dir)

    out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
    for frame in output_video_frames:
        out.write(frame)

    out.release()
    print("Saved:", output_video_path)