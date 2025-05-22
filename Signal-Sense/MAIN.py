import cv2
import torch
import warnings
import numpy as np
import time

warnings.filterwarnings("ignore", category=FutureWarning)

print("Loading YOLO model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print("YOLO Model Loaded.")

video_paths = [
    r"C:\Users\dalal\OneDrive\Desktop\6th sem\AIML\AIML Lab Programs\Signal-Sense\vid\jam3.mp4",
    r"C:\Users\dalal\OneDrive\Desktop\6th sem\AIML\AIML Lab Programs\Signal-Sense\vid\jam6.mp4",
    r"C:\Users\dalal\OneDrive\Desktop\6th sem\AIML\AIML Lab Programs\Signal-Sense\vid\carrs3.mp4",
    r"C:\Users\dalal\OneDrive\Desktop\6th sem\AIML\AIML Lab Programs\Signal-Sense\vid\jam.mp4"
]

print("Video paths:", video_paths)

vehicle_classes = torch.tensor([2, 3, 5, 7], device=device)

# Open video captures
caps = [cv2.VideoCapture(video) for video in video_paths]
for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_paths[i]}")

frame_size = (640, 360)
green_lane_index = -1  # Which lane is currently green

buffer_start_time = time.time()
accumulated_counts = [0] * 4  
frame_counts = [0] * 4        

while all(cap.isOpened() for cap in caps):
    frames = []
    current_counts = []

    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            frames.append(None)
            current_counts.append(0)
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        detections = results.pred[0].to(device)
        filtered = detections[torch.isin(detections[:, 5], vehicle_classes)]

        count = filtered.shape[0]
        current_counts.append(count)

        # Draw detections
        for *box, conf, cls in filtered:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (19, 220, 223), 2)
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        frames.append(frame)

    # Accumulate counts over 5 seconds
    for i in range(4):
        accumulated_counts[i] += current_counts[i]
        frame_counts[i] += 1

    elapsed = time.time() - buffer_start_time
    if elapsed >= 5:
        # Average counts over frames to get fair comparison
        avg_counts = [accumulated_counts[i] / max(frame_counts[i], 1) for i in range(4)]
        green_lane_index = int(np.argmax(avg_counts))

        # Reset timers and accumulators
        accumulated_counts = [0] * 4
        frame_counts = [0] * 4
        buffer_start_time = time.time()

    # Resize and annotate each lane view
    for i in range(4):
        if frames[i] is None:
            frames[i] = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        else:
            frames[i] = cv2.resize(frames[i], frame_size)

        color = (0, 255, 0) if i == green_lane_index else (0, 0, 255)
        label = "GREEN" if i == green_lane_index else "RED"
        cv2.putText(frames[i], f"{label} | Count: {current_counts[i]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

   
    top_row = np.hstack(frames[:2])
    bottom_row = np.hstack(frames[2:])
    grid_frame = np.vstack([top_row, bottom_row])

    cv2.imshow('Smart Traffic Signal System (5s Accumulation)', grid_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
