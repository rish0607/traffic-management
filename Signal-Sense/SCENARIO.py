import cv2
import torch
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model
print("Loading YOLO model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print("YOLO Model Loaded.")

video_paths = [
    r"C:\Users\dalal\OneDrive\Desktop\6th sem\AIML\AIML Lab Programs\Signal-Sense\vid\jam5.mp4",
    r"C:\Users\dalal\OneDrive\Desktop\6th sem\AIML\AIML Lab Programs\Signal-Sense\vid\jam5.mp4",
    r"C:\Users\dalal\OneDrive\Desktop\6th sem\AIML\AIML Lab Programs\Signal-Sense\vid\jam5.mp4",
    r"C:\Users\dalal\OneDrive\Desktop\6th sem\AIML\AIML Lab Programs\Signal-Sense\vid\jam5.mp4"
]

vehicle_classes = torch.tensor([2, 3, 5, 7], device=device)
caps = [cv2.VideoCapture(video) for video in video_paths]

for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_paths[i]}")

start_time = time.time()
green_lane = -1
last_green_lane = -1  # For round-robin on tie
interval = 5

frame_size = (640, 360)

while all(cap.isOpened() for cap in caps):
    frames = []
    vehicle_counts = []

    for cap in caps:
        ret, frame = cap.read()
        if not ret:
            frames.append(None)
            vehicle_counts.append(0)
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        detections = results.pred[0].to(device)
        filtered = detections[torch.isin(detections[:, 5], vehicle_classes)]
        count = len(filtered)

        for *box, conf, cls in filtered:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (19, 220, 223), 2)
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        frames.append(frame)
        vehicle_counts.append(count)

    if time.time() - start_time > interval:
        max_count = max(vehicle_counts)
        max_indices = [i for i, count in enumerate(vehicle_counts) if count == max_count]

        if len(max_indices) == 1:
            green_lane = max_indices[0]
        else:
            # All (or multiple) lanes are equal — use round robin
            green_lane = (last_green_lane + 1) % len(video_paths)
        
        last_green_lane = green_lane
        start_time = time.time()

    # Resize and add green signal overlay
    for i in range(len(frames)):
        if frames[i] is not None:
            frames[i] = cv2.resize(frames[i], frame_size)
            color = (0, 255, 0) if i == green_lane else (0, 0, 255)
            cv2.putText(frames[i], f'Lane {i+1} - Count: {vehicle_counts[i]}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Fill missing frames
    frames = [f if f is not None else np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8) for f in frames]

    top_row = np.hstack(frames[:2])
    bottom_row = np.hstack(frames[2:])
    grid_frame = np.vstack([top_row, bottom_row])

    cv2.imshow("Smart Traffic Control", grid_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
