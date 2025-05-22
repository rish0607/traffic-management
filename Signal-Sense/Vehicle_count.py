import cv2
import torch
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model
print("Loading YOLO model...")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
print("YOLO Model Loaded.")

# videos 
video_paths = [
    r"C:\Users\dalal\OneDrive\Desktop\6th sem\AIML\AIML Lab Programs\Signal-Sense\vid\carrs.mp4",
    r"C:\Users\dalal\OneDrive\Desktop\6th sem\AIML\AIML Lab Programs\Signal-Sense\vid\carrs2.mp4",
    r"C:\Users\dalal\OneDrive\Desktop\6th sem\AIML\AIML Lab Programs\Signal-Sense\vid\carrs3.mp4",
    r"C:\Users\dalal\OneDrive\Desktop\6th sem\AIML\AIML Lab Programs\Signal-Sense\vid\carrs4.mp4"
]

print("Video paths:", video_paths)

vehicle_classes = torch.tensor([2, 3, 5, 7], device=device)  # car, motorcycle, bus, truck

# Open video captures
caps = [cv2.VideoCapture(video) for video in video_paths]

for i, cap in enumerate(caps):
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_paths[i]}")

while all(cap.isOpened() for cap in caps):
    frames = []
    for idx, cap in enumerate(caps):
        ret, frame = cap.read()
        if not ret:
            frames.append(None)
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)

        detections = results.pred[0].to(device)
        filtered_detections = detections[torch.isin(detections[:, 5], vehicle_classes)]

        vehicle_count = filtered_detections.shape[0]  # Number of vehicle detections

        for *box, conf, cls in filtered_detections:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (19, 220, 223), 2)
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display vehicle count
        count_label = f"Video {idx+1}: Vehicles = {vehicle_count}"
        cv2.putText(frame, count_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        frames.append(frame)

    # Ensure all frames have the same size
    frame_size = (640, 360)
    frames = [cv2.resize(f, frame_size) if f is not None else np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8) for f in frames]

    # 2x2 grid
    top_row = np.hstack(frames[:2])
    bottom_row = np.hstack(frames[2:])
    grid_frame = np.vstack([top_row, bottom_row])

    cv2.imshow('Multi-Video YOLOv5 Detection with Counts', grid_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
for cap in caps:
    cap.release()
cv2.destroyAllWindows()
