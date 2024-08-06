import cv2
from ultralytics import YOLO

model_path = "C:\\Users\\surao\\.spyder-py3\\proje\\irk_tespiti\\race_detection_best.pt"
model = YOLO(model_path)

class_names = ['Asian', 'Black', 'Indian', 'White']

cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("Kamera açılamadı.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Kare alınamadı.")
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes

        for i, box in enumerate(boxes):
            # Koordinatları al
            top_left_x = int(box.xyxy[0][0])
            top_left_y = int(box.xyxy[0][1])
            bottom_right_x = int(box.xyxy[0][2])
            bottom_right_y = int(box.xyxy[0][3])

            label_id = int(box.cls[0]) if box.cls is not None else 0
            label = class_names[label_id] if label_id < len(class_names) else "Unknown"

            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
            cv2.putText(frame, label, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            print(f"Yüz {i+1}: {label}")

    cv2.imshow("Kamera - Yüz Tespiti", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
