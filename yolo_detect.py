import cv2, time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # 首次自动下载约 6MB

cap = cv2.VideoCapture(0)
prev = time.time()

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, verbose=False)
    for r in results:
        for box in r.boxes:
            if float(box.conf[0]) < 0.4: continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            name = model.names[int(box.cls[0])]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {float(box.conf[0]):.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cur = time.time()
    cv2.putText(frame, f"FPS: {1/(cur-prev):.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    prev = cur

    cv2.imshow("YOLO", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
