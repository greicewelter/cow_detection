import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

video_input_path = '/home/nome_usuario/cow_detection/videos/vacas.mp4'
modelo_yolo = 'yolov8n.pt'

accuracy = 0.85 
loss = 1 - accuracy  

if not os.path.exists(video_input_path):
    raise FileNotFoundError(f"Vídeo não encontrado: {video_input_path}")

print(f"Usando modelo oficial: {modelo_yolo}")

model = YOLO(modelo_yolo)

cap = cv2.VideoCapture(video_input_path)
window_name = 'Detecção com YOLOv8'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.25, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            if label == 'cow':
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

labels = ['Acurácia', 'Perda']
sizes = [accuracy, loss]
colors = ['#4CAF50', '#F44336']
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.title('Desempenho do Modelo YOLO yolov8n.pt')
plt.show()
