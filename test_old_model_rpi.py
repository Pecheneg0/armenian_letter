import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from modelold import ArmenianLetterNet  # подключаем новую модель
import time

# === Конфигурация ===
MODEL_PATH = "ALM_best.pth"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.85
CAMERA_ID = 0  # Камера по умолчанию

# === Загрузка новой модели ===
model = ArmenianLetterNet()
state_dict = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

# === Загрузка меток ===
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# === Подготовка преобразования ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Инициализация камеры ===
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

print("Запуск камеры. Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка захвата кадра")
        break

    # Преобразуем в градации серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Применим порог для выделения белого плаката на фоне
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > 10000:  # Настроить порог площади!
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            
            # Вырезаем и подготавливаем область
            letter_img = gray[y:y+h, x:x+w]
            img_pil = Image.fromarray(letter_img)
            img_tensor = transform(img_pil).unsqueeze(0)
            
            # Прогон через новую модель
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                conf = conf.item()
                label = pred.item()
                
            if conf > CONFIDENCE_THRESHOLD:
                text = f"{labels[label]} ({conf:.2f})"
                cv2.putText(frame, text, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                print(f"Найдена буква: {text}")
            else:
                print("Слишком низкая уверенность")
    
    # Отображаем изображение
    #cv2.imshow("Armenian Letter Detection", frame)
    
    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()
