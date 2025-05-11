import cv2
import torch
from torchvision import transforms
from PIL import Image
import os
import time
from datetime import datetime
import logging
from modeln import ArmenianLetterNet  # Импорт модели (как у тебя)

# ==== Параметры ====
SAVE_DIR = "/Users/aleksandr/Desktop/letter_test_output"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = "/Users/aleksandr/Desktop/Работа/СКАТ/To_real_dron/armenian_letters_model_improved.pth"
LABELS_PATH = "/Users/aleksandr/Desktop/Работа/СКАТ/To_real_dron/labels.txt"
CONFIDENCE_THRESHOLD = 0.85

CAMERA_ID = 0  # 0 = встроенная камера, 1 = USB-камера

logging.basicConfig(
    filename=os.path.join(SAVE_DIR, "letters.log"),
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

# ==== Загрузка модели и меток ====
model = ArmenianLetterNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f]

# ==== Преобразование для модели ====
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==== Камера ====
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("🔤 Начинаем распознавание букв (нажми 'q' для выхода)")
last_save_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Кадр не получен")
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            letter_img = gray[y:y+h, x:x+w]
            img_pil = Image.fromarray(letter_img)
            img_tensor = transform(img_pil).unsqueeze(0)

            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                confidence, pred = torch.max(probs, 1)
                confidence = confidence.item()
                label = labels[pred.item()]

            if confidence > CONFIDENCE_THRESHOLD:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_msg = f"✅ Распознано: {label} ({confidence:.2f})"
                print(log_msg)
                logging.info(log_msg)

                # Сохранение кадра (раз в 10 секунд)
                if time.time() - last_save_time > 10:
                    filename = os.path.join(SAVE_DIR, f"letter_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"💾 Сохранено изображение: {filename}")
                    last_save_time = time.time()

    cv2.imshow("Letter Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
