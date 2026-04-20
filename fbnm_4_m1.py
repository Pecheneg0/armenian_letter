import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from modelold import ArmenianLetterNet
import time
import os
from datetime import datetime
from picamera2 import Picamera2
from libcamera import controls

# === Конфигурация ===
MODEL_PATH = "ALM_best.pth"
LABELS_PATH = "labels.txt"
FULL_RES = (1920, 1080)
PREVIEW_RES = (640, 360)
SAVE_DIR = "/home/pi/tests/p1/p3/armenian_letters_output"
CONFIDENCE_THRESHOLD = 0.85
os.makedirs(SAVE_DIR, exist_ok=True)

# === Загрузка модели ===
model = ArmenianLetterNet()
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# === Загрузка меток ===
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# === Преобразование изображения ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Инициализация камеры ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": FULL_RES, "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
print("📸 Поиск букв армянского алфавита... Нажми 'q' для выхода.")

last_save_time = time.time()

while True:
    frame_rgb = picam2.capture_array()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Создаем цветное изображение порога для отрисовки контуров
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # Находим все контуры
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    processed_letter = None  # Чтобы было что показать в окне "Processed Letter"

    for cnt in contours:
        # Отрисовываем контур синим цветом
        cv2.drawContours(thresh_color, [cnt], -1, (255, 0, 0), 2)

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > 10000:
            # Рисуем рамки
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(thresh_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Создаем маску для текущего контура
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            # Извлекаем только букву из исходного gray (или можно из color)
            letter_img_masked = cv2.bitwise_and(thresh_color, thresh_color, mask=mask)
            x, y, w, h = cv2.boundingRect(cnt)
            letter_crop = letter_img_masked[y:y+h, x:x+w]

            # Для отображения в окне
            processed_letter = letter_crop.copy()

            # Готовим изображение для нейросети
            img_pil = Image.fromarray(letter_crop)
            img_tensor = transform(img_pil).unsqueeze(0)


            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                conf = conf.item()
                label = pred.item()

            if conf > CONFIDENCE_THRESHOLD:
                text = f"{labels[label]} ({conf:.2f})"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"✅ Найдена буква: {text}")
            else:
                print(f"⚠️ Низкая уверенность: {conf:.2f}")

    # Сохраняем кадр каждые 10 секунд
    if time.time() - last_save_time > 10:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"armenian_letter_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"💾 Сохранено: {filename}")
        last_save_time = time.time()

    # Окно 1: Основной кадр с результатом
    preview = cv2.resize(frame, PREVIEW_RES)
    cv2.imshow("Armenian Letters Preview", preview)

    # Окно 2: Пороговое изображение с контурами
    cv2.imshow("Threshold with Contours", thresh_color)

    # Окно 3: Вырезанное изображение буквы (если найдено)
    if processed_letter is not None:
        cv2.imshow("Processed Letter", processed_letter)
    else:
        # Показываем пустое изображение, если буква не найдена
        empty_image = np.ones((100, 100), dtype=np.uint8) * 255
        cv2.imshow("Processed Letter", empty_image)

    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()

