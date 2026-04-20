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
FULL_RES = (640, 360)
PREVIEW_RES = (640, 360)
SAVE_DIR = "/home/pi/tests/p3/armenian_letters_output"
CONFIDENCE_THRESHOLD = 0.85
os.makedirs(SAVE_DIR, exist_ok=True)
PADDING = 30
OUTPUT_SIZE = 32

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
    transforms.Resize((OUTPUT_SIZE, OUTPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Функция для поиска и выделения буквы с рамкой ===
def extract_letter_image(thresh_img, padding=PADDING, output_size=OUTPUT_SIZE):
    inverted_img = cv2.bitwise_not(thresh_img)
    contours, _ = cv2.findContours(inverted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, thresh_img, None

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    x_exp = max(x - padding, 0)
    y_exp = max(y - padding, 0)
    x_end = min(x + w + padding, inverted_img.shape[1])
    y_end = min(y + h + padding, inverted_img.shape[0])

    roi = inverted_img[y_exp:y_end, x_exp:x_end]

    roi_height, roi_width = roi.shape
    size = max(roi_height, roi_width)
    square_img = np.ones((size, size), dtype=np.uint8) * 255
    y_offset = (size - roi_height) // 2
    x_offset = (size - roi_width) // 2
    square_img[y_offset:y_offset+roi_height, x_offset:x_offset+roi_width] = roi

    resized_img = cv2.resize(square_img, (output_size, output_size), interpolation=cv2.INTER_AREA)
    inverted_again_img = cv2.bitwise_not(resized_img)

    return inverted_again_img, thresh_img, (largest_contour, (x_exp, y_exp, x_end, y_end))

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

    # Пороговое преобразование с высоким порогом (можно подобрать)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Используем новую функцию для поиска буквы и рамки
    processed_letter, thresh_img, contour_data = extract_letter_image(thresh)

    if contour_data is not None:
        largest_contour, (x_exp, y_exp, x_end, y_end) = contour_data

        # Создаем изображение для отрисовки (цветное)
        thresh_color = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(thresh_color, [largest_contour], -1, (0, 0, 255), 2)  # красный контур
        cv2.rectangle(thresh_color, (x_exp, y_exp), (x_end, y_end), (0, 255, 0), 2)  # зелёная рамка

        # Обрабатываем букву моделью, если она есть
        img_pil = Image.fromarray(processed_letter)
        img_tensor = transform(img_pil).unsqueeze(0)
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
            conf = conf.item()
            label = pred.item()

        if conf > CONFIDENCE_THRESHOLD:
            text = f"{labels[label]} ({conf:.2f})"
            print(f"✅ Найдена буква: {text}")
            cv2.putText(frame, text, (x_exp, y_exp - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:

            print(f"⚠️ Низкая уверенность: {conf:.2f}")

        # Отобразим контуры и рамку
        cv2.imshow("Threshold with Contour and Bounding Box", thresh_color)
        # Отобразим обрезанную букву, подготовленную для модели
        cv2.imshow("Processed Letter", processed_letter)
    else:
        cv2.imshow("Threshold with Contour and Bounding Box", cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR))

    # Показываем исходный кадр с результатом распознавания
    preview = cv2.resize(frame, PREVIEW_RES)
    cv2.imshow("Armenian Letters Preview", preview)

    # Сохраняем кадр каждые 10 секунд
    if time.time() - last_save_time > 10:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"armenian_letter_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"💾 Сохранено: {filename}")
        last_save_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()

#python3 fbnm_3.py
#python3 test_old_model_rpi.py

#cd Downloads
#source env_1/bin/activate
# cd ..
#cd tests/p1/p3/armenian_letter
#
