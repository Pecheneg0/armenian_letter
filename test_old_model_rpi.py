import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from modelold import ArmenianLetterNet  # Подключаем модель
import time

# === Конфигурация ===
MODEL_PATH = "ALM_best.pth"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.85

SAVE_DIR = "/home/pi/tests/p1/p3/armenian_letters_output_tomr"
# === Загрузка модели ===
model = ArmenianLetterNet()
state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()

# === Инициализация камеры ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": FULL_RES, "format": "RGB888"}
)
picam2.configure(config)
picam2.start()
print("📸 Поиск букв армянского алфавита... Нажми 'q' для выхода.")

last_save_time = time.time()


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
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Ошибка: не удалось открыть камеру.")
    exit()

print("Запуск камеры. Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Ошибка захвата кадра")
        continue

    # Преобразование в оттенки серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Пороговое преобразование
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        area = cv2.contourArea(cnt)
        if len(approx) == 4 and area > 10000:
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Вырезаем область и преобразуем
            letter_img = gray[y:y + h, x:x + w]
            img_pil = Image.fromarray(letter_img)
            img_tensor = transform(img_pil).unsqueeze(0)

            # Прогон через модель
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.nn.functional.softmax(output, dim=1)
                conf, pred = torch.max(probs, 1)
                conf = conf.item()
                label = pred.item()

            if conf > CONFIDENCE_THRESHOLD:
                text = f"{labels[label]} ({conf:.2f})"
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                print(f"Найдена буква: {text}")
            else:
                print(f"Низкая уверенность: {conf:.2f}")

    # Отображение изображения
    #cv2.imshow("Armenian Letter Detection", frame)
    if time.time() - last_save_time > 10:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(SAVE_DIR, f"armenian_letter_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"💾 Сохранено: {filename}")
        last_save_time = time.time()    
    # Выход по клавише 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()

