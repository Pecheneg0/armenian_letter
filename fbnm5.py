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
import math

# === Конфигурация ===
MODEL_PATH = "ALM_best.pth"
LABELS_PATH = "labels.txt"
FULL_RES = (720, 720)       # Разрешение захвата
PREVIEW_RES = (720, 720 )      # Разрешение для отображения
#SAVE_DIR = "/media/pi/ADATA/СКАТ/Фото/Folder_for_tests"
SAVE_DIR = "/home/pi/tests/p1/p3/armenian_letters_output"
CONFIDENCE_THRESHOLD = 0.85   # Порог уверенности распознавания
PROCESSING_FPS = 50            # Ограничение частоты обработки (кадров в секунду)
os.makedirs(SAVE_DIR, exist_ok=True)

# === Инициализация модели ===
model = ArmenianLetterNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()  # Переводим модель в режим оценки

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
    main={"size": FULL_RES, "format": "RGB888"},
    controls={"FrameRate": 10}  # Ограничение FPS камеры
)
picam2.configure(config)
picam2.start()
print("📸 Система запущена. Поиск букв армянского алфавита... Нажмите 'q' для выхода.")

# Переменные для контроля скорости обработки
last_processing_time = time.time()
processing_interval = 1.0 / PROCESSING_FPS  # Интервал между обработкой кадров

# Коэффициенты преобразования (настраиваются экспериментально)
PIXELS_PER_METER = 100       # Пикселей на метр (зависит от высоты)
home_lat = 0.000000#55.754107
home_lon = 0.000000#37.861527 

def pixels_to_meters(pixel_x, pixel_y, ppm):
    """Конвертирует смещение в пикселях в метры"""
    return pixel_x / ppm, pixel_y / ppm

def add_meters_to_coords(lat, lon, dx_m, dy_m, heading_deg=0):
    """
    Добавляет смещение в метрах к координатам с учетом курса
    и эллиптической формы Земли.
    
    Параметры:
        lat, lon - исходные координаты в градусах
        dx_m, dy_m - смещение в метрах (восток/север)
        heading_deg - курс в градусах (0 - север)
    
    Возвращает:
        новые координаты (lat, lon) в градусах
    """
    # Конвертируем курс в радианы
    heading_rad = math.radians(heading_deg)
    
    # Разделяем смещение на север/восток компоненты
    north_m = dy_m * math.cos(heading_rad) + dx_m * math.sin(heading_rad)
    east_m = dx_m * math.cos(heading_rad) - dy_m * math.sin(heading_rad)
    
    # Константы для расчетов
    METERS_PER_DEGREE_LAT = 111134.861111  # По меридиану, в метрах
    METERS_PER_DEGREE_LON_AT_EQUATOR = 111321.377778  # На экваторе, в метрах
    
    # Расчет для широты (просто по меридиану)
    dlat = north_m / METERS_PER_DEGREE_LAT
    
    # Расчет для долготы (учитываем широту)
    lat_rad = math.radians(lat)
    meters_per_degree_lon = METERS_PER_DEGREE_LON_AT_EQUATOR * math.cos(lat_rad)
    dlon = east_m / meters_per_degree_lon
    
    return lat + dlat, lon + dlon

def process_frame(frame):
    """Основная функция обработки кадра"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if  110000 < area <= 150000:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            center_x, center_y = int(rect[0][0]), int(rect[0][1])
            
            # Вычисляем смещение относительно центра кадра
            frame_center_x, frame_center_y = frame.shape[1]//2, frame.shape[0]//2
            dx_px = center_x - frame_center_x
            dy_px = center_y - frame_center_y
            
            # Преобразуем в метры и географические координаты
            dx_m, dy_m = pixels_to_meters(dx_px, dy_px, PIXELS_PER_METER)
            new_lat, new_lon = add_meters_to_coords(home_lat, home_lon , dx_m, dy_m)
            
            results.append({
                'center_px': (center_x, center_y),
                'center_m': (dx_m, dy_m),
                'coords': (new_lat, new_lon),
                'box': box
            })
    
    return results, thresh_color
last_save_time = time.time()
while True:
    # Ограничение частоты обработки
    current_time = time.time()
    if current_time - last_processing_time < processing_interval:
        time.sleep(0.01)  # Небольшая задержка для снижения нагрузки
        continue
    
    last_processing_time = current_time
    
    # Захват кадра
    frame_rgb = picam2.capture_array()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    
    # Обработка кадра
    results, thresh_color = process_frame(frame)
    processed_letter = None
    
    for result in results:
        # Отрисовка результатов
        cv2.drawContours(frame, [result['box']], 0, (0, 255, 0), 2)
        cv2.circle(frame, result['center_px'], 5, (0, 0, 255), -1)
        
        # Распознавание буквы
        width, height = map(int, cv2.minAreaRect(result['box'])[1])
        if width > 0 and height > 0:
            letter_crop = cv2.warpPerspective(
                thresh_color, 
                cv2.getPerspectiveTransform(
                    result['box'].astype("float32"),
                    np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
                ),
                (width, height)
            )
            
            if np.mean(letter_crop) < 250:
                processed_letter = letter_crop.copy()
                try:
                    img_tensor = transform(Image.fromarray(letter_crop)).unsqueeze(0)
                    with torch.no_grad():
                        output = model(img_tensor)
                        conf, pred = torch.max(torch.nn.functional.softmax(output, dim=1), 1)
                        
                        if conf.item() > CONFIDENCE_THRESHOLD:
                            text = f"{labels[pred.item()]} ({conf.item():.2f})"
                            cv2.putText(frame, text, result['center_px'], 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            print(f"Распознано: {text} | Координаты: {result['coords']}")
                except Exception as e:
                    print(f"Ошибка обработки: {e}")

    # Отображение результатов
    preview = cv2.resize(frame, PREVIEW_RES)
    cv2.imshow("Armenian Letters", preview)
    #cv2.imshow("Threshold", cv2.resize(thresh_color, PREVIEW_RES))
    cv2.imshow("Processed", processed_letter if processed_letter is not None else np.zeros((100,100,3), dtype=np.uint8))
    #if time.time() - last_save_time > 5:
     #   timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      #  filename = os.path.join(SAVE_DIR, f"armenian_letter_{timestamp}.jpg")
      #  cv2.imwrite(filename, frame)
      #  print(f"💾 Сохранено: {filename}")
      #  last_save_time = time.time() 
    # Управление и выход
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Завершение работы
cv2.destroyAllWindows()
picam2.stop()
