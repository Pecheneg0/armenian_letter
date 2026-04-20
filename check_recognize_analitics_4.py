import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from modelold import ArmenianLetterNet
import time
from datetime import datetime
import math
import os
from collections import defaultdict
import logging

# === Конфигурация ===
MODEL_PATH = "/Users/aleksandr/Desktop/Работа/СКАТ/To_real_dron/Документация /ALM_best.pth"
LABELS_PATH = "/Users/aleksandr/Desktop/Работа/СКАТ/To_real_dron/Документация /labels.txt"
FULL_RES = (720, 720)            # Разрешение обработки
PREVIEW_RES = (720, 720)         # Разрешение для отображения
CONFIDENCE_THRESHOLD = 0.85      # Порог уверенности распознавания
PROCESSING_FPS = 60              # Ограничение частоты обработки
LETTER_TIMEOUT = 2.0             # Таймаут между обнаружениями (сек)
MIN_DETECTIONS = 3               # Минимальное количество обнаружений для учета буквы
SAVE_INTERVAL = 0.2              # Интервал между сохранениями (секунды)

# === Параметры фильтрации контуров ===
MIN_AREA = 8000                  # Минимальная площадь контура (px)
MAX_AREA = 30000                 # Максимальная площадь контура (px)
MIN_WIDTH = 20                   # Минимальная ширина буквы (px)
ASPECT_RATIO_RANGE = (1.5, 8.0)  # Допустимый диапазон соотношения сторон
CONTOUR_DENSITY = 0.02           # Минимальная плотность контура
TEXTURE_THRESHOLD = 50           # Минимальная текстурная сложность

# === Режим работы ===
USE_VIDEO = 0                    # True для видео, False для фото
VIDEO_SOURCE = "/Users/aleksandr/Desktop/Запись экрана 2025-08-03 в 20.15.19.mov"
PHOTO_SOURCE = "/Users/aleksandr/Desktop/Снимок экрана 2025-08-04 в 01.42.56.png"
LETTERS_DIR = '/Users/aleksandr/Desktop/Работа/СКАТ/test_images'
LOG_DIR = '/Users/aleksandr/Desktop/Работа/СКАТ/logs'

# === Инициализация логгера ===
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "sla_controller.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Инициализация модели ===
device = torch.device('cpu')
model = ArmenianLetterNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
logging.info("Model loaded successfully")

# === Загрузка меток ===
with open(LABELS_PATH, "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# === Создание директорий ===
os.makedirs(LETTERS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# === Преобразование изображения ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# === Инициализация источника изображений ===
if USE_VIDEO:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        logging.error(f"Failed to open video: {VIDEO_SOURCE}")
        exit()
else:
    frame = cv2.imread(PHOTO_SOURCE)
    if frame is None:
        logging.error(f"Failed to load image: {PHOTO_SOURCE}")
        exit()

# Коэффициенты преобразования
PIXELS_PER_METER = 44.482
home_lat = 55.754107
home_lon = 37.861527 

def pixels_to_meters(pixel_x, pixel_y, ppm):
    """Конвертирует смещение в пикселях в метры"""
    return pixel_x / ppm, pixel_y / ppm

def add_meters_to_coords(lat, lon, dx_m, dy_m, heading_deg=0):
    """Добавляет смещение в метрах к координатам"""
    heading_rad = math.radians(heading_deg)
    north_m = dy_m * math.cos(heading_rad) + dx_m * math.sin(heading_rad)
    east_m = dx_m * math.cos(heading_rad) - dy_m * math.sin(heading_rad)
    
    METERS_PER_DEGREE_LAT = 111134.861111
    METERS_PER_DEGREE_LON_AT_EQUATOR = 111321.377778
    
    dlat = north_m / METERS_PER_DEGREE_LAT
    lat_rad = math.radians(lat)
    meters_per_degree_lon = METERS_PER_DEGREE_LON_AT_EQUATOR * math.cos(lat_rad)
    dlon = east_m / meters_per_degree_lon
    
    return lat + dlat, lon + dlon

def process_frame(frame):
    """Улучшенная функция обработки кадра с фильтрацией ложных срабатываний"""
    # 1. Предварительная обработка
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Адаптивная бинаризация
    thresh = cv2.adaptiveThreshold(blurred, 255, 
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # 3. Морфологическая очистка
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # 4. Поиск контуров
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for cnt in contours:
        # 5. Фильтрация по площади
        area = cv2.contourArea(cnt)
        if not (MIN_AREA < area < MAX_AREA):
            continue
            
        # 6. Фильтрация по форме
        rect = cv2.minAreaRect(cnt)
        width, height = rect[1]
        if min(width, height) < MIN_WIDTH:
            continue
            
        aspect_ratio = max(width, height) / min(width, height)
        if not (ASPECT_RATIO_RANGE[0] <= aspect_ratio <= ASPECT_RATIO_RANGE[1]):
            continue
            
        # 7. Проверка плотности контура
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        density = area / (perimeter ** 2)
        if density < CONTOUR_DENSITY:
            continue
            
        # 8. Проверка текстурной сложности
        x,y,w,h = cv2.boundingRect(cnt)
        roi = gray[y:y+h, x:x+w]
        laplacian_var = cv2.Laplacian(roi, cv2.CV_64F).var()
        if laplacian_var < TEXTURE_THRESHOLD:
            continue
            
        # 9. Проверка углов (должен быть четырехугольник)
        approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
            
        # Если все проверки пройдены
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        center_x, center_y = int(rect[0][0]), int(rect[0][1])
        
        frame_center_x, frame_center_y = frame.shape[1]//2, frame.shape[0]//2
        dx_px = center_x - frame_center_x
        dy_px = -center_y + frame_center_y
        
        dx_m, dy_m = pixels_to_meters(dx_px, dy_px, PIXELS_PER_METER)
        new_lat, new_lon = add_meters_to_coords(home_lat, home_lon, dx_m, dy_m)
        
        results.append({
            'center_px': (center_x, center_y),
            'center_m': (dx_m, dy_m),
            'box': box,
            'coords': (new_lat, new_lon),
            'angle': rect[2],
            'area': area,
            'aspect_ratio': aspect_ratio,
            'texture': laplacian_var
        })
    
    return results, cleaned

def recognize_letter(image_crop):
    """Распознавание буквы на вырезанном фрагменте"""
    try:
        img_tensor = transform(Image.fromarray(image_crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            conf, pred = torch.max(torch.nn.functional.softmax(output, dim=1), 1)
            return labels[pred.item()], conf.item()
    except Exception as e:
        logging.error(f"Recognition error: {e}")
        return None, 0.0

def draw_results(frame, results):
    """Отрисовка результатов на кадре"""
    for result in results:
        cv2.drawContours(frame, [result['box']], 0, (0, 255, 0), 2)
        cv2.circle(frame, result['center_px'], 5, (0, 0, 255), -1)
        
        # Отладочная информация
        debug_text = f"A:{result['area']} AR:{result['aspect_ratio']:.1f} T:{result['texture']:.0f}"
        cv2.putText(frame, debug_text, 
                   (result['center_px'][0], result['center_px'][1] + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

# Основной цикл обработки
last_processing_time = time.time()
processing_interval = 1.0 / PROCESSING_FPS
processing_sessions = []
last_detection_time = time.time()
letter_stats = defaultdict(lambda: {'confidences': [], 'coords': [], 'timestamps': []})
processing_active = False
last_save_time = 0

while True:
    # Ограничение частоты обработки
    current_time = time.time()
    if current_time - last_processing_time < processing_interval:
        time.sleep(0.01)
        continue
    
    last_processing_time = current_time
    
    # Получение кадра
    if USE_VIDEO:
        ret, frame = cap.read()
        if not ret:
            logging.info("Video processing completed")
            break
    else:
        frame = cv2.imread(PHOTO_SOURCE).copy()
    
    # Обработка кадра
    results, processed_img = process_frame(frame)
    
    # Распознавание букв
    letter_detected = False
    for result in results:
        width, height = map(int, cv2.minAreaRect(result['box'])[1])
        if width > 10 and height > 10:
            try:
                # Подготовка области для распознавания
                letter_crop = cv2.warpPerspective(
                    processed_img, 
                    cv2.getPerspectiveTransform(
                        result['box'].astype("float32"),
                        np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
                    ),
                    (width, height)
                )
                
                # Распознавание
                letter, confidence = recognize_letter(letter_crop)
                if confidence > CONFIDENCE_THRESHOLD:
                    letter_detected = True
                    last_detection_time = current_time
                    letter_stats[letter]['confidences'].append(confidence)
                    letter_stats[letter]['coords'].append(result['coords'])
                    letter_stats[letter]['timestamps'].append(current_time)
                    processing_active = True
                    
                    # Визуализация
                    text = f"{letter} ({confidence:.2f})"
                    cv2.putText(frame, text, result['center_px'], 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    logging.info(f"Detected: {text} | Coords: {result['coords']}")
                    
                    # Сохранение кадра
                    if current_time - last_save_time > SAVE_INTERVAL:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(LETTERS_DIR, f"letter_{timestamp}.jpg")
                        if not os.path.exists(filename):
                            cv2.imwrite(filename, frame)
                            logging.info(f"Saved: {filename}")
                            last_save_time = current_time
                            
            except Exception as e:
                logging.error(f"Processing error: {e}")

    # Обработка таймаута
    if processing_active and (current_time - last_detection_time) > LETTER_TIMEOUT:
        if letter_stats:
            best_letter, best_data = max(letter_stats.items(),
                                      key=lambda x: len(x[1]['confidences']))
            
            if len(best_data['confidences']) >= MIN_DETECTIONS:
                # Вычисление средних значений
                avg_conf = np.mean(best_data['confidences'])
                avg_lat = np.mean([c[0] for c in best_data['coords']])
                avg_lon = np.mean([c[1] for c in best_data['coords']])
                last_time = datetime.fromtimestamp(max(best_data['timestamps'])).strftime('%H:%M:%S')
                
                # Логирование результатов
                result_msg = f"""
                === Processing Results ===
                Letter: {best_letter}
                Detections: {len(best_data['confidences'])}
                Confidence: {avg_conf:.2f}
                Coordinates: ({avg_lat:.6f}, {avg_lon:.6f})
                Time: {last_time}
                =========================="""
                logging.info(result_msg)
                
                # Сохранение сессии
                processing_sessions.append({
                    'letter': best_letter,
                    'count': len(best_data['confidences']),
                    'avg_confidence': avg_conf,
                    'avg_coords': (avg_lat, avg_lon),
                    'timestamp': last_time
                })
        
        letter_stats = defaultdict(lambda: {'confidences': [], 'coords': [], 'timestamps': []})
        processing_active = False
    
    # Отрисовка результатов
    draw_results(frame, results)
    
    # Отображение
    cv2.imshow("Armenian Letters Recognition", cv2.resize(frame, PREVIEW_RES))
    cv2.imshow("Processed", processed_img if 'processed_img' in locals() else np.zeros((100,100,3), dtype=np.uint8))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if not USE_VIDEO:
        cv2.waitKey(0)
        break

# Финализация при завершении
if processing_active and letter_stats:
    best_letter, best_data = max(letter_stats.items(),
                               key=lambda x: len(x[1]['confidences']))
    
    if len(best_data['confidences']) >= MIN_DETECTIONS:
        avg_conf = np.mean(best_data['confidences'])
        avg_lat = np.mean([c[0] for c in best_data['coords']])
        avg_lon = np.mean([c[1] for c in best_data['coords']])
        last_time = datetime.fromtimestamp(max(best_data['timestamps'])).strftime('%H:%M:%S')
        
        logging.info(f"Final detection: {best_letter} (Conf: {avg_conf:.2f})")

# Вывод сводки
if processing_sessions:
    summary = "\n=== Processing Summary ==="
    for i, session in enumerate(processing_sessions, 1):
        summary += f"""
        {i}. {session['letter']}
           Detections: {session['count']}
           Confidence: {session['avg_confidence']:.2f}
           Coords: {session['avg_coords']}
           Time: {session['timestamp']}
        """
    logging.info(summary)

# Освобождение ресурсов
if USE_VIDEO:
    cap.release()
cv2.destroyAllWindows()