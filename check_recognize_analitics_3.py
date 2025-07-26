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

SAVE_INTERVAL = 0.2  # Интервал между сохранениями (секунды)

# === Режим работы ===
USE_VIDEO = True   # True для видео, False для фото
VIDEO_SOURCE = "/Users/aleksandr/Downloads/test2.mp4"  # Путь к видеофайлу
PHOTO_SOURCE = "/Users/aleksandr/Desktop/45 5.jpg"  # Путь к тестовому фото
LETTERS_DIR = '/Users/aleksandr/Desktop/Работа/СКАТ/test_images'
LOG_DIR = '/Users/aleksandr/Desktop/Работа/СКАТ/logs'

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "sla_controller.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# === Инициализация модели ===
device = torch.device('cpu')
model = ArmenianLetterNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
logging.info(" Model loaded")
model.eval()
os.makedirs(LETTERS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
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

# === Инициализация источника изображений ===
if USE_VIDEO:
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"Ошибка открытия видео: {VIDEO_SOURCE}")
        logging.error(f"Ошибка открытия видео: {VIDEO_SOURCE}")
        exit()
else:
    frame = cv2.imread(PHOTO_SOURCE)
    if frame is None:
        print(f"Ошибка загрузки фото: {PHOTO_SOURCE}")
        logging.error(f"Ошибка открытия фото: {PHOTO_SOURCE}")
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
    """Основная функция обработки кадра"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 11400 < area:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            center_x, center_y = int(rect[0][0]), int(rect[0][1])
            frame_center_x, frame_center_y = 720/2, 720/2
            dx_px = center_x - frame_center_x
            dy_px = -center_y + frame_center_y
            
            dx_m, dy_m = pixels_to_meters(dx_px, dy_px, PIXELS_PER_METER)
            new_lat, new_lon = add_meters_to_coords(home_lat, home_lon, dx_m, dy_m)
            
            results.append({
                'center_px': (center_x, center_y),
                'center_m': (dx_m, dy_m),
                'box': box,
                'coords': (new_lat, new_lon),
                'angle': rect[2]
            })
    
    return results, thresh

def recognize_letter(image_crop):
    """Распознавание буквы на вырезанном фрагменте"""
    try:
        img_tensor = transform(Image.fromarray(image_crop)).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(img_tensor)
            conf, pred = torch.max(torch.nn.functional.softmax(output, dim=1), 1)
            return labels[pred.item()], conf.item()
    except Exception as e:
        print(f"Ошибка распознавания: {e}")
        logging.error(f"Ошибка распознавания: {e}")
        return None, 0.0

def draw_results(frame, results):
    """Отрисовка результатов на кадре"""
    for result in results:
        cv2.drawContours(frame, [result['box']], 0, (0, 255, 0), 2)
        cv2.circle(frame, result['center_px'], 5, (0, 0, 255), -1)

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
        time.sleep(0.001)
        continue
    
    last_processing_time = current_time
    processed_letter = None
    
    # Получение кадра
    if USE_VIDEO:
        ret, frame = cap.read()
        if not ret:
            print("Видео закончилось")
            logging.info("Видео Закончилось")
            break
    else:
        frame = cv2.imread(PHOTO_SOURCE).copy()
    
    # Обработка кадра
    results, thresh = process_frame(frame)
    
    # Распознавание букв
    letter_detected = False
    for result in results:
        width, height = map(int, cv2.minAreaRect(result['box'])[1])
        if width > 10 and height > 10:
            letter_crop = cv2.warpPerspective(
                thresh, 
                cv2.getPerspectiveTransform(
                    result['box'].astype("float32"),
                    np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
                ),
                (width, height)
            )
            
            if np.mean(letter_crop) < 250:
                
                processed_letter = letter_crop.copy()
                try:
                    letter, confidence = recognize_letter(letter_crop)
                    if confidence > CONFIDENCE_THRESHOLD:
                        letter_detected = True
                        last_detection_time = current_time
                        letter_stats[letter]['confidences'].append(confidence)
                        letter_stats[letter]['coords'].append(result['coords'])
                        letter_stats[letter]['timestamps'].append(current_time)
                        
                        processing_active = True
                        
                        text = f"{letter} ({confidence:.2f})"
                        cv2.putText(frame, text, result['center_px'], 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        print(f"Распознано: {letter} ({confidence:.2f}) | "
                              f"Смещение: {result['center_m'][0]:.2f}m, {result['center_m'][1]:.2f}m | "
                              f"Угол: {result['angle']:.1f}° | Координаты: {result['coords']}")
                        
                        logging.info(f"Распознано: {letter} ({confidence:.2f}) | "
                              f"Смещение: {result['center_m'][0]:.2f}m, {result['center_m'][1]:.2f}m | "
                              f"Угол: {result['angle']:.1f}° | Координаты: {result['coords']}")
                        
                        #Сохраняем кадр 
                         # Сохранение кадра (вынесено из внутреннего цикла)
                    
                    if letter_detected and (current_time - last_save_time > SAVE_INTERVAL):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(LETTERS_DIR, f"letter_{timestamp}.jpg")
                        
                        # Проверка перед сохранением
                        if not os.path.exists(filename):
                            cv2.imwrite(filename, frame)
                            print(f"Сохранено: {filename}")
                            last_save_time = current_time
                except Exception as e:
                    print(f"Ошибка обработки: {e}")
                    logging.error(f"Ошибка обработки: {e}")

    # Проверка таймаута и обработка результатов
    if processing_active and (current_time - last_detection_time) > LETTER_TIMEOUT:
        if letter_stats:
            best_letter, best_data = max(letter_stats.items(),
                                      key=lambda x: len(x[1]['confidences']))
            
            if len(best_data['confidences']) >= MIN_DETECTIONS:
                avg_conf = sum(best_data['confidences'])/len(best_data['confidences'])
                avg_lat = sum(c[0] for c in best_data['coords'])/len(best_data['coords'])
                avg_lon = sum(c[1] for c in best_data['coords'])/len(best_data['coords'])
                last_time = datetime.fromtimestamp(max(best_data['timestamps'])).strftime('%H:%M:%S')
                
                print("\n=== Результаты обработки ===")
                print(f"Наиболее вероятная буква: {best_letter}")
                print(f"Количество обнаружений: {len(best_data['confidences'])}")
                print(f"Средняя уверенность: {avg_conf:.2f}")
                print(f"Средние координаты: ({avg_lat:.6f}, {avg_lon:.6f})")
                print(f"Время фиксации: {last_time}")
                print("===========================")



                logging.info("\n=== Результаты обработки ===")
                logging.info(f"Наиболее вероятная буква: {best_letter}")
                logging.info(f"Количество обнаружений: {len(best_data['confidences'])}")
                logging.info(f"Средняя уверенность: {avg_conf:.2f}")
                logging.info(f"Средние координаты: ({avg_lat:.6f}, {avg_lon:.6f})")
                logging.info(f"Время фиксации: {last_time}")
                logging.info("===========================")
                
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
    cv2.imshow("Processed", processed_letter if processed_letter is not None else np.zeros((100,100,3), dtype=np.uint8))
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if not USE_VIDEO:
        cv2.waitKey(0)
        break

# Финализация последнего сеанса (если видео закончилось во время обработки)
if processing_active and letter_stats:
    best_letter, best_data = max(letter_stats.items(),
                               key=lambda x: len(x[1]['confidences']))
    
    if len(best_data['confidences']) >= MIN_DETECTIONS:
        avg_conf = sum(best_data['confidences'])/len(best_data['confidences'])
        avg_lat = sum(c[0] for c in best_data['coords'])/len(best_data['coords'])
        avg_lon = sum(c[1] for c in best_data['coords'])/len(best_data['coords'])
        last_time = datetime.fromtimestamp(max(best_data['timestamps'])).strftime('%H:%M:%S')
        
        print("\n=== Финальные результаты обработки ===")
        print(f"Наиболее вероятная буква: {best_letter}")
        print(f"Количество обнаружений: {len(best_data['confidences'])}")
        print(f"Средняя уверенность: {avg_conf:.2f}")
        print(f"Средние координаты: ({avg_lat:.6f}, {avg_lon:.6f})")
        print(f"Время фиксации: {last_time}")
        print("=====================================")
        


        logging.info("\n=== Финальные результаты обработки ===")
        logging.info(f"Наиболее вероятная буква: {best_letter}")
        logging.info(f"Количество обнаружений: {len(best_data['confidences'])}")
        logging.info(f"Средняя уверенность: {avg_conf:.2f}")
        logging.info(f"Средние координаты: ({avg_lat:.6f}, {avg_lon:.6f})")
        logging.info(f"Время фиксации: {last_time}")
        logging.info("=====================================")

        processing_sessions.append({
            'letter': best_letter,
            'count': len(best_data['confidences']),
            'avg_confidence': avg_conf,
            'avg_coords': (avg_lat, avg_lon),
            'timestamp': last_time
        })

# Вывод сводки
if USE_VIDEO and processing_sessions:
    print("\n=== Сводка всех сеансов обработки ===")
    logging.info("\n=== Сводка всех сеансов обработки ===")
    for i, session in enumerate(processing_sessions, 1):
        print(f"{i}. Буква: {session['letter']}")
        print(f"   Обнаружений: {session['count']}")
        print(f"   Уверенность: {session['avg_confidence']:.2f}")
        print(f"   Координаты: {session['avg_coords']}")
        print(f"   Время: {session['timestamp']}")
        print("-----------------------------")


        logging.info(f"{i}. Буква: {session['letter']}")
        logging.info(f"   Обнаружений: {session['count']}")
        logging.info(f"   Уверенность: {session['avg_confidence']:.2f}")
        logging.info(f"   Координаты: {session['avg_coords']}")
        logging.info(f"   Время: {session['timestamp']}")
        logging.info("-----------------------------")


if USE_VIDEO:
    cap.release()
cv2.destroyAllWindows()