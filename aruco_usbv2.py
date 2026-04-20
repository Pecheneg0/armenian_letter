import cv2
import numpy as np
import time
from datetime import datetime
import os

# ===== КОНФИГУРАЦИЯ =====
CAMERA_ID = 1
# ID камеры (обычно 0 или 1)
CAMERA_RESOLUTION = (720, 720)
MARKER_42 = 42  # ID искомой метки
MARKER_41 = 41
LENGTH_41 = 0.08 # Длина стороны маркера в метрах
LENGTH_42 = 0.05
ARUCO_DIR = "/home/pi/Desktop/videos"
PROCESSING_FPS = 60  # Частота обработки (кадров в секунду)
LOOP_DELAY = 0.02  # Задержка между кадрами

TARGET_ALTITUDE = 0.2  # Высота переключения маркеров
current_target_id = MARKER_42  # Начинаем с маркера 42
current_length = LENGTH_42

os.makedirs(ARUCO_DIR, exist_ok=True)

# Параметры камеры
frame_w, frame_h = 720, 720
camera_matrix = np.array([
    [1000, 0, frame_w / 2],
    [0, 1000, frame_h / 2],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# Инициализация детектора ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Инициализация камеры
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ОШИБКА: Не удалось открыть камеру!")
    exit()

# Установка параметров камеры для работы на солнце
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
cap.set(cv2.CAP_PROP_FPS, PROCESSING_FPS)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # 1 = manual exposure
cap.set(cv2.CAP_PROP_EXPOSURE, 0)     # Уменьшаем экспозицию (значение подбирается экспериментально)
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0)   # Средняя яркость
cap.set(cv2.CAP_PROP_CONTRAST, 0)     # Увеличиваем контраст
cap.set(cv2.CAP_PROP_SATURATION, 0)   # Уменьшаем насыщенность
cap.set(cv2.CAP_PROP_SHARPNESS, 0)   # Максимальная резкость
cap.set(cv2.CAP_PROP_GAIN, 0)          # Минимальное усиление
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 4500) 
out = cv2.VideoWriter('ARUCO_DIR/output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 12, (1280, 720))
print(f"[{datetime.now().strftime('%H:%M:%S')}] === Старт системы поиска ArUco маркеров ===")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Начальный целевой маркер: {current_target_id}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Разрешение камеры: {CAMERA_RESOLUTION}")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Частота обработки: {PROCESSING_FPS} FPS")



try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ОШИБКА: Проблема с захватом кадра")
            break
        
        corners, ids, _ = detector.detectMarkers(frame)

        if ids is not None and current_target_id in ids.flatten():
            idx = list(ids.flatten()).index(current_target_id)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], current_length, camera_matrix, dist_coeffs
            )
            
            altitude = tvecs[0][0][2]
            tvec = tvecs[0][0]
            offset_north = -tvec[0] 
            offset_east = tvec[1]
            
            # Форматированный вывод в консоль
            print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Маркер {current_target_id} | "
                  f"Высота: {altitude:.3f}m | "
                  f"N: {offset_north:.3f}m | "
                  f"E: {offset_east:.3f}m | "
                  f"Размер: {current_length}m")
            
            # Проверка переключения маркера
            if current_target_id == MARKER_42 and altitude <= TARGET_ALTITUDE:
                current_target_id = MARKER_41
                current_length = LENGTH_41
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ПЕРЕКЛЮЧЕНИЕ: "
                      f"Новый целевой маркер {current_target_id} (достигнута высота {altitude:.2f}m ≤ {TARGET_ALTITUDE}m)")
                
            # Отрисовка маркера
            cv2.aruco.drawDetectedMarkers(frame, [corners[idx]])
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.1)
            
        elif ids is not None:
            # Вывод информации о других обнаруженных маркерах (для отладки)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Обнаружены другие маркеры: {ids.flatten()} (целевой: {current_target_id})")
        else:
            # Периодический вывод информации об отсутствии маркеров
            if time.time() % 5 < 0.1:  # Каждые ~5 секунд
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Поиск маркера {current_target_id}...")

        # Отображение кадра (может не работать в SSH без X11 forwarding)
        cv2.imshow("Original", frame)
        #cv2.imshow("Gray", gray)
        #cv2.imshow("Binary", binary)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Ручное прерывание пользователем")
            break

        time.sleep(LOOP_DELAY)

except KeyboardInterrupt:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Завершение работы по сигналу KeyboardInterrupt")
except Exception as e:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Ресурсы освобождены, программа завершена")
