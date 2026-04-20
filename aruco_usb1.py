import cv2
import numpy as np
import time
from datetime import datetime
import os

# ===== КОНФИГУРАЦИЯ =====
CAMERA_ID = 0 # ID камеры (обычно 0 или 1)
CAMERA_RESOLUTION = (720, 480)
MARKER_42 = 42  # ID искомой метки
MARKER_41 = 41
LENGTH_41 = 0.08# Длина стороны маркера в метрах
LENGTH_42 = 0.32
ARUCO_DIR = "/home/pi/tests/aruco_images"  # Папка для сохранения снимков
PROCESSING_FPS = 60  # Частота обработки (кадров в секунду)
LOOP_DELAY = 0.1  # Задержка между кадрами

os.makedirs(ARUCO_DIR, exist_ok=True)

# Параметры камеры (замените на свои, если известны)
frame_w, frame_h = 720, 480
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
    print("Ошибка: Не удалось открыть камеру!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
cap.set(cv2.CAP_PROP_FPS, PROCESSING_FPS)

last_save_time = time.time()

print("=== Поиск ArUco маркера ===")
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        # Сохранение снимка каждые 10 секунд
      # if time.time() - last_save_time > 10:
       #    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #   filename = os.path.join(ARUCO_DIR, f"aruco_{timestamp}.jpg")
         #  cv2.imwrite(filename, frame)
          # print(f"Сохранено: {filename}")
           #last_save_time = time.time()

        # Если найден маркер с нужным ID
        if ids is not None :
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in [MARKER_42, MARKER_41]:
                    current_length = LENGTH_41 if marker_id == MARKER_41 else LENGTH_42
            
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers([corners[i]],current_length,  camera_matrix  , dist_coeffs ) 

                # Отрисовка маркера и осей
                cv2.aruco.drawDetectedMarkers(frame, [corners[i]])
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.1)

                # Вывод смещения (в метрах)
                k = 0.5
                tvec = tvecs[0][0]
                N =  -tvec[0] 
                E =  tvec[1] 
                Z = k * tvec[2]
                 
                print(f"Смещение: X={N:.2f}m, Y={E:.2f}m, Z={Z:.2f}m")

        # Отображение кадра
        cv2.imshow("ArUco Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(LOOP_DELAY)

except KeyboardInterrupt:
    print("Завершение работы...")
finally:
    cap.release()
    cv2.destroyAllWindows()
