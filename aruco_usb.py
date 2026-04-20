import cv2
import numpy as np
import time
from datetime import datetime
import os

# ===== КОНФИГУРАЦИЯ =====
CAMERA_ID = 0 # ID камеры (обычно 0 или 1)
CAMERA_RESOLUTION = (720  ,480)
MARKER_42 = 42  # ID искомой метки
MARKER_41 = 41
LENGTH_41 = 0.08# Длина стороны маркера в метрах
LENGTH_42 = 0.32
ARUCO_DIR = "/home/pi/tests/aruco_images"  # Папка для сохранения снимков
PROCESSING_FPS = 60  # Частота обработки (кадров в секунду)
LOOP_DELAY = 0.1  # Задержка между кадрами


TARGET_ALTITUDE = 1.2  # Высота переключения маркеров
current_target_id = MARKER_42  # Начинаем с маркера 42
current_length = LENGTH_42
    

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

        if ids is not None and current_target_id in ids.flatten():
            idx = list(ids.flatten()).index(current_target_id)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                [corners[idx]], current_length, camera_matrix, dist_coeffs
            )
            
            # Получаем реальную высоту (Z-координата)
            altitude = 0.5 * tvecs[0][0][2]
            
            # Проверка необходимости переключения маркера
            if current_target_id == MARKER_42 and altitude <= TARGET_ALTITUDE:
                current_target_id = MARKER_41
                current_length = LENGTH_41
                print(f"Переключение на маркер {current_target_id} (высота ≤ {TARGET_ALTITUDE}m)")
                continue
                
            # Обработка только активного маркера
            #offset_north, offset_east, _ = self.calculate_offset(tvecs[0])
            tvec = tvecs[0][0]
            offset_north = -tvec[0] 
            offset_east = tvec[1]
            # Отладочная информация
            (f"Маркер {current_target_id} | Высота: {altitude:.2f}m | N: {offset_north:.2f} | E: {offset_east:.2f}")
            
            # Отрисовка
            cv2.aruco.drawDetectedMarkers(frame, [corners[idx]])
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.1)
            

            # Запуск процедуры посадки
            #self.execute_landing(offset_north, offset_east, altitude)


            
        # Управление режимами полёта
        #if not self.drone_in_vtol:
            #self.switch_to_vtol_mode()
            
        #self.master.set_mode("QSTABILIZE")
       
        # Отображение кадра
        cv2.imshow("ArUco Detection", frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
            #break

        time.sleep(LOOP_DELAY)

except KeyboardInterrupt:
    print("Завершение работы...")
finally:
    cap.release()
    cv2.destroyAllWindows()

