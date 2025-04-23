import cv2
import numpy as np
import time
import math
from collections import deque
from pymavlink import mavutil
from datetime import datetime

# === НАСТРОЙКИ ===
CAM_WIDTH = 640
CAM_HEIGHT = 480
MARKER_ID = 10
CENTER_TOLERANCE = 5
THRUST = 1400
MAX_SPEED = 500
SCALE = 2.0
SMOOTHING_WINDOW = 3
LOG_PATH = "/home/pi/tests/logs/aruco_log.csv"
SEARCH_TIMEOUT = 60  # секунд

# === MAVLINK ПОДКЛЮЧЕНИЕ ===
master = mavutil.mavlink_connection('/dev/ttyAMA0', baud=57600)
master.wait_heartbeat()
print("✅ MAVLink: Подключено")

master.set_mode_apm("GUIDED")
time.sleep(1)

# ARM
master.mav.command_long_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
    0, 1, 21196, 0, 0, 0, 0, 0
)
time.sleep(2)

# === КАМЕРА USB ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
time.sleep(1)

# === ArUco ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

# === Буферы сглаживания ===
offset_x_buf = deque(maxlen=SMOOTHING_WINDOW)
offset_y_buf = deque(maxlen=SMOOTHING_WINDOW)

# === ЛОГ ===
with open(LOG_PATH, "w") as f:
    f.write("timestamp,offset_x,offset_y,event\n")

def get_center(corners):
    return np.mean(corners[0], axis=0)

def get_smoothed_offset(buffer, new_value):
    buffer.append(new_value)
    return sum(buffer) / len(buffer)

# === ОСНОВНОЙ ЦИКЛ ===
start_time = time.time()
found_marker = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Ошибка захвата изображения")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None and MARKER_ID in ids.flatten():
        found_marker = True
        idx = np.where(ids.flatten() == MARKER_ID)[0][0]
        center = get_center(corners[idx])

        offset_x = center[0] - CAM_WIDTH / 2
        offset_y = CAM_HEIGHT / 2 - center[1]  # Инверсия по Y

        smooth_x = get_smoothed_offset(offset_x_buf, offset_x)
        smooth_y = get_smoothed_offset(offset_y_buf, offset_y)

        print(f"📍 Смещение: X={smooth_x:.1f}, Y={smooth_y:.1f}")

        with open(LOG_PATH, "a") as f:
            f.write(f"{timestamp},{smooth_x:.1f},{smooth_y:.1f},tracking\n")

        if abs(smooth_x) < CENTER_TOLERANCE and abs(smooth_y) < CENTER_TOLERANCE:
            print("✅ Центрировано — начинаем посадку")
            with open(LOG_PATH, "a") as f:
                f.write(f"{timestamp},{smooth_x:.1f},{smooth_y:.1f},centered\n")

            master.set_mode_apm("LAND")
            time.sleep(2)

            for pwm in range(THRUST, 1150, -50):
                print(f"🕹️ Снижение тяги: {pwm}")
                master.mav.manual_control_send(master.target_system, 0, 0, pwm, 0, 0)
                time.sleep(0.8)

            master.mav.command_long_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 0, 0, 0, 0, 0, 0, 0
            )
            print("🔴 DISARM — посадка завершена")
            with open(LOG_PATH, "a") as f:
                f.write(f"{timestamp},0,0,disarm\n")
            break
        else:
            vx = int(np.clip(-smooth_x * SCALE, -MAX_SPEED, MAX_SPEED))
            vy = int(np.clip(-smooth_y * SCALE, -MAX_SPEED, MAX_SPEED))
            master.mav.manual_control_send(master.target_system, vx, vy, THRUST, 0, 0)
    else:
        print("🔍 Поиск маркера...")
        with open(LOG_PATH, "a") as f:
            f.write(f"{timestamp},0,0,not_found\n")

        # ⏱️ Прерывание, если маркер не найден за SEARCH_TIMEOUT
        if not found_marker and time.time() - start_time > SEARCH_TIMEOUT:
            print("⏹️ Время поиска маркера истекло, завершение работы.")
            with open(LOG_PATH, "a") as f:
                f.write(f"{timestamp},0,0,timeout_exit\n")
            break

    time.sleep(0.3)

cap.release()
