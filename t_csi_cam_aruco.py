import cv2
import numpy as np
import time
import os
from datetime import datetime
from picamera2 import Picamera2
from libcamera import controls

# === Настройки ===
MARKER_ID = 42
MARKER_LENGTH = 0.18  # в метрах
SAVE_DIR = "/home/pi/aruco_test_output"
os.makedirs(SAVE_DIR, exist_ok=True)

print("📡 Поиск ArUco маркера через CSI-камеру... (нажми 'q' для выхода)")

# === Параметры изображения ===
frame_w, frame_h = 640, 480
camera_matrix = np.array([
    [1000, 0, frame_w / 2],
    [0, 1000, frame_h / 2],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# === Инициализация Picamera2 ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (frame_w, frame_h), "format": "RGB888"})
picam2.configure(config)
picam2.start()

# === ArUco ===
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

last_save_time = time.time()
drone_yaw_deg = 0.0  # заменить на значение с компаса при необходимости

def calculate_offset(rvec, tvec, yaw_deg):
    x_cam = tvec[0][0]
    z_cam = tvec[0][1]
    alt = tvec[0][2]

    yaw_rad = np.radians(yaw_deg)
    R = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad)],
        [np.sin(yaw_rad),  np.cos(yaw_rad)]
    ])
    offset = R @ np.array([x_cam, z_cam])
    return offset[1], offset[0], alt  # North, East, Altitude

# === Главный цикл ===
while True:
    frame_rgb = picam2.capture_array()
    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is not None and MARKER_ID in ids.flatten():
        idx = list(ids.flatten()).index(MARKER_ID)
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            [corners[idx]], MARKER_LENGTH, camera_matrix, dist_coeffs)

        offset_north, offset_east, altitude = calculate_offset(rvecs[0], tvecs[0], drone_yaw_deg)
        print(f"✅ Offset: N = {offset_north:.2f} m, E = {offset_east:.2f} m, Alt = {altitude:.2f} m")

        cv2.aruco.drawDetectedMarkers(frame, [corners[idx]])
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.1)

        if time.time() - last_save_time > 10:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SAVE_DIR, f"aruco_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"💾 Сохранено изображение: {filename}")
            last_save_time = time.time()

    cv2.imshow("Aruco CSI Camera Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()
