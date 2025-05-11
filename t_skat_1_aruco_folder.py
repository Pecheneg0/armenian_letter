import cv2
import numpy as np
import time
import os
from datetime import datetime

# === Настройки ===
CAMERA_ID = 0
MARKER_ID = 42
MARKER_LENGTH = 0.18  # в метрах
SAVE_DIR = "/Users/aleksandr/Desktop/aruco_test_output"
os.makedirs(SAVE_DIR, exist_ok=True)

print("📡 Поиск ArUco маркера... (нажми 'q' для выхода)")

# === Параметры камеры ===
frame_w, frame_h = 640, 480
camera_matrix = np.array([
    [1000, 0, frame_w / 2],
    [0, 1000, frame_h / 2],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# === Камера ===
cap = cv2.VideoCapture(CAMERA_ID)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

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

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Кадр не получен")
        continue

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

    cv2.imshow("Aruco Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
