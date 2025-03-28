import cv2
import numpy as np
import time
from picamera2 import Picamera2
import cv2.aruco as aruco

# Настройка камеры
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"size": (3280, 2464), "format": "RGB888"}))
camera.start()

dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

while True:
    frame = camera.capture_array("main")
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None:
        for marker_id in ids.flatten():
            print(f"Обнаружен ArUco-маркер с ID: {marker_id}")
    
    time.sleep(2.5)
