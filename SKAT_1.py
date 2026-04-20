import cv2
import numpy as np
import time
import torch
from pymavlink import mavutil
from torchvision import transforms
from PIL import Image
import os
import math
from collections import deque
import logging
import signal
import sys
from datetime import datetime 

# ===== КОНФИГУРАЦИЯ =====
MAVLINK_PORT = '/dev/ttyAMA0'
MAVLINK_BAUD = 57600
RC_CHANNEL = 6

MODE_OFF = 0
MODE_LETTERS = 1
MODE_ARUCO = 2

USE_PICAM = False
CAMERA_RESOLUTION = (1920, 1080)

MARKER_ID = 42
CENTER_TOLERANCE = 150

MODEL_PATH = "armenian_letters_model_improved.pth"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.85

TELEMETRY_INTERVAL = 2
LOOP_DELAY = 0.01

LOG_DIR = "/home/pi/tests/logs"
ARUCO_IMAGE_DIR = "/home/pi/tests/aruco_images"
LETTERS_IMAGE_DIR = "/home/pi/tests/letter_images"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARUCO_IMAGE_DIR, exist_ok=True)
os.makedirs(LETTERS_IMAGE_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "sla_controller.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if USE_PICAM:
    from picamera2 import Picamera2

from modeln import ArmenianLetterNet

class DroneController:
    def __init__(self):
        self.master = None
        self.current_mode = MODE_LETTERS # MODE_OFF
        self.camera = None
        self.model = None
        self.labels = []
        self.last_rc_value = 0
        self.rc_stable_count = 0
        self.proc_letters = None
        self.proc_aruco = None
        self.last_telemetry_time = 0
        self.current_mode_str = "OFF"

        self.connect_to_pixhawk()
        self.load_model()
        logging.info("Контроллер инициализирован")

    def connect_to_pixhawk(self):
        while True:
            try:
                self.master = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD)
                self.master.wait_heartbeat()
                logging.info("MAVLink подключен")
                break
            except Exception as e:
                logging.error(f"Ошибка подключения к Pixhawk: {e}. Повтор через 5 сек...")
                time.sleep(5)

    def send_ssh_message(self, message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        sys.stdout.flush()

    def load_model(self):
        try:
            self.model = ArmenianLetterNet()
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f.readlines()]
            logging.info("Модель загружена")
        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {e}")
            raise

    def init_camera(self):
        try:
            self.release_camera()

            if USE_PICAM:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": CAMERA_RESOLUTION, "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                logging.info("PiCamera запущена")
            else:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                logging.info("USB-камера запущена")
        except Exception as e:
            logging.error(f"Ошибка запуска камеры: {e}")
            raise
    

    def send_offset_to_pixhawk(self, offset_north, offset_east, altitude):
        try:
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                50001,  # кастомная команда, должна быть обработана в Lua
                0,  # confirmation
                offset_north,
                offset_east,
                altitude,
                0, 0, 0, 0
            )
            self.send_ssh_message(f"Отправлено смещение: N={offset_north:.2f}, E={offset_east:.2f}, Alt={altitude:.2f}")
        except Exception as e:
            self.send_ssh_message(f"Ошибка отправки offset: {e}")




    def release_camera(self):
        if self.camera:
            try:
                if USE_PICAM:
                    self.camera.stop()
                    self.camera.close()
                    logging.info("PiCamera остановлена и закрыта")
                else:
                    self.camera.release()
                    logging.info("USB-камера остановлена")
            except Exception as e:
                logging.warning(f"Ошибка при остановке камеры: {e}")
            self.camera = None

    def get_frame(self):
        try:
            if USE_PICAM:
                return self.camera.capture_array()
            else:
                ret, frame = self.camera.read()
                return frame if ret else None
        except Exception as e:
            logging.error(f"Ошибка захвата кадра: {e}")
            return None

    def calculate_offset_north_east(self, rvec, tvec, drone_yaw_deg):
        x_cam = tvec[0][0]
        z_cam = tvec[0][1]
        altitude = tvec[0][2]
        yaw_rad = np.radians(drone_yaw_deg)
        rotation_matrix = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad),  np.cos(yaw_rad)]
        ])
        offset_camera = np.array([x_cam, z_cam])
        offset_world = rotation_matrix @ offset_camera
        offset_east = offset_world[0]
        offset_north = offset_world[1]
        return offset_north, offset_east, altitude

    def process_aruco(self):
        from cv2 import aruco
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        self.send_ssh_message("Поиск ArUco маркера (USB-камера)...")
        last_save_time = time.time()

        drone_yaw_deg = 0.0  # заменить на значение с автопилота

        while self.current_mode == MODE_ARUCO:
            new_mode = self.check_rc_mode()
            if new_mode != self.current_mode:
                self.send_ssh_message("Выход из режима ARUCO")
                self.current_mode = new_mode
                return

            frame = self.get_frame()
            if frame is None:
                time.sleep(LOOP_DELAY)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if ids is not None and MARKER_ID in ids.flatten():
                idx = list(ids.flatten()).index(MARKER_ID)
                pts = corners[idx][0]
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers([corners[idx]], 0.18, np.array([[1000, 0, CAMERA_RESOLUTION[0]//2], [0, 1000, CAMERA_RESOLUTION[1]//2], [0, 0, 1]], dtype=np.float32), np.zeros((5, 1), dtype=np.float32))
                offset_north, offset_east, altitude = self.calculate_offset_north_east(rvecs[0], tvecs[0], drone_yaw_deg)
                self.send_ssh_message(f"📍 Offset: N={offset_north:.2f}, E={offset_east:.2f}, Alt={altitude:.2f}")
                self.send_offset_to_pixhawk(offset_north, offset_east, altitude)

                if time.time() - last_save_time > 10:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(ARUCO_IMAGE_DIR, f"aruco_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    self.send_ssh_message(f" Сохранено изображение: {filename}")
                    last_save_time = time.time()

            time.sleep(LOOP_DELAY)

    def process_letters(self):
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.send_ssh_message("🟣 Распознавание букв...")
        last_save_time = time.time()

        while self.current_mode == MODE_LETTERS:
            new_mode = self.check_rc_mode()
            if new_mode != self.current_mode:
                self.send_ssh_message(f"🔁 Выход из режима LETTERS")
                self.current_mode = new_mode
                return

            frame = self.get_frame()
            if frame is None:
                time.sleep(LOOP_DELAY)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                area = cv2.contourArea(cnt)
                if len(approx) == 4 and area > 1000:
                    x, y, w, h = cv2.boundingRect(approx)
                    letter_img = gray[y:y + h, x:x + w]
                    img_pil = Image.fromarray(letter_img)
                    img_tensor = transform(img_pil).unsqueeze(0)

                    with torch.no_grad():
                        output = self.model(img_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        confidence = confidence.item()
                        label = predicted.item()

                    if confidence > CONFIDENCE_THRESHOLD:
                        result = f"Буква: {self.labels[label]} ({confidence:.2f})"
                        self.send_ssh_message(result)
                        logging.info(result)

                        if time.time() - last_save_time > 10:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                            img_path = os.path.join(LETTERS_IMAGE_DIR, f"letter_{timestamp}.jpg")
                            cv2.imwrite(img_path, frame)
                            self.send_ssh_message(f"Сохранено изображение: {img_path}")
                            last_save_time = time.time()

            time.sleep(LOOP_DELAY)

    def check_rc_mode(self):
        msg = self.master.recv_match(type="RC_CHANNELS", blocking=True, timeout=1)
        if msg:
            rc_value = getattr(msg, f'chan{RC_CHANNEL}_raw')
            if rc_value == self.last_rc_value:
                self.rc_stable_count += 1
            else:
                self.rc_stable_count = 0
                self.last_rc_value = rc_value

            if self.rc_stable_count >= 3:
                if rc_value < 1200:
                    return MODE_OFF
                elif 1300 < rc_value < 1700:
                    return MODE_LETTERS
                elif rc_value > 1800:
                    return MODE_ARUCO
        return self.current_mode

    def run(self):
        self.send_ssh_message("🚀 Запуск контроллера")
        try:
            while True:
                new_mode = self.check_rc_mode()

                if new_mode != self.current_mode:
                    self.send_ssh_message(f"Режим: {self.current_mode_str} → {new_mode}")
                    self.current_mode = new_mode
                    self.current_mode_str = {MODE_OFF: "OFF", MODE_LETTERS: "LETTERS", MODE_ARUCO: "ARUCO"}.get(new_mode, "UNKNOWN")
                    self.release_camera()

                    if self.current_mode == MODE_LETTERS:
                        self.init_camera()
                        self.process_letters()
                    elif self.current_mode == MODE_ARUCO:
                        self.init_camera()
                        self.process_aruco()

                time.sleep(LOOP_DELAY)
        except KeyboardInterrupt:
            self.send_ssh_message("---  Stop all scripts  ---")
            self.release_camera()

if __name__ == '__main__':
    controller = DroneController()
    controller.run()
