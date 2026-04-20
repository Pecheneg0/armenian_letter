import cv2
import numpy as np
import time
import torch
import logging
import os
import sys
import math
from datetime import datetime
from pymavlink import mavutil
from torchvision import transforms
from PIL import Image
import cv2.aruco as aruco

# ===== КОНФИГУРАЦИЯ =====
MAVLINK_PORT = '/dev/ttyAMA0'
MAVLINK_BAUD = 57600
RC_CHANNEL = 6

MODE_OFF = 0
MODE_LETTERS = 1
MODE_ARUCO = 2

USE_PICAM = False
CAMERA_RESOLUTION = (640, 480)

MARKER_ID = 42
MARKER_REAL_SIZE = 0.2  # meters
ALIGNMENT_THRESHOLD = 0.05  # 5 cm
MIN_ALTITUDE = 1.0  # meters

MODEL_PATH = "armenian_letters_model_improved.pth"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.85

TELEMETRY_INTERVAL = 2
LOOP_DELAY = 0.1
NO_DATA_TIMEOUT = 5  # seconds

CMD_VISION_DATA = 50001
CMD_MODE_SWITCH = 50002

LOG_DIR = "/home/pi/tests/logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "drone_controller.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if USE_PICAM:
    from picamera2 import Picamera2

from modeln import ArmenianLetterNet

class DroneController:
    def __init__(self):
        self.master = None
        self.current_mode = MODE_OFF
        self.camera = None
        self.model = None
        self.labels = []
        self.last_offset_sent = 0
        self.aruco_detector = None
        self.altitude = 0.0
        self.last_telemetry = 0
        self.frame_counter = 0

        self.connect_to_pixhawk()
        self.init_aruco_detector()
        self.load_model()
        self.init_camera()
        logging.info("Контроллер инициализирован")

    def connect_to_pixhawk(self):
        """Установка соединения с Pixhawk"""
        while True:
            try:
                self.master = mavutil.mavlink_connection(
                    MAVLINK_PORT, 
                    baud=MAVLINK_BAUD
                )
                self.master.wait_heartbeat()
                self.send_ssh_message("✅ MAVLink подключен")
                break
            except Exception as e:
                self.log_error(f"Ошибка подключения: {str(e)}")
                time.sleep(5)

    def init_aruco_detector(self):
        """Инициализация детектора ArUco"""
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        self.aruco_detector = aruco.ArucoDetector(aruco_dict, parameters)

    def load_model(self):
        """Загрузка модели распознавания букв"""
        try:
            self.model = ArmenianLetterNet()
            state_dict = torch.load(MODEL_PATH, map_location='cpu')
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f.readlines()]
            
            self.send_ssh_message("✅ Модель загружена")
        except Exception as e:
            self.log_error(f"Ошибка загрузки модели: {str(e)}")
            raise

    def init_camera(self):
        """Инициализация камеры"""
        self.release_camera()
        try:
            if USE_PICAM:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": CAMERA_RESOLUTION, "format": "RGB888"}
                )
                self.camera.configure(config)
                self.camera.start()
                self.send_ssh_message("✅ PiCamera активирована")
            else:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                self.send_ssh_message("✅ USB-камера активирована")
        except Exception as e:
            self.log_error(f"Ошибка камеры: {str(e)}")


    def get_frame(self):
        """Захват кадра с камеры"""
        try:
            if USE_PICAM:
                return self.camera.capture_array()  # Для PiCamera
            else:
                ret, frame = self.camera.read()     # Для USB-камеры
                return frame if ret else None
        except Exception as e:
            self.log_error(f"Ошибка захвата кадра: {str(e)}")
            self.send_ssh_message("Ошибка захвата кадра")
            return None

    def handle_mavlink_commands(self):
        """Обработка входящих MAVLink команд"""
        while True:
            msg = self.master.recv_match(type='COMMAND_LONG', blocking=False)
            #self.send_ssh_message("Получена команда") 
            if not msg:
                break
                
            if msg.command == CMD_MODE_SWITCH:
                new_mode = int(msg.param1)
                self.change_mode(new_mode)

    def change_mode(self, new_mode):
        """Смена режима работы"""
        if new_mode == self.current_mode:
            return

        mode_names = {
            MODE_OFF: "OFF",
            MODE_LETTERS: "LETTERS",
            MODE_ARUCO: "ARUCO"
        }
        self.send_ssh_message(
            f"🔄 Смена режима: {mode_names.get(self.current_mode, 'UNKNOWN')} → "
            f"{mode_names.get(new_mode, 'UNKNOWN')}"
        )
        self.current_mode = new_mode
        self.init_camera()

    def process_frame(self):
        """Обработка кадра в зависимости от режима"""
        frame = self.get_frame()
        if frame is None:
            return

        if self.current_mode == MODE_ARUCO:
            self.process_aruco(frame)
        elif self.current_mode == MODE_LETTERS:
            self.process_letters(frame)

    def process_aruco(self, frame):
        """Обработка ArUco маркеров"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is None or MARKER_ID not in ids.flatten():
            self.send_null_offset()
            return

        try:
            msg = self.master.recv_match(
                type='GLOBAL_POSITION_INT',
                blocking=False,
                timeout=0.5
            )
            if not msg or msg.relative_alt < MIN_ALTITUDE*1000:
                self.send_ssh_message("⚠️ Низкая высота!")
                return

            self.altitude = msg.relative_alt / 1000.0
            idx = np.where(ids == MARKER_ID)[0][0]
            marker_corners = corners[idx][0]

            dx_px = marker_corners[:, 0].mean() - CAMERA_RESOLUTION[0]/2
            dy_px = CAMERA_RESOLUTION[1]/2 - marker_corners[:, 1].mean()
            
            pixel_size = MARKER_REAL_SIZE / np.linalg.norm(marker_corners[0] - marker_corners[1])
            dx_m = dx_px * pixel_size * self.altitude
            dy_m = dy_px * pixel_size * self.altitude

            if abs(dx_m) < ALIGNMENT_THRESHOLD and abs(dy_m) < ALIGNMENT_THRESHOLD:
                return

            if time.time() - self.last_offset_sent > 0.2:
                self.send_real_offset(dx_m, dy_m)
                self.last_offset_sent = time.time()
                self.send_ssh_message(
                    f"📍 Смещение: dx={dx_m:.2f}m dy={dy_m:.2f}m "
                    f"(alt={self.altitude:.1f}m)"
                )

        except Exception as e:
            self.log_error(f"Ошибка ArUco: {str(e)}")

    def send_real_offset(self, dx, dy):
        """Отправка смещения через MAVLink"""
        try:
            # Создаем стандартное MAVLink сообщение
            msg = self.master.mav.command_long_encode(
                target_system=self.master.target_system,
                target_component=self.master.target_component,
                command=CMD_VISION_DATA,  # 50001 - наш кастомный номер команды
                confirmation=0,
                param1=float(dx),  # Смещение по X (метры)
                param2=float(dy),  # Смещение по Y (метры)
                param3=0,
                param4=0,
                param5=0,
                param6=0,
                param7=0
            )

            self.master.mav.send(msg)

        except Exception as e:
            self.log_error(f"Ошибка отправки: {str(e)}")
            self.connect_to_pixhawk()

    def send_null_offset(self):
        """Отправка нулевого смещения"""
        self.send_real_offset(0.0, 0.0)

    def process_letters(self, frame):
        """Обработка распознавания букв"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])

            for cnt in contours:
                if cv2.contourArea(cnt) < 1000:
                    continue
                
                approx = cv2.approxPolyDP(cnt, 0.02*cv2.arcLength(cnt, True), True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    roi = gray[y:y+h, x:x+w]
                    
                    img_pil = Image.fromarray(roi)
                    tensor = transform(img_pil).unsqueeze(0)
                    
                    with torch.no_grad():
                        outputs = self.model(tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)
                        conf, pred = torch.max(probs, 1)
                        
                    if conf.item() > CONFIDENCE_THRESHOLD:
                        self.send_ssh_message(
                            f"🔤 Распознано: {self.labels[pred.item()]} ({conf.item():.2f})"
                        )
        except Exception as e:
            self.log_error(f"Ошибка распознавания: {str(e)}")

    def run(self):
        """Основной цикл работы"""
        self.send_ssh_message("🚀 Контроллер активирован")
        try:
            while True:
                self.handle_mavlink_commands()
                self.process_frame()
                time.sleep(LOOP_DELAY)
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        """Корректное завершение работы"""
        self.send_ssh_message("🛑 Выключение контроллера")
        self.release_camera()
        self.master.close()

    def release_camera(self):
        """Освобождение ресурсов камеры"""
        if self.camera:
            try:
                if USE_PICAM:
                    self.camera.stop()
                    self.camera.close()
                else:
                    self.camera.release()
            except Exception as e:
                self.log_error(f"Ошибка выключения камеры: {str(e)}")
            self.camera = None

    def log_error(self, message):
        """Логирование ошибок"""
        logging.error(message)
        self.send_ssh_message(f"⚠️ {message}")

    def send_ssh_message(self, message):
        """Отправка сообщения в консоль"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")
        sys.stdout.flush()

if __name__ == "__main__":
    controller = DroneController()
    controller.run()
