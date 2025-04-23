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
# MAVLink
MAVLINK_PORT = '/dev/ttyAMA0'
MAVLINK_BAUD = 57600
RC_CHANNEL = 6  # Канал для переключения режимов

# Режимы работы
MODE_OFF = 0
MODE_LETTERS = 1
MODE_ARUCO = 2

# Камеры
USE_PICAM = True  # True для PiCamera, False для USB
CAMERA_RESOLUTION = (640, 480)

# ArUco
MARKER_ID = 10
CENTER_TOLERANCE = 40

# Распознавание букв
MODEL_PATH = "armenian_letters_model.pth"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.85

# Логирование
LOG_DIR = "/home/pi/tests/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "sla_controller.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# ===== ОСНОВНОЙ КОНТРОЛЛЕР =====
class DroneController:
    def __init__(self):
        self.master = None
        self.current_mode = MODE_OFF
        self.camera = None
        self.model = None
        self.labels = []
        self.last_rc_value = 0
        self.rc_stable_count = 0
        self.proc_letters = None
        self.proc_aruco = None
        self.last_telemetry_time = 0
        self.current_mode_str = "OFF"
        
        # Инициализация
        self.connect_to_pixhawk()
        self.load_model()
        self.init_camera()
        logging.info("Контроллер инициализирован")

    def connect_to_pixhawk(self):
        """Подключение к Pixhawk с переподключением при ошибке"""
        while True:
            try:
                self.master = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD)
                self.master.wait_heartbeat()
                logging.info("✅ MAVLink подключен")
                break
            except Exception as e:
                logging.error(f"Ошибка подключения к Pixhawk: {e}. Повтор через 5 сек...")
                time.sleep(5)

    def load_model(self):
        """Загрузка модели для распознавания букв"""
        try:
            self.model = torch.load(MODEL_PATH, map_location='cpu')
            self.model.eval()
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f.readlines()]
            logging.info("✅ Модель загружена")
        except Exception as e:
            logging.error(f"⚠️ Ошибка загрузки модели: {e}")
            raise

    def init_camera(self):
        """Инициализация камеры с проверкой доступности"""
        try:
            if USE_PICAM:
                from picamera2 import Picamera2
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": CAMERA_RESOLUTION, "format": "RGB888"}
                )
                self.camera.configure(config)
                logging.info("✅ PiCamera инициализирована")
            else:
                self.camera = cv2.VideoCapture(0)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                logging.info("✅ USB-камера инициализирована")
        except Exception as e:
            logging.error(f"⚠️ Ошибка инициализации камеры: {e}")
            raise



####### Отпарака телеметрии и смены режимов на ноутбук по ssh 
    def send_ssh_message(self, message):
        """Отправка сообщения в stdout (будет видно в SSH-сессии)"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        sys.stdout.flush()  # Очистка буфера для мгновенного вывода

    def monitor_telemetry(self):
        """Чтение и отправка телеметрии каждые 10 секунд"""
        self.send_ssh_message("🔵 Режим OFF: мониторинг телеметрии")
        while self.current_mode == MODE_OFF:
            current_time = time.time()
            if current_time - self.last_telemetry_time >= 10:  # Раз в 10 сек
                try:
                    msg_att = self.master.recv_match(type='ATTITUDE', blocking=True, timeout=1)
                    msg_gps = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
                    msg_bat = self.master.recv_match(type='SYS_STATUS', blocking=True, timeout=1)
                    
                    if msg_att and msg_gps and msg_bat:
                        telemetry = (
                            f"📊 Телеметрия: "
                            f"Высота={msg_gps.relative_alt / 1000:.1f} м, "
                            f"Скорость={msg_gps.vel / 100:.1f} м/с, "
                            f"Координаты={msg_gps.lat / 1e7:.6f}, {msg_gps.lon / 1e7:.6f}, "
                            f"Углы: pitch={math.degrees(msg_att.pitch):.1f}°, "
                            f"roll={math.degrees(msg_att.roll):.1f}°, "
                            f"yaw={math.degrees(msg_att.yaw):.1f}°, "
                            f"Батарея={msg_bat.voltage_battery / 1000:.1f} V"
                        )
                        self.send_ssh_message(telemetry)
                        self.last_telemetry_time = current_time
                
                except Exception as e:
                    self.send_ssh_message(f"⚠️ Ошибка телеметрии: {str(e)}")
                    self.connect_to_pixhawk()
            
            time.sleep(0.1)

    def handle_mode_switch(self, new_mode):
        mode_names = {MODE_OFF: "OFF", MODE_LETTERS: "LETTERS", MODE_ARUCO: "ARUCO"}
        new_mode_str = mode_names.get(new_mode, "UNKNOWN")
    
        if new_mode_str != self.current_mode_str:
            self.send_ssh_message(f"🔄 Переключение режима: {self.current_mode_str} → {new_mode_str}")
            self.current_mode_str = new_mode_str

######

    def release_camera(self):
        """Корректное освобождение камеры"""
        if self.camera:
            if USE_PICAM:
                self.camera.stop()
            else:
                self.camera.release()
            self.camera = None
            logging.info("Камера освобождена")

    def get_frame(self):
        """Получение кадра с камеры"""
        try:
            if USE_PICAM:
                return self.camera.capture_array()
            else:
                ret, frame = self.camera.read()
                return frame if ret else None
        except Exception as e:
            logging.error(f"Ошибка захвата кадра: {e}")
            return None

    def process_aruco(self):
        """Обработка ArUco маркеров"""
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        
        logging.info("Режим ArUco: запущен")
        while self.current_mode == MODE_ARUCO:
            frame = self.get_frame()
            if frame is None:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None and MARKER_ID in ids.flatten():
                logging.info(f"Обнаружен маркер ID: {MARKER_ID}")
                self.land()
                break
                
            time.sleep(0.1)
        logging.info("Режим ArUco: остановлен")

    def process_letters(self):
        """Обработка распознавания букв"""
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        logging.info("Режим Letters: запущен")
        while self.current_mode == MODE_LETTERS:
            frame = self.get_frame()
            if frame is None:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                area = cv2.contourArea(cnt)
                
                if len(approx) == 4 and area > 1000:
                    x, y, w, h = cv2.boundingRect(approx)
                    letter_img = gray[y:y+h, x:x+w]
                    img_pil = Image.fromarray(letter_img)
                    img_tensor = transform(img_pil).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = self.model(img_tensor)
                        probabilities = torch.nn.functional.softmax(output, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        confidence = confidence.item()
                        label = predicted.item()
                    
                    if confidence > CONFIDENCE_THRESHOLD:
                        logging.info(f"Обнаружена буква: {self.labels[label]}, уверенность: {confidence:.2f}")
            
            time.sleep(0.2)
        logging.info("Режим Letters: остановлен")

    def land(self):
        """Процедура посадки"""
        logging.info("Начало посадки...")
        self.master.set_mode_apm("LAND")
        time.sleep(5)
        self.disarm()
        logging.info("Посадка завершена")

    def disarm(self):
        """Выключение моторов"""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )
        logging.info("Моторы выключены")

    def check_rc_mode(self):
        """Проверка текущего режима по RC-каналу с фильтрацией дребезга"""
        msg = self.master.recv_match(type="RC_CHANNELS", blocking=True, timeout=1)
        if msg:
            rc_value = getattr(msg, f'chan{RC_CHANNEL}_raw')
            
            # Фильтр дребезга (ждём 3 одинаковых значения подряд)
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
        """Основной цикл работы"""
        try:
            logging.info("🚀 Контроллер запущен")
            while True:
                new_mode = self.check_rc_mode()
                
                if new_mode != self.current_mode:
                    self.handle_mode_switch (new_mode)
                    logging.info(f"Переключение режима: {self.current_mode} -> {new_mode}")
                    self.current_mode = new_mode
                    self.release_camera()
                    
                    if self.current_mode == MODE_OFF:
                        logging.info("Режим OFF")
                    elif self.current_mode == MODE_LETTERS:
                        self.init_camera()
                        self.process_letters()
                    elif self.current_mode == MODE_ARUCO:
                        self.init_camera()
                        self.process_aruco()
                
                time.sleep(0.2)
                
        except KeyboardInterrupt:
            logging.info("Завершение работы...")
            self.release_camera()
            self.disarm()

# ===== ЗАПУСК =====
if __name__ == "__main__":
    controller = DroneController()
    controller.run()