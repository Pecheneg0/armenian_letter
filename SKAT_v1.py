import cv2
import numpy as np
import time
import torch
from pymavlink import mavutil
from torchvision import transforms
from PIL import Image
import os
import math
import logging
import sys
from datetime import datetime 
from picamera2 import Picamera2
from libcamera import controls

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
CAMERA_ID = 1

MODEL_PATH = "ALM_best.pth"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.85

TELEMETRY_INTERVAL = 2
LOOP_DELAY = 0.1

PREVIEW_RES = (640, 360)      # Разрешение для отображения
CONFIDENCE_THRESHOLD = 0.85   # Порог уверенности распознавания
PROCESSING_FPS = 10           # Ограничение частоты обработки (кадров в секунду)
PIXELS_PER_METER = 100  

ARUCO_DIR = "/home/pi/tests/aruco_images"
LETTERS_DIR = "/home/pi/tests/letters_images"
LOG_DIR = "/home/pi/tests/logs"
MARKER_LENGTH = 0.26 

# Глобальные координаты (пример)
home_lat = 55.754107
home_lon = 37.861527
drone_yaw_deg = 0.0  
 
frame_w, frame_h = 1920, 1080
camera_matrix = np.array([
    [1000, 0, frame_w / 2],
    [0, 1000, frame_h / 2],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "sla_controller.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if USE_PICAM:
    from picamera2 import Picamera2

from modelold import ArmenianLetterNet

class DroneController:
    def __init__(self):
        self.master = None
        self.current_mode = MODE_OFF
        self.camera = None
        self.model = None
        self.labels = []
        self.rc_stable_count = 0
        self.last_telemetry_time = 0
        self.current_mode_str = "OFF"
        self.last_rc_value = None
        self.last_wp_value = None        
        self.process_ARUCO = None
        self.process_LETTERS = None
        self.no_process = None
        self.transform = None

        # Для обработки букв
        self.last_processing_time = 0
        self.last_save_time = 0
        self.processing_interval = 1.0 / PROCESSING_FPS

        self.connect_to_pixhawk()
        self.load_model()
        self.logger = logging.getLogger("modeswitcher")
        self.logger.info("Инициализация контроллера")
        
        # Соответствие точек миссии режимам
        self.wp_modes = {9: MODE_LETTERS, 17: MODE_ARUCO, 0: MODE_OFF}
        
        # Инициализация преобразований изображения
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
    def connect_to_pixhawk(self):
        while True:
            try:
                self.master = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD)
                self.master.wait_heartbeat()
                self.logger.info(f"Подключено к системе {self.master.target_system}")
                break
            except Exception as e:
                logging.error(f"Ошибка подключения к Pixhawk: {e}. Повтор через 5 сек...")
                time.sleep(5)

    def get_current_waypoint(self):
        msg = self.master.recv_match(type='MISSION_CURRENT', blocking=False)
        return msg.seq if msg else None

    def get_rc_value(self):
        msg = self.master.recv_match(type='RC_CHANNELS', blocking=False)
        if msg:
            try:
                return getattr(msg, f'chan{RC_CHANNEL}_raw')
            except AttributeError:
                pass
        return None

    def rc_to_mode(self, rc_value):
        if rc_value < 1200: return MODE_OFF
        if rc_value < 1700: return MODE_LETTERS
        return MODE_ARUCO

    def send_ssh_message(self, message):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        sys.stdout.flush()

    def load_model(self):
        try:
            self.model = ArmenianLetterNet()
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            self.model.eval()
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f.readlines()]
            logging.info(" Модель загружена")
        except Exception as e:
            logging.error(f" Ошибка загрузки модели: {e}")
            raise

    def init_camera(self):
        try:
            self.release_camera()

            if USE_PICAM:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": CAMERA_RESOLUTION, "format": "RGB888"},
                    controls={"FrameRate" : PROCESSING_FPS}
                )
                self.camera.configure(config)
                self.camera.start()
                logging.info(" PiCamera запущена")
            else:
                self.camera = cv2.VideoCapture(1)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                logging.info(" USB-камера запущена")
        except Exception as e:
            logging.error(f"⚠️ Ошибка запуска камеры: {e}")
            raise

    def release_camera(self):
        if self.camera:
            try:
                if USE_PICAM:
                    self.camera.stop()
                    self.camera.close()
                    logging.info(" PiCamera остановлена и закрыта")
                else:
                    self.camera.release()
                    logging.info("USB-камера остановлена")
            except Exception as e:
                logging.warning(f" Ошибка при остановке камеры: {e}")
            self.camera = None

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

    def send_ssh_message(self, message):
        """Отправка сообщения в stdout (будет видно в SSH-сессии)"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        sys.stdout.flush()
    
    def handle_mode_switch(self, new_mode):
        mode_names = {MODE_OFF: "OFF", MODE_LETTERS: "LETTERS", MODE_ARUCO: "ARUCO"}
        new_mode_str = mode_names.get(new_mode, "UNKNOWN")
    
        if new_mode_str != self.current_mode_str:
            self.send_ssh_message(f" Переключение режима: {self.current_mode_str} → {new_mode_str}")
            self.current_mode_str = new_mode_str

    def pixels_to_meters(self, pixel_x, pixel_y, ppm):
        """Конвертирует смещение в пикселях в метры"""
        return pixel_x / ppm, pixel_y / ppm

    def add_meters_to_coords(self, lat, lon, dx_m, dy_m, heading_deg=0):
        """
        Добавляет смещение в метрах к координатам с учетом курса
        """
        heading_rad = math.radians(heading_deg)
        north_m = dy_m * math.cos(heading_rad) + dx_m * math.sin(heading_rad)
        east_m = dx_m * math.cos(heading_rad) - dy_m * math.sin(heading_rad)
        
        METERS_PER_DEGREE_LAT = 111134.861111
        METERS_PER_DEGREE_LON_AT_EQUATOR = 111321.377778
        
        dlat = north_m / METERS_PER_DEGREE_LAT
        lat_rad = math.radians(lat)
        meters_per_degree_lon = METERS_PER_DEGREE_LON_AT_EQUATOR * math.cos(lat_rad)
        dlon = east_m / meters_per_degree_lon
        
        return lat + dlat, lon + dlon

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
        
    def process_frame(self, frame):
        """Основная функция обработки кадра для распознавания букв"""
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10000:  # Фильтрация мелких контуров
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                center_x, center_y = int(rect[0][0]), int(rect[0][1])
                
                # Вычисляем смещение относительно центра кадра
                frame_center_x, frame_center_y = frame.shape[1]//2, frame.shape[0]//2
                dx_px = center_x - frame_center_x
                dy_px = center_y - frame_center_y
                
                # Преобразуем в метры и географические координаты
                dx_m, dy_m = self.pixels_to_meters(dx_px, dy_px, PIXELS_PER_METER)
                new_lat, new_lon = self.add_meters_to_coords(home_lat, home_lon, dx_m, dy_m)
                
                results.append({
                    'center_px': (center_x, center_y),
                    'center_m': (dx_m, dy_m),
                    'coords': (new_lat, new_lon),
                    'box': box
                })
        
        return results, thresh_color

    def process_aruco(self):
        from cv2 import aruco
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        self.send_ssh_message(" Поиск ArUco маркера (USB-камера)...")

        while self.current_mode == MODE_ARUCO:
            new_mode = self.check_rc_mode()
            if new_mode != self.current_mode:
                self.send_ssh_message(" Выход из режима ARUCO")
                self.current_mode = new_mode
                return

            frame = self.get_frame()
            if frame is None:
                time.sleep(LOOP_DELAY)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)

            if time.time() - last_save_time > 10:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(ARUCO_DIR, f"aruco_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                print(f"Сохранено изображение: {filename}")
                self.send_ssh_message(f" Сохранено изображение: {filename}")
                last_save_time = time.time()   

            if ids is not None and MARKER_ID in ids.flatten():
                
                idx = list(ids.flatten()).index(MARKER_ID)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[idx]], MARKER_LENGTH, camera_matrix, dist_coeffs)
                
                offset_north, offset_east, altitude = DroneController.calculate_offset(rvecs[0], tvecs[0], drone_yaw_deg)

                self.send_vision_offset(offset_east, offset_north, altitude)

                if time.time() - last_save_time > 1: 
                    self.send_ssh_message(f" Offset: N = {offset_north:.2f} m, E = {offset_east:.2f} m, Alt = {altitude:.2f} m") 
                    print(f" Offset: N = {offset_north:.2f} m, E = {offset_east:.2f} m, Alt = {altitude:.2f} m")
                    cv2.aruco.drawDetectedMarkers(frame, [corners[idx]])
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.1)
            time.sleep(LOOP_DELAY)


    def process_letters(self):
        """Реализация распознавания армянских букв из оригинального скрипта"""
        self.send_ssh_message("=== Активирован режим распознавания букв ===")
        self.send_ssh_message(f"Используется порог уверенности: {CONFIDENCE_THRESHOLD}")
        self.send_ssh_message(f"Частота обработки: {PROCESSING_FPS} FPS")
        
        # Инициализация временных меток
        self.last_processing_time = time.time()
        self.last_save_time = time.time()
        
        try:
            while True:

                # Ограничение частоты обработки
                current_time = time.time()
                if current_time - self.last_processing_time < self.processing_interval:
                    time.sleep(0.01)
                    continue
                
                self.last_processing_time = current_time
                
                # Захват кадра
                frame = self.get_frame()
                if frame is None:
                    time.sleep(LOOP_DELAY)
                    continue
                
                # Обработка кадра
                results, thresh_color = self.process_frame(frame)
                processed_letter = None
                
                for result in results:
                    # Отрисовка результатов
                    cv2.drawContours(frame, [result['box']], 0, (0, 255, 0), 2)
                    cv2.circle(frame, result['center_px'], 5, (0, 0, 255), -1)
                    
                    # Распознавание буквы
                    width, height = map(int, cv2.minAreaRect(result['box'])[1])
                    if width > 0 and height > 0:
                        try:
                            letter_crop = cv2.warpPerspective(
                                thresh_color, 
                                cv2.getPerspectiveTransform(
                                    result['box'].astype("float32"),
                                    np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")
                                ),
                                (width, height)
                            )
                            
                            if np.mean(letter_crop) < 250:
                                processed_letter = letter_crop.copy()
                                img_tensor = self.transform(Image.fromarray(letter_crop)).unsqueeze(0)
                                
                                with torch.no_grad():
                                    output = self.model(img_tensor)
                                    conf, pred = torch.max(torch.nn.functional.softmax(output, dim=1), 1)
                                    
                                    if conf.item() > CONFIDENCE_THRESHOLD:
                                        label = self.labels[pred.item()]
                                        text = f"{label} ({conf.item():.2f})"
                                        cv2.putText(frame, text, result['center_px'], 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                        
                                        # Логирование результата
                                        self.send_ssh_message(f"Распознано: {text} | Координаты: {result['coords']}")
                        except Exception as e:
                            logging.error(f"Ошибка обработки буквы: {e}")
                
                # Сохранение кадров

                if current_time - self.last_save_time > 10:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(LETTERS_DIR, f"armenian_letter_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    self.send_ssh_message(f" Сохранено: {filename}")
                    self.last_save_time = current_time
                
                time.sleep(LOOP_DELAY)
                
        except Exception as e:
            logging.error(f"Критическая ошибка в режиме LETTERS: {e}")
            self.send_ssh_message(f"ОШИБКА: {str(e)}")

    def check_mode(self):
        wp = self.get_current_waypoint()
        rc_val = self.get_rc_value()        
        new_mode = None
                
        # Приоритет ручного управления (RC)
        if rc_val is not None and rc_val != self.last_rc_value:
            new_mode = self.rc_to_mode(rc_val)
            self.last_rc_value = rc_val
            self.logger.info(f"RC: {rc_val} → {self.mode_name(new_mode)}")
                
        # Проверка точки миссии
        elif wp is not None and wp != self.last_wp_value and wp in self.wp_modes:
            new_mode = self.wp_modes[wp]
            self.last_wp_value = wp
            self.logger.info(f"WP: {wp} → {self.mode_name(new_mode)}")
                
        if new_mode is not None and new_mode != self.current_mode:
            self.current_mode = new_mode
            self.handle_mode_switch(new_mode)
        
        return self.current_mode

    def mode_name(self, mode):
        return ["OFF", "LETTERS", "ARUCO"][mode]

    def run(self):
        self.send_ssh_message("--- Запуск контроллера ---")
        try:
            while True:
                self.current_mode = self.check_mode()
                
                self.release_camera()
                if self.current_mode == MODE_OFF:
                    self.process_LETTERS = None
                    self.process_ARUCO = None
                    self.no_process = True
                    time.sleep(LOOP_DELAY)
                    
                elif self.current_mode == MODE_LETTERS and not self.process_LETTERS:
                    self.init_camera() 
                    self.process_letters()
                    self.process_LETTERS = True
                    self.process_ARUCO = None
                    self.no_process = None

                elif self.current_mode == MODE_ARUCO and not self.process_ARUCO:
                    self.init_camera()
                    self.process_aruco()
                    self.process_ARUCO = True
                    self.process_LETTERS = None
                    self.no_process = None

                time.sleep(LOOP_DELAY)
                
        except KeyboardInterrupt:
            self.send_ssh_message("--- Завершение работы ---")
            self.release_camera()

if __name__ == '__main__':
    controller = DroneController()
    controller.run()
