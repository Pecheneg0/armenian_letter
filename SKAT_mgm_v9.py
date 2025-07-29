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
from modelold import ArmenianLetterNet
from collections import defaultdict
from cv2 import aruco
# ===== КОНФИГУРАЦИЯ =====
MAVLINK_PORT = '/dev/ttyAMA0'
MAVLINK_BAUD = 57600
RC_CHANNEL = 6

MODE_OFF = 0
MODE_LETTERS = 1
MODE_ARUCO = 2
MODE_MGM = 3

CAMERA_RESOLUTION_PICAM = (720, 720)  
CAMERA_RESOLUTION_USBCAM = (720, 720)

MARKER_42 = 42  # ID искомой метки
MARKER_41 = 41
LENGTH_41 = 0.08 # Длина стороны маркера в метрах
LENGTH_42 = 0.32
CAMERA_ID = 0

DROP_SERVO_PWM_THRESHOLD = 1700

MODEL_PATH = "ALM_best.pth"
LABELS_PATH = "labels.txt"


TELEMETRY_INTERVAL = 2
LOOP_DELAY = 0.02

CONFIDENCE_THRESHOLD = 0.85   # Порог уверенности распознавания
PROCESSING_FPS = 60           # Ограничение частоты обработки (кадров в секунду)
PIXELS_PER_METER = 100  

ARUCO_DIR = "/home/pi/Desktop/aruco_images"
LETTERS_DIR = "/home/pi/Desktop/letters_images"
LOG_DIR = "/home/pi/Desktop"
MGM_DIR = "/home/pi/Desktop/mgm_images" 
MARKER_LENGTH = 0.1 
 

# Параметры для детекции МГМ
MODE_MGM = 3  # Новый режим для детекции МГМ
LOWER_BLUE = np.array([83, 119, 40])
UPPER_BLUE = np.array([129, 245, 255])
MIN_CONTOUR_AREA = 500
DETECTION_WINDOW_SEC = 7
CONSECUTIVE_DETECTIONS_TO_CONFIRM = 10
DEBUG_MODE_MGM = False
DROP_SERVO_ID = 8


frame_w, frame_h = 720, 720
camera_matrix = np.array([
    [1000, 0, frame_w / 2],
    [0, 1000, frame_h / 2],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARUCO_DIR, exist_ok=True)
os.makedirs(LETTERS_DIR, exist_ok=True)
os.makedirs(MGM_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "sla_controller.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("DroneController")


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
        self.process_ARUCO = False
        self.process_LETTERS = False
        self.process_MGM = False
        self.process_OFF = False
        self.USE_PICAM = False
        self.USE_USB_CAM = False
        self.no_process = False
        self.transform = None
        self.min_altitude = 2
        self.pos_tolerance = 0.15
        self.land_command_sent = False
        self.drone_in_guided = False
        self.guided_mode = False
        self.reached_200m = False

        self.last_wp_time = 0
        self.last_rc_time = 0
        self.wp_interval = 3
        self.rc_interval = 0.7

        self.letter_stats = defaultdict(lambda: {'confidences': [], 'coords': [], 'timestamps': []})
        self.processing_active = False
        self.last_detection_time = 0
        self.processing_sessions = []
        self.LETTER_TIMEOUT = 2.0
        self.MIN_DETECTIONS = 3
        self.SAVE_INTERVAL = 0.2
        self.last_save_time = 0

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        # Для обработки букв
        self.last_processing_time = 0
        self.last_save_time = 0
        self.processing_interval = 1.0 / PROCESSING_FPS
        self.is_pixhawk_connected = False
        self.model_loaded = False
        if not self.is_pixhawk_connected:
            self.connect_to_pixhawk()
            self.is_pixhawk_connected = True
        
        self.logger = logging.getLogger("modeswitcher")
        
        # Соответствие точек миссии режимам
        self.wp_modes = {12: MODE_LETTERS, 17: MODE_ARUCO, 0: MODE_OFF, 20: MODE_MGM}
        
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
                break
            except Exception as e:
                logging.error(f"Ошибка подключения к Pixhawk: {e}. Повтор через 5 с...")
                time.sleep(5)

    def get_current_waypoint(self):
        current_time = time.time()
        if current_time - self.last_wp_time < self.wp_interval:
            return None
        self.last_wp_time = current_time
        msg = self.master.recv_match(type='MISSION_CURRENT', blocking=False)
        return msg.seq if msg else None

    def get_rc_value(self):
        current_time = time.time()
        if current_time - self.last_rc_time < self.rc_interval:
            return None
        self.last_rc_time = current_time 
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
        
        sys.stdout.flush()

    def load_model(self):
        try:
            self.model = ArmenianLetterNet()
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            self.model.eval()
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f.readlines()]
            
        except Exception as e:
            logging.error(f" Ошибка загрузки модели: {e}")
            raise

    def init_camera(self):
            
        try:
            if self.USE_PICAM:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": CAMERA_RESOLUTION_PICAM, "format": "RGB888"},
                    controls={
                        "FrameRate": PROCESSING_FPS,
                        "AwbEnable": True,  # Автобаланс белого
                        "AeEnable": True,   # Автоэкспозиция
                    }
                )
                self.camera.configure(config)
                self.camera.start()
                self.camera.set_controls({"Sharpness": 1.0})
                                    
            elif self.USE_USB_CAM:
                self.camera = cv2.VideoCapture(CAMERA_ID)
                if not self.camera.isOpened():
                    logging.error("Ошибка открытия камеры USB")
                    self.camera = None
                    return
                    
                # Пробуем установить параметры
                try:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION_USBCAM[0])
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION_USBCAM[1])
                    self.camera.set(cv2.CAP_PROP_FPS, PROCESSING_FPS)
                except:
                    logging.info("USB камера запущена без дополнительных настройек")
                    
        except Exception as e:
            logging.error(f"Ошибка запуска камеры: {e}")
            #self.send_ssh_message(f"Ошибка запуска камеры: {e}") 
            self.camera = None
    

    def process_mgm(self):
        while True:
            msg =self. master.recv_match(type='SERVO_OUTPUT_RAW', blocking=True)
            if not msg: continue

            servo_pwm = getattr(msg, f'servo{DROP_SERVO_ID+1}_raw', 0)
            if servo_pwm > DROP_SERVO_PWM_THRESHOLD:
                logger.info(f"Получен сигнал сброса! (Серво {DROP_SERVO_ID+1} PWM = {servo_pwm})")
                logger.info("Возврат в режим ожидания...")
                time.sleep(5)
                        
                """Режим детекции МГМ"""
                self.send_ssh_message("=== Активирован режим детекции МГМ ===")
                
                # Инициализация камеры для MGM
                self.USE_PICAM = False
                self.USE_USB_CAM = True
                self.init_camera()
                
                if not (self.USE_USB_CAM and self.camera and self.camera.isOpened()):
                    #self.send_ssh_message("Ошибка инициализации камеры для MGM")
                    return
                
                start_time = time.time()
                consecutive_hits = 0
                max_hits_achieved = 0
                detection_confirmed = False
                last_save_time = time.time()
                #self.send_ssh_message(f"===== АКТИВАЦИЯ ОКНА ДЕТЕКЦИИ ({DETECTION_WINDOW_SEC} сек) =====")
                
                try:
                    while time.time() - start_time < DETECTION_WINDOW_SEC:
                        # Проверяем, не сменился ли режим
                        if self.check_mode() != MODE_MGM:
                            #self.send_ssh_message("Прерывание детекции MGM: сменился режим")
                            return
                        
                        frame = self.get_frame()
                        if frame is None:
                            time.sleep(0.1)
                            continue
                        
                        # Обработка изображения для детекции синего объекта
                        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
                        mask = cv2.erode(mask, None, iterations=2)
                        mask = cv2.dilate(mask, None, iterations=2)
                        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        found = False
                        if len(contours) > 0:
                            main_contour = max(contours, key=cv2.contourArea)
                            if cv2.contourArea(main_contour) > MIN_CONTOUR_AREA:
                                found = True
                        
                        if found:
                            consecutive_hits += 1
                        
                        if consecutive_hits > max_hits_achieved:
                                max_hits_achieved = consecutive_hits
                        else:
                            consecutive_hits = 0
                        
                        # Условие подтверждения детекции
                        if consecutive_hits >= CONSECUTIVE_DETECTIONS_TO_CONFIRM:
                            detection_confirmed = True
                            #self.send_ssh_message(f"!!! МГМ ПОДТВЕРЖДЕН ({consecutive_hits} последовательных кадров) !!!")
                            break

                        # Отладочная информация
                        if DEBUG_MODE_MGM:

                            cv2.putText(frame, f"Hits: {consecutive_hits}/{CONSECUTIVE_DETECTIONS_TO_CONFIRM}", 
                                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                            cv2.imshow("MGM Detection", frame)

                            cv2.waitKey(1)
                        
                        time.sleep(0.05)
                    
                    if detection_confirmed:
                        logger.info(f"!!! МГМ ПОДТВЕРЖДЕН ({max_hits_achieved} последовательных кадров) !!!")
                        #logger.info("===== ЦИКЛ ДЕТЕКЦИИ ЗАВЕРШЕН (УСПЕХ) =====")
                        if time.time() - last_save_time > 10:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                            filename = os.path.join(MGM_DIR, f"mgm_{timestamp}.jpg")
                            cv2.imwrite(filename, frame)
                            #print(f"Сохранено изображение: {filename}")
                            #self.send_ssh_message(f" Сохранено изображение: {filename}")
                            last_save_time = time.time()

                    #else:
                        #self.send_ssh_message(f"МГМ не подтвержден. Макс. последовательных кадров: {max_hits_achieved}")
                    
                    #self.send_ssh_message("===== ЦИКЛ ДЕТЕКЦИИ МГМ ЗАВЕРШЕН =====")
                    time.sleep(5)
                    
                finally:
                    # Всегда освобождаем ресурсы
                    #self.release_camera()
                    if DEBUG_MODE_MGM:
                        cv2.destroyAllWindows()


    def get_current_altitude(self):
        msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
        if not self.reached_200m :
            altitude = msg.relative_alt / 1e3 
            if altitude is not None and altitude >= 200:
                logging.info(f"Отметка в 200 метров достигнута, высота : {altitude}")
                #self.send_ssh_message(f"Отметка в 200 метров достигнута, высота : {altitude}")
                self.reached_200m = True
        return msg.relative_alt / 1e3 if msg else 0.0

    def move_to_offset(self, dx, dy, target_alt):
        
        self.master.mav.set_position_target_local_ned_send(
            int(time.time() * 1000),
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,
            int(0b100111111000),
            dx, dy, target_alt,
            0, 0, 0, 0, 0, 0, 0, 0)


    def release_camera(self):
        if self.camera:
            try:
                if self.USE_PICAM:
                    self.camera.stop()
                    self.camera.close()
                    self.camera = None
                    logging.info("PiCamera остановлена")
                    #self.send_ssh_message("PiCamera остановлена") 
                elif self.USE_USB_CAM:
                    self.camera.release()
                    self.camera = None
                    logging.info("USB-камера остановлена")
                    #self.send_ssh_message("USB-камера остановлена")
            except Exception as e:
                logging.warning(f"Ошибка при остановке камеры: {e}")
                #self.send_ssh_message(f"Ошибка при остановке камеры: {e}") 
    
    def calculate_offset(self, tvec):
        x_cam = -tvec[0][1]
        y_cam = tvec[0][0]
        yaw_deg = self.get_yaw()
        yaw_rad = np.radians(yaw_deg)
        R = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad),  np.cos(yaw_rad)]
        ])
        offset = R @ np.array([x_cam, y_cam])
        return offset[0], offset[1]  # North, East
    
    def handle_mode_switch(self, new_mode):
        mode_names = {MODE_OFF: "OFF", MODE_LETTERS: "LETTERS", MODE_ARUCO: "ARUCO", MODE_MGM: "MGM"}
        new_mode_str = mode_names.get(new_mode, "UNKNOWN")
    
        if new_mode_str != self.current_mode_str:
            #self.send_ssh_message(f" Переключение режима: {self.current_mode_str} -> {new_mode_str}")
            self.current_mode_str = new_mode_str

    def pixels_to_meters(self, pixel_x, pixel_y, ppm):
        return pixel_x / ppm, pixel_y / ppm

    def add_meters_to_coords(self, lat, lon, dx_m, dy_m, yaw_deg=0):
        yaw_rad = math.radians(yaw_deg)
        north_m = dy_m * math.cos(yaw_rad) + dx_m * math.sin(yaw_rad)
        east_m = dx_m * math.cos(yaw_rad) - dy_m * math.sin(yaw_rad)
        
        METERS_PER_DEGREE_LAT = 111134.861111
        METERS_PER_DEGREE_LON_AT_EQUATOR = 111321.377778
        
        dlat = north_m / METERS_PER_DEGREE_LAT
        lat_rad = math.radians(lat)
        meters_per_degree_lon = METERS_PER_DEGREE_LON_AT_EQUATOR * math.cos(lat_rad)
        dlon = east_m / meters_per_degree_lon
        
        return lat + dlat, lon + dlon

    def get_frame(self):
        try:
            if self.USE_PICAM and self.camera:
                return self.camera.capture_array()
                
            if self.USE_USB_CAM and self.camera:  
                ret, frame = self.camera.read()
                return frame if ret else None
        except Exception as e:
            logging.error(f"Ошибка захвата кадра: {e}")
            return None

    def execute_landing(self, offset_north, offset_east, altitude): 
        
        last_command_time = 0
        command_interval = 1  # Интервал между командами (сек)

        try:
        
            while not self.land_command_sent:
                #altitude = self.get_current_altitude()
                current_time = time.time()
                if current_time - last_command_time < command_interval:
                    time.sleep(0.01)
                    continue


                if altitude > self.min_altitude:
                    if abs(offset_north) < self.pos_tolerance and abs(offset_east) < self.pos_tolerance:
                        self.move_to_offset(0, 0, 0.6)

                    else:
                        self.move_to_offset(offset_north, offset_east, altitude)
                    time.sleep(LOOP_DELAY)
                else:
                    if abs(offset_north) < self.pos_tolerance and abs(offset_east) < self.pos_tolerance:
                        self.master.set_mode("QLAND")
                        self.land_command_sent = True
                    else:
                        self.move_to_offset(offset_north, offset_east, 0)
            time.sleep(0.02)

        except Exception as e:
            logging.error(f"Критическая ошибка: {str(e)}")
        
    def get_yaw(self):
        try:
            msg = self.master.recv_match(type='ATTITUDE', blocking=True)
            if msg:
                yaw = math.degrees(msg.yaw)
                logging.info(f"Курс: {yaw:.1f}°")
                return yaw
        except Exception as e:
            logging.error(f"Ошибка получения курса: {e}")
            return 0.0

    def get_coordinates(self):
        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg:
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            #print(f"Местоположение: Широта: {lat} | Долгота: {lon} ")
            return lat, lon
        return 0.0, 0.0
        
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if  11400 < area: 
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                center_x, center_y = int(rect[0][0]), int(rect[0][1])
                
                frame_center_x, frame_center_y = 720/2, 720/2
                dx_px = center_x - frame_center_x
                dy_px = -center_y + frame_center_y
                current_lat, current_lon = self.get_coordinates() 

                dx_m, dy_m = self.pixels_to_meters(dx_px, dy_px, PIXELS_PER_METER)
                new_lat, new_lon = self.add_meters_to_coords(current_lat, current_lon, dx_m, dy_m)
                
                results.append({
                    'center_px': (center_x, center_y),
                    'center_m': (dx_m, dy_m),
                    'coords': (new_lat, new_lon),
                    'box': box
                })
        
        return results, thresh_color

    def process_aruco(self):

        TARGET_ALTITUDE = 1.5  # Высота переключения маркеров
        current_target_id = MARKER_42  # Начинаем с маркера 42
        current_length = LENGTH_42
        
        #self.send_ssh_message(f"Старт поиска маркера {current_target_id} ")

        logging.info(f"Старт поиска маркера {current_target_id} ")
        
        last_save_time = time.time()
        last_command_time = 0
        command_interval = 0.12  # Интервал между командами (сек)

        while self.current_mode == MODE_ARUCO:
            current_time = time.time()
            if current_time - last_command_time < command_interval:
                time.sleep(0.01)
                continue
                
            last_command_time = current_time

            new_mode = self.check_mode()
            if new_mode != self.current_mode:
                #self.send_ssh_message(" Выход из режима ARUCO")
                self.current_mode = new_mode
                return
                
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.detector.detectMarkers(gray)
            
            if not self.drone_in_guided:
                self.master.set_mode("GUIDED")
                #print("Установлен режим GUIDED")
                logging.info("Установлен режим GUIDED")
                #self.send_ssh_message("Установлен режим GUIDED")
                self.drone_in_guided = True
            
            if ids is not None and current_target_id in ids.flatten():
                idx = list(ids.flatten()).index(current_target_id)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[idx]], current_length, camera_matrix, dist_coeffs
                )
                
                # Получаем реальную высоту (Z-координата)
                altitude = 0.5 * tvecs[0][0][2]
                #tvec = tvecs[0][0]
                #offset_north = -tvec[1] 
                #offset_east = tvec[0]
                offset_north, offset_east = self.calculate_offset(tvecs[0]) 
                
                #print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Маркер {current_target_id} | "
                  #f"Высота: {altitude:.3f}m | "
                  #f"Размер: {current_length}m")
                
                if time.time() - last_save_time > 5:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(ARUCO_DIR, f"aruco_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    #print(f"Сохранено изображение: {filename}")
                    #self.send_ssh_message(f" Сохранено изображение: {filename}")
                    last_save_time = time.time()
                
                # Проверка необходимости переключения маркера
                if current_target_id == MARKER_42 and altitude <= TARGET_ALTITUDE:
                    current_target_id = MARKER_41
                    current_length = LENGTH_41
                    #self.send_ssh_message(f"[{datetime.now().strftime('%H:%M:%S')}] ПЕРЕКЛЮЧЕНИЕ: "
                      #f"Новый целевой маркер {current_target_id} (достигнута высота {altitude:.2f}m ≤ {TARGET_ALTITUDE}m)")
                    continue
    
                # Запуск процедуры посадки
                self.execute_landing(offset_north, offset_east, altitude)

            time.sleep(0.01)

    def process_letters(self):
        if not self.model_loaded :
            self.load_model()
            self.model_loaded = True

        #self.send_ssh_message("=== Активирован режим распознавания букв ===")
        self.last_processing_time = time.time()
        self.last_save_time = time.time()
        
        try:
            while self.current_mode == MODE_LETTERS:
                new_mode = self.check_mode()
                if new_mode != self.current_mode:
                    #self.send_ssh_message(" Выход из режима LETTERS")
                    self.current_mode = new_mode
                    return

                current_time = time.time()
                if current_time - self.last_processing_time < self.processing_interval:
                    time.sleep(0.01)
                    continue
                
                self.last_processing_time = current_time
                

                frame = self.get_frame()
                if frame is None:
                    time.sleep(0.1)
                    #self.send_ssh_message("Кадр не получен")
                    #print("Кадр не получен") 
                    continue
                
                results, thresh_color = self.process_frame(frame)
                letter_detected = False
                
                for result in results:
                    cv2.drawContours(frame, [result['box']], 0, (0, 255, 0), 2)
                    cv2.circle(frame, result['center_px'], 5, (0, 0, 255), -1)
                    
                    width, height = map(int, cv2.minAreaRect(result['box'])[1])
                    if width > 10 and height > 10:
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
                                
                                img_tensor = self.transform(Image.fromarray(letter_crop)).unsqueeze(0)
                                
                                with torch.no_grad():
                                    output = self.model(img_tensor)
                                    conf, pred = torch.max(torch.nn.functional.softmax(output, dim=1), 1)
                                    
                                    if conf.item() > CONFIDENCE_THRESHOLD:
                                        letter_detected = True
                                        self.last_detection_time = current_time
                                        label = self.labels[pred.item()]
                                        self.letter_stats[label]['confidences'].append(conf.item())
                                        self.letter_stats[label]['coords'].append(result['coords'])
                                        self.letter_stats[label]['timestamps'].append(current_time)
                                    
                                        self.processing_active = True
                                        text = f"{label} ({conf.item():.2f})"
                                        cv2.putText(frame, text, result['center_px'], 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                        #self.send_ssh_message(f"Распознано: {text} | Координаты: {result['coords']}")
                                        logging.info(f"Распознано: {text} | Координаты: {result['coords']}")
                        except Exception as e:
                            logging.error(f"Ошибка обработки буквы: {e}")
                            #self.send_ssh_message(f"Ошибка обработки буквы: {e}")
                
                if letter_detected and (current_time - self.last_save_time > self.SAVE_INTERVAL):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(LETTERS_DIR, f"armenian_letter_{timestamp}.jpg")
                    if not os.path.exists(filename):
                        cv2.imwrite(filename, frame)
                        #self.send_ssh_message(f" Сохранено: {filename}")
                        self.last_save_time = current_time
                

                if self.processing_active and (current_time - self.last_detection_time) > self.LETTER_TIMEOUT:
                    if self.letter_stats:
                        best_letter, best_data = max(self.letter_stats.items(),
                                                key=lambda x: len(x[1]['confidences']))
                        
                        if len(best_data['confidences']) >= self.MIN_DETECTIONS:
                            avg_conf = sum(best_data['confidences'])/len(best_data['confidences'])
                            avg_lat = sum(c[0] for c in best_data['coords'])/len(best_data['coords'])
                            avg_lon = sum(c[1] for c in best_data['coords'])/len(best_data['coords'])
                            last_time = datetime.fromtimestamp(max(best_data['timestamps'])).strftime('%H:%M:%S')
                            
                            log_msg = "\n=== Результаты обработки ==="
                            log_msg += f"\nНаиболее вероятная буква: {best_letter}"
                            log_msg += f"\nКоличество обнаружений: {len(best_data['confidences'])}"
                            log_msg += f"\nСредняя уверенность: {avg_conf:.2f}"
                            log_msg += f"\nСредние координаты: ({avg_lat:.6f}, {avg_lon:.6f})"
                            log_msg += f"\nВремя фиксации: {last_time}"
                            log_msg += "\n==========================="
                            
                            #self.send_ssh_message(log_msg)
                            logging.info(log_msg)
                            
                            self.processing_sessions.append({
                                'letter': best_letter,
                                'count': len(best_data['confidences']),
                                'avg_confidence': avg_conf,
                                'avg_coords': (avg_lat, avg_lon),
                                'timestamp': last_time
                            })
                    
                    self.letter_stats = defaultdict(lambda: {'confidences': [], 'coords': [], 'timestamps': []})
                    self.processing_active = False
                
                time.sleep(LOOP_DELAY)
                
        except Exception as e:
            logging.error(f"Критическая ошибка в режиме LETTERS: {e}")
            #self.send_ssh_message(f"ОШИБКА: {str(e)}")
        
        # Финализация последнего сеанса при выходе
        if self.processing_active and self.letter_stats:
            best_letter, best_data = max(self.letter_stats.items(),
                                    key=lambda x: len(x[1]['confidences']))
            
            if len(best_data['confidences']) >= self.MIN_DETECTIONS:
                avg_conf = sum(best_data['confidences'])/len(best_data['confidences'])
                avg_lat = sum(c[0] for c in best_data['coords'])/len(best_data['coords'])
                avg_lon = sum(c[1] for c in best_data['coords'])/len(best_data['coords'])
                last_time = datetime.fromtimestamp(max(best_data['timestamps'])).strftime('%H:%M:%S')
                
                log_msg = "\n=== Финальные результаты обработки ==="
                log_msg += f"\nНаиболее вероятная буква: {best_letter}"
                log_msg += f"\nКоличество обнаружений: {len(best_data['confidences'])}"
                log_msg += f"\nСредняя уверенность: {avg_conf:.2f}"
                log_msg += f"\nСредние координаты: ({avg_lat:.6f}, {avg_lon:.6f})"
                log_msg += f"\nВремя фиксации: {last_time}"
                log_msg += "\n====================================="
                
                #self.send_ssh_message(log_msg)
                logging.info(log_msg)

                self.processing_sessions.append({
                    'letter': best_letter,
                    'count': len(best_data['confidences']),
                    'avg_confidence': avg_conf,
                    'avg_coords': (avg_lat, avg_lon),
                    'timestamp': last_time
                })
        
        # Вывод сводки всех сеансов
        if self.processing_sessions:
            log_msg = "\n=== Сводка всех сеансов обработки ==="
            for i, session in enumerate(self.processing_sessions, 1):
                log_msg += f"\n{i}. Буква: {session['letter']}"
                log_msg += f"\n   Обнаружений: {session['count']}"
                log_msg += f"\n   Уверенность: {session['avg_confidence']:.2f}"
                log_msg += f"\n   Координаты: {session['avg_coords']}"
                log_msg += f"\n   Время: {session['timestamp']}"
                log_msg += "\n-----------------------------"
            
            #self.send_ssh_message(log_msg)
            logging.info(log_msg)


    def check_mode(self):
        wp = self.get_current_waypoint()
        rc_val = self.get_rc_value()        
        new_mode = None
                
        if rc_val is not None and rc_val != self.last_rc_value:
            new_mode = self.rc_to_mode(rc_val)
            self.last_rc_value = rc_val
            logging.info(f"RC: {rc_val} -> {self.mode_name(new_mode)}")
            #print(f"RC: {rc_val} -> {self.mode_name(new_mode)}")
            #self.send_ssh_message(f"RC: {rc_val} -> {self.mode_name(new_mode)}") 
                
        elif wp is not None and wp != self.last_wp_value and wp in self.wp_modes:
            new_mode = self.wp_modes[wp]
            self.last_wp_value = wp
            logging.info(f"WP: {wp} -> {self.mode_name(new_mode)}")
            #print(f"WP: {wp} -> {self.mode_name(new_mode)}")
            #self.send_ssh_message(f"WP: {wp} -> {self.mode_name(new_mode)}") 
                
        if new_mode is not None and new_mode != self.current_mode:
            self.current_mode = new_mode
            self.handle_mode_switch(new_mode)
        
        return self.current_mode

    def mode_name(self, mode):
        return ["OFF", "LETTERS", "ARUCO", "MGM"][mode]

    def run(self):
        #self.send_ssh_message("--- Запуск компьютера-компаньона ---")
        try:
            while True:
                try:  
                    self.current_mode = self.check_mode()
                    
                    # Всегда освобождаем камеру при переходе в OFF
                    if self.current_mode == MODE_OFF and not self.process_OFF:
                        self.release_camera()
                        self.process_LETTERS = False
                        self.process_ARUCO = False
                        self.process_MGM = False
                        self.process_OFF = True
                        time.sleep(0.1)
                        continue
                        
                    elif self.current_mode == MODE_LETTERS and not self.process_LETTERS:
                        self.USE_PICAM = True
                        self.USE_USB_CAM = False
                        self.init_camera()
                        
                        # Проверяем успешность инициализации
                        if self.camera:
                            self.process_letters()
                            self.process_LETTERS = True
                            self.process_ARUCO = False
                            self.process_MGM = False
                            self.process_OFF = False
                        else:
                            #self.send_ssh_message("Ошибка инициализации камеры для LETTERS")
                            self.current_mode = MODE_OFF

                    elif self.current_mode == MODE_ARUCO and not self.process_ARUCO:
                        self.USE_PICAM = False
                        self.USE_USB_CAM = True     
                        self.init_camera()
                        # Проверяем успешность инициализации
                        if self.camera and (self.USE_USB_CAM and self.camera.isOpened()):
                            self.process_aruco()
                            self.process_ARUCO = True
                            self.process_LETTERS = False
                            self.process_MGM = False
                            self.process_OFF = False
                        else:
                            #self.send_ssh_message("Ошибка инициализации камеры для ARUCO")
                            self.current_mode = MODE_OFF
                    
                    elif self.current_mode == MODE_MGM and not self.process_MGM:
                        self.USE_PICAM = False
                        self.USE_USB_CAM = True  
                        self.process_mgm()
                        self.process_ARUCO = False
                        self.process_LETTERS = False                        
                        self.process_MGM = True
                        self.process_OFF = False
                        # После завершения MGM возвращаемся в режим OFF
                        self.current_mode = MODE_OFF
                        self.handle_mode_switch(MODE_OFF)                       

                    time.sleep(0.01)
                    
                except Exception as inner_e:
                    logging.error(f"Ошибка в основном цикле: {inner_e}")
                    #self.send_ssh_message(f"Внутренняя ошибка: {inner_e}")
                    time.sleep(1)
                
        except KeyboardInterrupt:
            #self.send_ssh_message("--- Завершение работы ---")
            self.release_camera()
        except Exception as e:
            #self.send_ssh_message(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
            logging.critical(f"Необработанное исключение: {e}")

if __name__ == '__main__':
    controller = DroneController()
    controller.run()
