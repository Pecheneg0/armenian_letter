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

# ===== КОНФИГУРАЦИЯ =====
MAVLINK_PORT = '/dev/ttyAMA0'
MAVLINK_BAUD = 57600
RC_CHANNEL = 6

MODE_OFF = 0
MODE_LETTERS = 1
MODE_ARUCO = 2

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
 

frame_w, frame_h = 1920, 1080
camera_matrix = np.array([
    [1000, 0, frame_w / 2],
    [0, 1000, frame_h / 2],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARUCO_DIR, exist_ok=True)
os.makedirs(LETTERS_DIR, exist_ok=True)


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
        self.USE_PICAM = False
        self.USE_USB_CAM = False
        self.no_process = False
        self.transform = None
        self.min_altitude = 2
        self.pos_tolerance = 0.15
        self.yaw_tolerance = 5.0
        self.land_command_sent = False
        self.drone_in_vtol = False
        # Для обработки букв
        self.last_processing_time = 0
        self.last_save_time = 0
        self.processing_interval = 1.0 / PROCESSING_FPS

        self.connect_to_pixhawk()
        self.load_model()
        self.logger = logging.getLogger("modeswitcher")
        
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
                logging.info(f"Подключено к системе {self.master.target_system}")
                self.send_ssh_message(f"Подключено к системе {self.master.target_system}")

                break
            except Exception as e:
                logging.error(f"Ошибка подключения к Pixhawk: {e}. Повтор через 5 сек...")
                self.send_ssh_message(f"Ошибка подключения к Pixhawk: {e}. Повтор через 5 сек...") 

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
        # Если камера уже инициализирована - пропускаем
        if (self.USE_PICAM and hasattr(self, 'camera') and self.camera) or \
           (self.USE_USB_CAM and hasattr(self, 'camera') and self.camera ):
            return
            
        try:
            if self.USE_PICAM:
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                    main={"size": CAMERA_RESOLUTION, "format": "RGB888"},
                    controls={
                        "FrameRate": PROCESSING_FPS,
                        "AwbEnable": True,  # Автобаланс белого
                        "AeEnable": True,   # Автоэкспозиция
                    }
                )
                self.camera.configure(config)
                self.camera.start()
                
                # Пробуем установить резкость
                try:
                    self.camera.set_controls({"Sharpness": 1.0})
                    logging.info("PiCamera запущена с настройкой резкости")
                    self.send_ssh_message("PiCamera запущена с настройкой резкости") 
                except:
                    logging.info("PiCamera запущена (без настройки резкости)")
                    self.send_ssh_message("PiCamera запущена (без настройки резкости)")
                    
            elif self.USE_USB_CAM:
                self.camera = cv2.VideoCapture(CAMERA_ID)
                if not self.camera.isOpened():
                    logging.error("Не удалось открыть USB-камеру")
                    self.send_ssh_message("Не удалось открыть USB-камеру")
                    self.camera = None
                    return
                    
                # Пробуем установить параметры
                try:
                    self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
                    self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
                    self.camera.set(cv2.CAP_PROP_FPS, PROCESSING_FPS)
                    logging.info("USB-камера запущена с настройками")
                except:
                    logging.info("USB-камера запущена (без дополнительных настроек)")
                    
        except Exception as e:
            logging.error(f"Ошибка запуска камеры: {e}")
            self.send_ssh_message(f"Ошибка запуска камеры: {e}") 
            self.camera = None
    
    def get_current_altitude(self):
        msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=1.0)
        return msg.relative_alt / 1000 if msg else 0.0

    def move_to_offset(self, dx, dy, target_alt=None):
        current_alt = self.get_current_altitude()

        if target_alt is None:
            target_alt = current_alt
    
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
                    self.send_ssh_message("PiCamera остановлена") 
                elif self.USE_USB_CAM:
                    self.camera.release()
                    self.camera = None
                    logging.info("USB-камера остановлена")
                    self.send_ssh_message("USB-камера остановлена")
            except Exception as e:
                logging.warning(f"Ошибка при остановке камеры: {e}")
                self.send_ssh_message(f"Ошибка при остановке камеры: {e}") 
    
    def calculate_offset(self, tvec):
        x_cam = tvec[0][0]
        z_cam = tvec[0][1]
        yaw_deg = self.get_yaw()
        yaw_rad = np.radians(yaw_deg)
        R = np.array([
            [np.cos(yaw_rad), -np.sin(yaw_rad)],
            [np.sin(yaw_rad),  np.cos(yaw_rad)]
        ])
        offset = R @ np.array([x_cam, z_cam])
        return offset[1], offset[0]  # North, East
    
    def handle_mode_switch(self, new_mode):
        mode_names = {MODE_OFF: "OFF", MODE_LETTERS: "LETTERS", MODE_ARUCO: "ARUCO"}
        new_mode_str = mode_names.get(new_mode, "UNKNOWN")
    
        if new_mode_str != self.current_mode_str:
            self.send_ssh_message(f" Переключение режима: {self.current_mode_str} → {new_mode_str}")
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
                
            if self.USE_USB_CAM and self.camera:  # Упрощенная проверка
                ret, frame = self.camera.read()
                return frame if ret else None
        except Exception as e:
            logging.error(f"Ошибка захвата кадра: {e}")

        return None

    def execute_landing(self, offset_north, offset_east): 
        print("\n=== Начало процедуры посадки ===")
        self.send_ssh_message("\n=== Начало процедуры посадки ===")

        try:
            while not self.land_command_sent:
                altitude = self.get_current_altitude()
                print(f"Текущая высота: {altitude:.2f} m")
                self.send_ssh_message(f"Текущая высота: {altitude:.2f} m") 

                if altitude > self.min_altitude:
                    if abs(offset_north) < self.pos_tolerance and abs(offset_east) < self.pos_tolerance:
                        print("Малые смещения -> Снижение")
                        self.send_ssh_message("Малые смещения -> Снижение") 
                        if altitude > 10:
                            self.move_to_offset(0, 0, 5)
                        elif 2 < altitude <= 10:
                            self.move_to_offset(0, 0, 1)
                    else:
                        print(f"Коррекция: Север={offset_north:.2f}m | Восток={offset_east:.2f}m")
                        self.send_ssh_message(f"Коррекция: Север={offset_north:.2f}m | Восток={offset_east:.2f}m") 
                        self.move_to_offset(offset_north, offset_east, altitude)
                    time.sleep(2)
                else:
                    if abs(offset_north) < self.pos_tolerance and abs(offset_east) < self.pos_tolerance:
                        self.master.set_mode("QLAND")
                        self.land_command_sent = True
                        print("Команда посадки отправлена, режим QLAND")
                        self.send_ssh_message("Команда посадки отправлена, режим QLAND")
                    else:
                        self.move_to_offset(offset_north, offset_east, 0)
        except Exception as e:
            print(f"Критическая ошибка: {str(e)}")
        finally:
            print("Посадка завершена")
        
    def get_yaw(self):
        try:
            msg = self.master.recv_match(type='ATTITUDE', blocking=True, timeout=1.0)
            if msg:
                yaw = math.degrees(msg.yaw)
                logging.info(f"Курс: {yaw:.1f}°")
                return yaw
        except Exception as e:
            logging.error(f"Ошибка получения курса: {e}")
        return 0.0

    def get_coordinates(self):
        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1.0)
        if msg:
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.alt / 1e3
            print(f"Местоположение: Широта: {lat} | Долгота: {lon} | Высота: {alt}")
            return lat, lon, alt
        return 0.0, 0.0, 0.0

    def switch_to_vtol_mode(self):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
            0, 3, 0, 0, 0, 0, 0, 0  
        )
        print("Запрос перехода в режим VTOL отправлен")
        self.send_ssh_message("Запрос перехода в режим VTOL отправлен")
        self.drone_in_vtol = True

        # Ожидаем подтверждения команды
        ack = self.master.recv_match(
            type='COMMAND_ACK',
            command=mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
            blocking=True,
            timeout=3
        )
        
        if not ack:
            print("Ошибка: Нет подтверждения команды перехода")
            return False
        elif ack.result != mavutil.mavlink.MAV_RESULT_ACCEPTED:
            print(f"Ошибка: Команда отклонена ({ack.result})")
            return False

        print("Команда перехода принята, ожидаем смены режима...")
        return True
        
    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10000: 
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                center_x, center_y = int(rect[0][0]), int(rect[0][1])
                
                frame_center_x, frame_center_y = frame.shape[1]//2, frame.shape[0]//2
                dx_px = center_x - frame_center_x
                dy_px = center_y - frame_center_y
                current_lat, current_lon, _ = self.get_coordinates() 

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
        from cv2 import aruco
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        self.send_ssh_message(" Поиск ArUco маркера (USB-камера)...")
        
        last_save_time = time.time()

        while self.current_mode == MODE_ARUCO:
            new_mode = self.check_mode()
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
                
                offset_north, offset_east = self.calculate_offset(tvecs[0])
                
                if not self.drone_in_vtol:
                    if self.switch_to_vtol_mode():
                        self.master.set_mode("GUIDED")
                        self.execute_landing(offset_north, offset_east)

                if time.time() - last_save_time > 1: 
                    self.send_ssh_message(f" Offset: N = {offset_north:.2f} m, E = {offset_east:.2f} m") 
                    print(f" Offset: N = {offset_north:.2f} m, E = {offset_east:.2f} m")
                    cv2.aruco.drawDetectedMarkers(frame, [corners[idx]])
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[0], tvecs[0], 0.1)
            time.sleep(LOOP_DELAY)

    def process_letters(self):
        self.send_ssh_message("=== Активирован режим распознавания букв ===")
        self.last_processing_time = time.time()
        self.last_save_time = time.time()
        
        try:
            while self.current_mode == MODE_LETTERS:
                new_mode = self.check_mode()
                if new_mode != self.current_mode:
                    self.send_ssh_message(" Выход из режима LETTERS")
                    self.current_mode = new_mode
                    return

                current_time = time.time()
                if current_time - self.last_processing_time < self.processing_interval:
                    time.sleep(0.01)
                    continue
                
                self.last_processing_time = current_time
                
                frame = self.get_frame()
                if frame is None:
                    time.sleep(LOOP_DELAY)
                    continue
                
                results, thresh_color = self.process_frame(frame)
                processed_letter = None
                
                for result in results:
                    cv2.drawContours(frame, [result['box']], 0, (0, 255, 0), 2)
                    cv2.circle(frame, result['center_px'], 5, (0, 0, 255), -1)
                    
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
                                        self.send_ssh_message(f"Распознано: {text} | Координаты: {result['coords']}")
                        except Exception as e:
                            logging.error(f"Ошибка обработки буквы: {e}")
               
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
                
        if rc_val is not None and rc_val != self.last_rc_value:
            new_mode = self.rc_to_mode(rc_val)
            self.last_rc_value = rc_val
            logging.info(f"RC: {rc_val} → {self.mode_name(new_mode)}")
            print(f"RC: {rc_val} → {self.mode_name(new_mode)}") 
                
        elif wp is not None and wp != self.last_wp_value and wp in self.wp_modes:
            new_mode = self.wp_modes[wp]
            self.last_wp_value = wp
            logging.info(f"WP: {wp} → {self.mode_name(new_mode)}")
            self.send_ssh_message(f"WP: {wp} → {self.mode_name(new_mode)}")
                
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
                try:  # Внутренний обработчик ошибок
                    self.current_mode = self.check_mode()
                    
                    # Всегда освобождаем камеру при переходе в OFF
                    if self.current_mode == MODE_OFF:
                        self.release_camera()
                        time.sleep(LOOP_DELAY)
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
                            self.no_process = False
                        else:
                            self.send_ssh_message("Ошибка инициализации камеры для LETTERS")
                            self.current_mode = MODE_OFF

                    elif self.current_mode == MODE_ARUCO and not self.process_ARUCO:
                        self.USE_PICAM = False
                        self.USE_USB_CAM = True 
                        
                        try:
                            self.master.mav.command_long_send(
                                self.master.target_system, 
                                self.master.target_component, 
                                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                                0, 
                                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 
                                15, 0, 0, 0, 0, 0
                            )
                        except Exception as e:
                            logging.error(f"Ошибка установки режима: {e}")
                        
                        self.init_camera()
                        
                        # Проверяем успешность инициализации
                        if self.camera and (self.USE_USB_CAM and self.camera.isOpened()):
                            self.process_aruco()
                            self.process_ARUCO = True
                            self.process_LETTERS = False
                            self.no_process = False
                        else:
                            self.send_ssh_message("Ошибка инициализации камеры для ARUCO")
                            self.current_mode = MODE_OFF

                    time.sleep(LOOP_DELAY)
                    
                except Exception as inner_e:
                    logging.error(f"Ошибка в основном цикле: {inner_e}")
                    self.send_ssh_message(f"Внутренняя ошибка: {inner_e}")
                    time.sleep(1)
                
        except KeyboardInterrupt:
            self.send_ssh_message("--- Завершение работы ---")
            self.release_camera()
        except Exception as e:
            self.send_ssh_message(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
            logging.critical(f"Необработанное исключение: {e}")

if __name__ == '__main__':
    controller = DroneController()
    controller.run()
