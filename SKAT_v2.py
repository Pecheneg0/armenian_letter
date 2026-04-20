import cv2
import numpy as np
import time
import torch
import threading
import queue
from pymavlink import mavutil
from torchvision import transforms
from PIL import Image
import os
import math
import logging
import sys
from datetime import datetime 
from concurrent.futures import ThreadPoolExecutor
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

CAMERA_RESOLUTION = (1280, 720)  # Уменьшено для производительности
PROCESSING_RESOLUTION = (640, 360)  # Разрешение для обработки

MARKER_ID = 42
CAMERA_ID = 1

MODEL_PATH = "ALM_best.pth"
LABELS_PATH = "labels.txt"
CONFIDENCE_THRESHOLD = 0.85

TELEMETRY_INTERVAL = 2
LOOP_DELAY = 0.05  # Уменьшено для большей отзывчивости
LANDING_TIMEOUT = 120  # Максимальное время посадки (сек)

PREVIEW_RES = (640, 360)
PROCESSING_FPS = 10
PIXELS_PER_METER = 100  
SAFE_LAND_ALT = 0.5  # Минимальная безопасная высота

ARUCO_DIR = "/home/pi/tests/aruco_images"
LETTERS_DIR = "/home/pi/tests/letters_images"
LOG_DIR = "/home/pi/tests/logs"
MARKER_LENGTH = 0.26 

# Создаем директории
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARUCO_DIR, exist_ok=True)
os.makedirs(LETTERS_DIR, exist_ok=True)

# Настройка логирования
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "sla_controller.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DroneController")

# Параметры камеры
frame_w, frame_h = PROCESSING_RESOLUTION
camera_matrix = np.array([
    [800, 0, frame_w / 2],
    [0, 800, frame_h / 2],
    [0, 0, 1]
], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

class FPSCounter:
    """Контроллер частоты кадров"""
    def __init__(self, target_fps):
        self.interval = 1.0 / target_fps
        self.last_time = time.monotonic()
    
    def ready(self):
        current = time.monotonic()
        if current - self.last_time >= self.interval:
            self.last_time = current
            return True
        return False

class PositionCache:
    """Кэш для позиционных данных"""
    def __init__(self, update_interval=0.5):
        self.update_interval = update_interval
        self.last_update = 0
        self.cached_position = (0, 0, 0)
        self.cached_yaw = 0
    
    def update(self, getter_func):
        current_time = time.monotonic()
        if current_time - self.last_update > self.update_interval:
            self.cached_position = getter_func()
            self.cached_yaw = self.get_yaw()
            self.last_update = current_time
    
    def get_position(self):
        return self.cached_position
    
    def get_yaw(self):
        # Заглушка - в реальном коде нужно получать курс
        return 0

class DroneController:
    def __init__(self):
        self.master = None
        self.current_mode = MODE_OFF
        self.current_mode_str = self.mode_names[MODE_OFF]
        self.camera = None
        self.model = None
        self.labels = []
        
        # Инициализация компонентов
        self.connect_to_pixhawk()
        self.load_model()
        self.init_cameras()
        
        # Оптимизационные компоненты
        self.fps_aruco = FPSCounter(PROCESSING_FPS)
        self.fps_letters = FPSCounter(PROCESSING_FPS)
        self.position_cache = PositionCache()
        self.frame_queue = queue.Queue(maxsize=2)  # Очередь для обработки кадров
        self.land_command_sent = False
        
        # Режимы работы
        self.mode_names = {
            MODE_OFF: "OFF",
            MODE_LETTERS: "LETTERS",
            MODE_ARUCO: "ARUCO"
        }
        
        # Соответствие точек миссии режимам
        self.wp_modes = {9: MODE_LETTERS, 17: MODE_ARUCO, 0: MODE_OFF}
        
        # Преобразования изображений
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Запуск потока обработки
        self.processing_thread = threading.Thread(target=self.processing_worker, daemon=True)
        self.processing_thread.start()
        
        logger.info("Контроллер инициализирован")

    def connect_to_pixhawk(self):
        """Подключение к Pixhawk с повторными попытками"""
        while True:
            try:
                self.master = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD)
                self.master.wait_heartbeat(timeout=5)
                logger.info(f"Подключено к системе {self.master.target_system}")
                return
            except Exception as e:
                logger.error(f"Ошибка подключения: {e}. Повтор через 5 сек...")
                time.sleep(5)

    def init_cameras(self):
        """Инициализация камер один раз при старте"""
        try:
            # PiCamera для режима LETTERS
            self.picam = Picamera2()
            config = self.picam.create_preview_configuration(
                main={"size": CAMERA_RESOLUTION, "format": "RGB888"},
                controls={"FrameRate": PROCESSING_FPS}
            )
            self.picam.configure(config)
            self.picam.start()
            logger.info("PiCamera инициализирована")
            
            # USB-камера для режима ARUCO
            self.usb_cam = cv2.VideoCapture(CAMERA_ID)
            self.usb_cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
            self.usb_cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
            logger.info("USB-камера инициализирована")
            
        except Exception as e:
            logger.error(f"Ошибка инициализации камер: {e}")
            raise

    def get_frame(self):
        """Получение кадра в зависимости от режима"""
        try:
            if self.current_mode == MODE_LETTERS:
                frame = self.picam.capture_array()
                return cv2.resize(frame, PROCESSING_RESOLUTION)  # Сразу уменьшаем
                
            elif self.current_mode == MODE_ARUCO:
                ret, frame = self.usb_cam.read()
                if ret:
                    return cv2.resize(frame, PROCESSING_RESOLUTION)  # Сразу уменьшаем
        except Exception as e:
            logger.error(f"Ошибка получения кадра: {e}")
        return None

    def get_current_waypoint(self):
        """Неблокирующее получение текущей точки миссии"""
        msg = self.master.recv_match(type='MISSION_CURRENT', blocking=False, timeout=0.1)
        return msg.seq if msg else None

    def get_rc_value(self):
        """Неблокирующее получение значения RC канала"""
        msg = self.master.recv_match(type='RC_CHANNELS', blocking=False, timeout=0.1)
        if msg:
            try:
                return getattr(msg, f'chan{RC_CHANNEL}_raw')
            except AttributeError:
                pass
        return None

    def rc_to_mode(self, rc_value):
        """Преобразование значения RC в режим"""
        if rc_value < 1200: return MODE_OFF
        if rc_value < 1700: return MODE_LETTERS
        return MODE_ARUCO

    def send_ssh_message(self, message):
        """Отправка сообщения в SSH-сессию"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] {message}")
        sys.stdout.flush()

    def load_model(self):
        """Загрузка модели машинного обучения"""
        try:
            self.model = ArmenianLetterNet()
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            self.model.eval()
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f.readlines()]
            logger.info("Модель загружена")
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            raise

    def get_current_altitude(self):
        """Получение высоты с кэшированием"""
        msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=True, timeout=0.5)
        return msg.relative_alt / 1000 if msg else 0

    def move_to_offset(self, dx, dy, target_alt=None):
        """Команда перемещения с оптимизацией"""
        current_alt = self.get_current_altitude()
        target_alt = target_alt or current_alt
        
        # Ограничение минимальной высоты
        target_alt = max(target_alt, self.min_altitude)
        
        # Отправка команды
        self.master.mav.set_position_target_local_ned_send(
            int(time.monotonic() * 1000),
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_OFFSET_NED,
            0b0000111111111000,  # Используем только позицию
            dx, dy, -target_alt,  # Z вниз, поэтому отрицательное значение
            0, 0, 0, 0, 0, 0, 0, 0
        )

    def get_yaw(self):
        """Неблокирующее получение курса"""
        msg = self.master.recv_match(type='ATTITUDE', blocking=False, timeout=0.1)
        return math.degrees(msg.yaw) if msg else 0

    def get_coordinates(self):
        """Получение координат с кэшированием"""
        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=0.5)
        if msg:
            return msg.lat / 1e7, msg.lon / 1e7
        return 0, 0

    def switch_to_vtol_mode(self):
        """Переключение в VTOL режим"""
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
            0, 3, 0, 0, 0, 0, 0, 0
        )
        self.send_ssh_message("Запрос перехода в VTOL режим")

        # Ожидание подтверждения
        ack = self.master.recv_match(
            type='COMMAND_ACK',
            command=mavutil.mavlink.MAV_CMD_DO_VTOL_TRANSITION,
            blocking=True,
            timeout=2
        )
        
        if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
            self.send_ssh_message("Переход в VTOL подтвержден")
            return True
        
        self.send_ssh_message("Ошибка перехода в VTOL режим")
        return False

    def execute_landing(self, offset_north, offset_east):
        """Процедура посадки с таймаутом"""
        self.send_ssh_message("Начало процедуры посадки")
        start_time = time.monotonic()
        
        while time.monotonic() - start_time < LANDING_TIMEOUT:
            altitude = self.get_current_altitude()
            
            # Условия прерывания
            if self.land_command_sent or altitude < SAFE_LAND_ALT:
                break
                
            # Коррекция положения
            self.move_to_offset(offset_north, offset_east, altitude * 0.9)
            time.sleep(1)
        
        # Финализация посадки
        if altitude <= SAFE_LAND_ALT:
            self.master.set_mode("QLAND")
            self.land_command_sent = True
            self.send_ssh_message("Команда посадки отправлена")
        else:
            self.send_ssh_message("Таймаут посадки! Аварийное завершение")
            
        return self.land_command_sent

    def processing_worker(self):
        """Фоновый поток для ресурсоемкой обработки"""
        while True:
            try:
                task = self.frame_queue.get(timeout=1.0)
                mode, frame, timestamp = task
                
                if mode == MODE_ARUCO:
                    self.process_aruco_frame(frame, timestamp)
                elif mode == MODE_LETTERS:
                    self.process_letters_frame(frame, timestamp)
                    
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"Ошибка в обработчике: {e}")

    def process_aruco_frame(self, frame, timestamp):
        """Обработка ArUco маркера в фоновом потоке"""
        try:
            from cv2 import aruco
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
            detector = aruco.ArucoDetector(aruco_dict)
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            
            if ids is not None and MARKER_ID in ids.flatten():
                idx = list(ids.flatten()).index(MARKER_ID)
                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    [corners[idx]], MARKER_LENGTH, camera_matrix, dist_coeffs)
                
                # Расчет смещения
                x_cam = tvecs[0][0][0]
                z_cam = tvecs[0][0][2]
                yaw_deg = self.position_cache.cached_yaw
                yaw_rad = np.radians(yaw_deg)
                
                R = np.array([
                    [np.cos(yaw_rad), -np.sin(yaw_rad)],
                    [np.sin(yaw_rad),  np.cos(yaw_rad)]
                ])
                
                offset = R @ np.array([x_cam, z_cam])
                offset_north, offset_east = offset[1], offset[0]
                
                # Сохранение кадра для отладки
                cv2.aruco.drawDetectedMarkers(frame, [corners[idx]])
                filename = os.path.join(ARUCO_DIR, f"aruco_{timestamp}.jpg")
                cv2.imwrite(filename, frame)
                
                # Переключение и посадка
                if self.switch_to_vtol_mode():
                    self.execute_landing(offset_north, offset_east)
                    
        except Exception as e:
            logger.error(f"Ошибка обработки ArUco: {e}")

    def process_letters_frame(self, frame, timestamp):
        """Обработка армянских букв в фоновом потоке"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Параллельная обработка контуров
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = []
                for cnt in contours:
                    if cv2.contourArea(cnt) > 1000:
                        futures.append(executor.submit(self.process_contour, cnt, thresh))
                
                # Обработка результатов
                for future in futures:
                    result = future.result()
                    if result:
                        label, conf, coords = result
                        self.send_ssh_message(f"Распознано: {label} ({conf:.2f})")
            
            # Сохранение кадра
            filename = os.path.join(LETTERS_DIR, f"letter_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            
        except Exception as e:
            logger.error(f"Ошибка обработки букв: {e}")

    def process_contour(self, cnt, thresh_img):
        """Обработка одного контура буквы"""
        try:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Вычисление центра
            center_x, center_y = rect[0]
            
            # Преобразование перспективы
            width, height = map(int, rect[1])
            if width > 0 and height > 0:
                src_pts = box.astype("float32")
                dst_pts = np.array([
                    [0, height-1], 
                    [0, 0], 
                    [width-1, 0], 
                    [width-1, height-1]
                ], dtype="float32")
                
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                letter_crop = cv2.warpPerspective(thresh_img, M, (width, height))
                
                if np.mean(letter_crop) > 50:  # Проверка на валидность
                    img_pil = Image.fromarray(letter_crop)
                    img_tensor = self.transform(img_pil).unsqueeze(0)
                    
                    with torch.no_grad():
                        output = self.model(img_tensor)
                        conf, pred = torch.max(torch.nn.functional.softmax(output, dim=1), 1)
                        conf = conf.item()
                        
                        if conf > CONFIDENCE_THRESHOLD:
                            label = self.labels[pred.item()]
                            return label, conf, (center_x, center_y)
        except Exception as e:
            logger.error(f"Ошибка обработки контура: {e}")
        return None

    def handle_mode_switch(self, new_mode):
        """Обработка переключения режима"""
        new_mode_name = self.mode_names.get(new_mode, "UNKNOWN")
        if not hasattr(self, 'current_mode_str'):
            self.current_mode_str =self.mode_names.get(self.current_mode, "UNKNOWN")
        if new_mode_name != self.current_mode_str:
            self.send_ssh_message(f"Режим: {self.current_mode_str} → {new_mode_name}")
            self.current_mode_str = new_mode_name
            self.current_mode = new_mode

    def run(self):
        """Основной цикл управления"""
        self.send_ssh_message("Запуск контроллера")
        last_telemetry = time.monotonic()
        
        try:
            while True:
                # Обновление кэша позиции
                self.position_cache.update(self.get_coordinates)
                
                # Проверка и переключение режима
                self.check_mode()
                
                # Обработка текущего режима
                if self.current_mode == MODE_OFF:
                    time.sleep(LOOP_DELAY)
                    continue
                    
                # Получение кадра
                frame = self.get_frame()
                if frame is None:
                    time.sleep(LOOP_DELAY)
                    continue
                
                # Проверка частоты кадров
                fps_counter = self.fps_aruco if self.current_mode == MODE_ARUCO else self.fps_letters
                if not fps_counter.ready():
                    time.sleep(0.01)
                    continue
                
                # Отправка в очередь обработки
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                try:
                    self.frame_queue.put((self.current_mode, frame.copy(), timestamp), timeout=0.1)
                except queue.Full:
                    pass
                
                # Отправка телеметрии
                if time.monotonic() - last_telemetry > TELEMETRY_INTERVAL:
                    alt = self.get_current_altitude()
                    self.send_ssh_message(f"Высота: {alt:.1f}m, Режим: {self.current_mode_str}")
                    last_telemetry = time.monotonic()
                
                time.sleep(LOOP_DELAY)
                
        except KeyboardInterrupt:
            self.send_ssh_message("Завершение работы")
        finally:
            self.cleanup()

    def cleanup(self):
        """Очистка ресурсов при завершении"""
        if self.picam:
            self.picam.stop()
            self.picam.close()
        if self.usb_cam:
            self.usb_cam.release()
        cv2.destroyAllWindows()
        logger.info("Ресурсы освобождены")

    def check_mode(self):
        """Проверка и обновление текущего режима"""
        # Проверка RC
        rc_val = self.get_rc_value()
        if rc_val is not None:
            new_mode = self.rc_to_mode(rc_val)
            if new_mode != self.current_mode:
                self.handle_mode_switch(new_mode)
                return
                
        # Проверка точки миссии
        wp = self.get_current_waypoint()
        if wp is not None and wp in self.wp_modes:
            new_mode = self.wp_modes[wp]
            if new_mode != self.current_mode:
                self.handle_mode_switch(new_mode)

if __name__ == '__main__':
    controller = DroneController()
    controller.run()
