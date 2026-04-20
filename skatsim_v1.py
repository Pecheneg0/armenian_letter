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
#from libcamera import controls # Раскомментируйте, если нужно специфическое управление
from modelold import ArmenianLetterNet
from collections import defaultdict
from cv2 import aruco
import resource
import threading
import matplotlib
matplotlib.use('Agg')  # Headless-режим для Raspberry Pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import base64
from datetime import datetime
import json

# ===== МОДУЛЬ ЭМУЛЯЦИИ MAVLINK (ИСПРАВЛЕННЫЙ) =====
class MockMavlink:
    def __init__(self):
        self.target_system = 1
        self.target_component = 1  # <--- ИСПРАВЛЕНО: Добавлен отсутствующий атрибут
        self.mav = self 
        
        # Начальные координаты
        self.sim_lat = 55.754066  
        self.sim_lon = 37.617498
        self.sim_alt_agl = 30.0   
        self.sim_alt_amsl = 150.0 
        self.sim_yaw = 0.0        
        
        # Состояние миссии
        self.current_wp_seq = 0   
        self.wp_change_time = time.time()
        
        # Состояние посадки
        self.guided_mode_set = False
        self.qland_mode_set = False

    def wait_heartbeat(self):
        print("[MOCK] Heartbeat received from simulated drone.")
        return True

    def recv_match(self, type=None, blocking=False, timeout=0):
        if type == 'MISSION_CURRENT':
            if time.time() - self.wp_change_time > 45: # Смена WP каждые 15 сек
                self.current_wp_seq = (self.current_wp_seq + 1) % 3 
                self.wp_change_time = time.time()
                print(f"[MOCK] >>> Waypoint changed to: {self.current_wp_seq}")
            
            class Msg:
                seq = self.current_wp_seq
            return Msg()

        elif type == 'GLOBAL_POSITION_INT':
            class Msg:
                lat = int(self.sim_lat * 1e7)
                lon = int(self.sim_lon * 1e7)
                alt = int(self.sim_alt_amsl * 1e3)
                relative_alt = int(self.sim_alt_agl * 1e3)
                vx = 0; vy = 0; vz = 0; hdg = 0
            return Msg()

        elif type == 'ATTITUDE':
            class Msg:
                roll = 0; pitch = 0; yaw = math.radians(self.sim_yaw)
                rollspeed = 0; pitchspeed = 0; yawspeed = 0
            return Msg()
        
        elif type == 'RC_CHANNELS':
             class Msg:
                 chan6_raw = 2000 
             return Msg()

        elif type == 'SERVO_OUTPUT_RAW':
             return None 

        return None

    def set_mode(self, mode_name):
        print(f"[MOCK] Drone mode set to: {mode_name}")
        self.guided_mode_set = (mode_name == "GUIDED")
        self.qland_mode_set = (mode_name == "QLAND")

    def command_long_send(self, target_sys, target_comp, cmd_id, conf, p1, p2, p3, p4, p5, p6, p7):
        if cmd_id == 31000: 
            print(f"[MOCK COMMAND] Offset sent: North={p1:.2f}m, East={p2:.2f}m")
            
            if self.guided_mode_set and not self.qland_mode_set:
                dist = math.sqrt(p1**2 + p2**2)
                if dist < 2.0:
                    self.sim_alt_agl = max(0.1, self.sim_alt_agl - 0.5) 
                    if self.sim_alt_agl <= 0.2:
                        print("[MOCK] Touchdown simulated!")
                else:
                    step_factor = 0.1
                    self.sim_lat += (p1 / 111000) * step_factor
                    self.sim_lon += (p2 / (111000 * math.cos(math.radians(self.sim_lat)))) * step_factor

    def set_position_target_local_ned_send(self, *args, **kwargs):
        pass

# Подменяем подключение
import pymavlink.mavutil as real_mavutil
def mock_connection(*args, **kwargs):
    return MockMavlink()
real_mavutil.mavlink_connection = mock_connection


# ===== КОНФИГУРАЦИЯ =====
MAVLINK_PORT = '/dev/ttyAMA0'
MAVLINK_BAUD = 57600
RC_CHANNEL = 6

MODE_OFF = 0
MODE_LETTERS = 1
MODE_ARUCO = 2
MODE_MGM = 3

CAMERA_RESOLUTION_PICAM = (800, 600)  
CAMERA_RESOLUTION_USBCAM = (720, 720)

MARKER_42 = 42  
MARKER_41 = 41
LENGTH_41 = 0.08 
LENGTH_42 = 0.422
# ВАЖНО: Попробуйте изменить CAMERA_ID на 1 или 2, если USB камера не открывается
CAMERA_ID = 1 

DROP_SERVO_PWM_THRESHOLD = 1700
MIN_AREA = 11000           
MAX_AREA = 400000

MODEL_PATH = "ALM_best.pth"
LABELS_PATH = "labels.txt"
MAV_CMD_USER_1 = 31000

TELEMETRY_INTERVAL = 2
LOOP_DELAY = 0.02

CONFIDENCE_THRESHOLD = 0.7   
PROCESSING_FPS = 30 # Снижено для стабильности
PIXELS_PER_METER = 44.482  

ARUCO_DIR = "/home/pi/Desktop/aruco_images"
LETTERS_DIR = "/home/pi/Desktop/letters_images"
LOG_DIR = "/home/pi/Desktop"
MGM_DIR = "/home/pi/Desktop/mgm_images"
CSV_PATH = "/home/pi/Desktop/objects-coordinates.csv" 
 
MAX_ALTITUDE_M = 150.0 
ANOMALOUS_HEIGHT = 14.5 

MODE_MGM = 3  
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

#torch.set_num_threads(2)
#cv2.setNumThreads(2)

try:
    resource.setrlimit(resource.RLIMIT_AS, (2*1024*1024*1024, 2*1024*1024*1024))  
except:
    pass  

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(ARUCO_DIR, exist_ok=True)
os.makedirs(LETTERS_DIR, exist_ok=True)
os.makedirs(MGM_DIR, exist_ok=True)

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, 'w') as f:
        f.write("index,lat_e7,lon_e7\n")

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "SKAT_mgm.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

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
        self.wp_interval = 2
        self.rc_interval = 0.25

        self._last_lat = 0.0
        self._last_lon = 0.0
        self._last_alt = 0.0 
        self._last_alt_agl = 0.0
        self._last_yaw = 0.0

        self.letter_stats = defaultdict(lambda: {'confidences': [], 'coords': [], 'timestamps': []})
        self.processing_active  = False
        self.last_detection_time = 0
        self.processing_sessions = []
        self.LETTER_TIMEOUT = 2.0
        self.MIN_DETECTIONS = 3
        self.SAVE_INTERVAL = 0.3
        self.last_save_time = 0


        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.last_processing_time = 0
        self.last_save_time = 0
        self.processing_interval = 1.0 / PROCESSING_FPS
        self.is_pixhawk_connected = False
        self.model_loaded = False

        # Для сбора статистики посадки
        self.landing_offsets_log = []  # [{'time': t, 'north': n, 'east': e, 'alt': h, 'yaw': y}, ...]
        self.aruco_start_time = None
        
        # Для сбора статистики букв
        self.mission_start_time = time.time()
        self.total_letters_detected = 0
        self.total_aruco_frames = 0
        
        # Для отчёта
        self.report_data = {
            'mission_start': None,
            'mission_end': None,
            'modes_used': [],
            'letters_sessions': [],
            'aruco_landing': {
                'performed': False,
                'target_markers': [42, 41],
                'switch_altitude': None,
                'final_offset_n': 0.0,
                'final_offset_e': 0.0,
                'max_offset': 0.0,
                'total_frames': 0,
                'trajectory_points': []
            },
            'system_stats': {
                'pixhawk_connected': False,
                'model_loaded': False,
                'cpu_temp_max': 0.0,
                'memory_usage_max': 0.0
            }
        }
        
        if not self.is_pixhawk_connected:
            self.connect_to_pixhawk()
            self.is_pixhawk_connected = True
    
        # Для теста: 0=OFF, 1=LETTERS, 2=ARUCO
        self.wp_modes = {3: MODE_OFF, 4: MODE_LETTERS, 0: MODE_ARUCO, 20: MODE_MGM}
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    

    def generate_3d_trajectory_graph(self, report_dir, filename_base):
        """Генерирует 3D-график траектории посадки"""
        if not self.landing_offsets_log:
            return None, "Нет данных для графика"
        
        try:
            # Извлекаем данные
            east_offsets = [x['east'] for x in self.landing_offsets_log]
            north_offsets = [x['north'] for x in self.landing_offsets_log]
            altitudes = [x['alt'] for x in self.landing_offsets_log]
            times = [x['time'] for x in self.landing_offsets_log]
            
            # Создаём 3D-фигуру
            fig = plt.figure(figsize=(12, 9), dpi=150)
            ax = fig.add_subplot(111, projection='3d')
            
            # Цветовая кодировка по времени
            norm = plt.Normalize(vmin=min(times), vmax=max(times))
            scatter = ax.scatter(
                east_offsets, north_offsets, altitudes,
                c=times, cmap='viridis', s=30, alpha=0.8,
                edgecolors='black', linewidth=0.3, norm=norm
            )
            
            # Линия траектории
            ax.plot(east_offsets, north_offsets, altitudes, 
                    color='gray', linewidth=0.5, alpha=0.3, label='Траектория')
            
            # Оси
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            ax.set_xlabel('Восток-Запад (м)\n(+ = восток)', fontsize=10, labelpad=10)
            ax.set_ylabel('Север-Юг (м)\n(+ = север)', fontsize=10, labelpad=10)
            ax.set_zlabel('Высота (м)', fontsize=10, labelpad=10)
            ax.set_zlim(0, max(50, max(altitudes) * 1.2))
            
            # Сетка и зона допуска
            ax.grid(True, alpha=0.3)
            tolerance = self.pos_tolerance
            theta = np.linspace(0, 2*np.pi, 50)
            x_cyl = tolerance * np.cos(theta)
            y_cyl = tolerance * np.sin(theta)
            z_cyl = np.linspace(0, 2, 2)
            X_cyl, Z_cyl = np.meshgrid(x_cyl, z_cyl)
            Y_cyl, Z_cyl = np.meshgrid(y_cyl, z_cyl)
            ax.plot_surface(X_cyl, Y_cyl, Z_cyl, color='green', alpha=0.05)
            
            # Целевая точка
            ax.scatter([0], [0], [0], color='red', s=80, marker='*', 
                    label='Целевая точка', zorder=5)
            
            # Цветовая шкала
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1, 
                            label='Время (сек)')
            ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
            ax.set_title('3D-траектория снижения по ArUco', fontsize=13, 
                        fontweight='bold', pad=20)
            ax.view_init(elev=25, azim=45)
            
            # Сохранение
            os.makedirs(report_dir, exist_ok=True)
            png_path = f"{report_dir}/{filename_base}_trajectory_3d.png"
            pdf_path = f"{report_dir}/{filename_base}_trajectory_3d.pdf"
            
            plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
            plt.close()
            
            return {'png': png_path, 'pdf': pdf_path}, None
            
        except Exception as e:
            return None, f"Ошибка построения 3D-графика: {e}"

    def generate_landing_gif(self, report_dir, filename_base):
        """Генерирует GIF-анимацию траектории"""
        if not self.landing_offsets_log or len(self.landing_offsets_log) < 10:
            return None, "Недостаточно данных для GIF"
        
        try:
            east_offsets = [x['east'] for x in self.landing_offsets_log]
            north_offsets = [x['north'] for x in self.landing_offsets_log]
            altitudes = [x['alt'] for x in self.landing_offsets_log]
            
            fig = plt.figure(figsize=(10, 8), dpi=100)
            ax = fig.add_subplot(111, projection='3d')
            
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            ax.set_zlim(0, max(50, max(altitudes) * 1.2))
            ax.set_xlabel('Восток-Запад (м)')
            ax.set_ylabel('Север-Юг (м)')
            ax.set_zlabel('Высота (м)')
            
            # Начальная точка
            point, = ax.plot([], [], [], 'ro', markersize=6)
            line, = ax.plot([], [], [], 'b-', linewidth=1, alpha=0.6)
            
            def init():
                point.set_data([], [])
                point.set_3d_properties([])
                line.set_data([], [])
                line.set_3d_properties([])
                return point, line
            
            def update(frame):
                point.set_data([east_offsets[frame]], [north_offsets[frame]])
                point.set_3d_properties([altitudes[frame]])
                
                if frame > 0:
                    line.set_data(east_offsets[:frame+1], north_offsets[:frame+1])
                    line.set_3d_properties(altitudes[:frame+1])
                
                return point, line
            
            anim = FuncAnimation(fig, update, frames=len(east_offsets), 
                                init_func=init, blit=True, interval=100)
            
            gif_path = f"{report_dir}/{filename_base}_landing.gif"
            anim.save(gif_path, writer='pillow', fps=10)
            plt.close()
            
            return gif_path, None
            
        except Exception as e:
            return None, f"Ошибка создания GIF: {e}"

    def generate_pdf_report(self, report_dir, filename_base):
        """Генерирует полный PDF-отчёт"""
        try:
            pdf_path = f"{report_dir}/{filename_base}_report.pdf"
            doc = SimpleDocTemplate(pdf_path, pagesize=A4,
                                rightMargin=2*cm, leftMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
            
            styles = getSampleStyleSheet()
            styles.add(ParagraphStyle(name='Mono', fontName='Courier', fontSize=9, leading=11))
            styles.add(ParagraphStyle(name='Center', alignment=TA_CENTER))
            styles.add(ParagraphStyle(name='Heading3', parent=styles['Heading2'], fontSize=12))
            
            story = []
            
            # Заголовок
            story.append(Paragraph('<b>ПРОТОКОЛ ИСПЫТАНИЙ</b>', styles['Title']))
            story.append(Paragraph('<b>АВТОМАТИЧЕСКАЯ ПОСАДКА ПО ARUCO МАРКЕРАМ</b>', styles['Heading2']))
            story.append(Spacer(1, 0.5*cm))
            
            # Метаданные
            story.append(Paragraph('<b>1. ОБЩИЕ СВЕДЕНИЯ</b>', styles['Heading3']))
            meta_table = Table([
                ['Дата начала:', self.report_data.get('mission_start', 'N/A')],
                ['Дата окончания:', self.report_data.get('mission_end', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))],
                ['Версия ПО:', 'SKAT_mgm v9.3.2'],
                ['Pixhawk подключен:', 'Да' if self.report_data['system_stats']['pixhawk_connected'] else 'Нет']
            ], colWidths=[5*cm, 11*cm])
            meta_table.setStyle(TableStyle([
                ('FONTNAME', (0,0), (0,-1), 'Courier-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 10),
                ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                ('BACKGROUND', (0,0), (0,-1), colors.lightgrey)
            ]))
            story.append(meta_table)
            story.append(Spacer(1, 1*cm))
            
            # Статистика посадки
            story.append(Paragraph('<b>2. АНАЛИЗ ТОЧНОСТИ ПОСАДКИ (ARUCO)</b>', styles['Heading3']))
            aruco_data = self.report_data['aruco_landing']
            
            if aruco_data['performed']:
                stats_table = Table([
                    ['Параметр', 'Значение', 'Допуск', 'Статус'],
                    ['Финальное смещение N (м)', f"{aruco_data['final_offset_n']:.3f}", '< 0.15', 
                    'PASS' if abs(aruco_data['final_offset_n']) < 0.15 else 'FAIL'],
                    ['Финальное смещение E (м)', f"{aruco_data['final_offset_e']:.3f}", '< 0.15',
                    'PASS' if abs(aruco_data['final_offset_e']) < 0.15 else 'FAIL'],
                    ['Макс. отклонение (м)', f"{aruco_data['max_offset']:.3f}", '—', 'INFO'],
                    ['Всего кадров', str(aruco_data['total_frames']), '—', '—'],
                    ['Точек в траектории', str(len(self.landing_offsets_log)), '—', '—']
                ], colWidths=[5*cm, 4*cm, 3*cm, 2.5*cm])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                    ('FONTNAME', (0,0), (-1,0), 'Courier-Bold'),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    ('ALIGN', (1,1), (1,-1), 'RIGHT')
                ]))
                story.append(stats_table)
            else:
                story.append(Paragraph('<i>Посадка по ArUco не выполнялась</i>', styles['Normal']))
            
            story.append(Spacer(1, 1*cm))
            
            # Распознавание букв
            story.append(Paragraph('<b>3. РАСПОЗНАВАНИЕ ОБЪЕКТОВ (LETTERS)</b>', styles['Heading3']))
            if self.processing_sessions:
                story.append(Paragraph(f'Всего сеансов: {len(self.processing_sessions)}', styles['Normal']))
                story.append(Spacer(1, 0.3*cm))
                
                letters_table_data = [['№', 'Буква', 'Обнаружений', 'Уверенность', 'Координаты']]
                for i, session in enumerate(self.processing_sessions, 1):
                    letters_table_data.append([
                        str(i),
                        session['letter'],
                        str(session['count']),
                        f"{session['avg_confidence']:.2f}",
                        f"{session['avg_coords'][0]:.6f}, {session['avg_coords'][1]:.6f}"
                    ])
                
                letters_table = Table(letters_table_data, colWidths=[1*cm, 3*cm, 3*cm, 3*cm, 6*cm])
                letters_table.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
                    ('FONTNAME', (0,0), (-1,0), 'Courier-Bold'),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
                    ('FONTSIZE', (0,0), (-1,-1), 9)
                ]))
                story.append(letters_table)
            else:
                story.append(Paragraph('<i>Распознавание букв не выполнялось</i>', styles['Normal']))
            
            story.append(Spacer(1, 1*cm))
            
            # 3D График
            story.append(Paragraph('<b>4. ТРАЕКТОРИЯ СНИЖЕНИЯ</b>', styles['Heading3']))
            graph_path = f"{report_dir}/{filename_base}_trajectory_3d.png"
            if os.path.exists(graph_path):
                img = Image(graph_path, width=16*cm, height=12*cm)
                story.append(img)
                story.append(Spacer(1, 0.5*cm))
                story.append(Paragraph('<i>Рисунок 1 — 3D-траектория снижения по ArUco маркерам</i>', styles['Normal']))
            else:
                story.append(Paragraph('<i>График не сгенерирован</i>', styles['Normal']))
            
            story.append(Spacer(1, 1*cm))
            
            # Футер
            story.append(Paragraph('<hr/>', styles['Normal']))
            story.append(Paragraph(f'<i>Сформировано автоматически: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</i>', styles['Normal']))
            
            # Сборка PDF
            doc.build(story)
            return pdf_path
            
        except Exception as e:
            logging.error(f"Ошибка генерации PDF: {e}")
            return None

    def log_aruco_data(self, offset_north, offset_east, altitude):
        """Записывает данные для отчёта во время посадки"""
        if self.aruco_start_time is None:
            self.aruco_start_time = time.time()
        
        current_time = time.time() - self.aruco_start_time
        yaw = self.get_yaw()
        
        self.landing_offsets_log.append({
            'time': current_time,
            'north': offset_north,
            'east': offset_east,
            'alt': altitude,
            'yaw': yaw
        })
        
        self.total_aruco_frames += 1
        
        # Обновляем статистику
        if self.report_data['aruco_landing']['performed']:
            self.report_data['aruco_landing']['max_offset'] = max(
                self.report_data['aruco_landing']['max_offset'],
                abs(offset_north), abs(offset_east)
            )
            self.report_data['aruco_landing']['final_offset_n'] = offset_north
            self.report_data['aruco_landing']['final_offset_e'] = offset_east
            self.report_data['aruco_landing']['total_frames'] = self.total_aruco_frames

    def connect_to_pixhawk(self):
        while True:
            try:
                self.master = mavutil.mavlink_connection(MAVLINK_PORT, baud=MAVLINK_BAUD)
                self.master.wait_heartbeat()
                logging.info(f"Подключение: {self.master.target_system}")
                break
            except Exception as e:
                logging.error(f"Ошибка подключения к Pixhawk: {e}. Повтор через 5 с...")
                self.send_ssh_message(f"Ошибка подключения к Pixhawk {e}. Повтор 5 с...") 
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        sys.stdout.flush()



    def load_model(self):
        try:
            self.model = ArmenianLetterNet()
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            self.model.eval()
            with open(LABELS_PATH, "r", encoding="utf-8") as f:
                self.labels = [line.strip() for line in f.readlines()]
            logging.info(" Модель загружена ")
        except Exception as e:
            logging.error(f" Ошибка загрузки модели: {e} ")
            raise

    def init_camera(self):
        # Небольшая пауза перед инициализацией, чтобы драйверы отпустили устройство
        time.sleep(0.5)
        try:
            if self.USE_PICAM:
                print("[INFO] Initializing CSI Camera (Picamera2)...")
                self.camera = Picamera2()
                config = self.camera.create_preview_configuration(
                     main={"size": CAMERA_RESOLUTION_PICAM, "format": "RGB888"},
                    controls={
                         "FrameRate": PROCESSING_FPS,
                         "AwbEnable": True,  
                         "AwbMode": 1,       
                         "AeEnable": True,   
                         "ExposureTime": 10000,  
                         "AnalogueGain": 1.0,    
                         "Brightness": 0.0,      
                         "Contrast": 1.0,        
                         "Saturation": 1.0,      
                         "Sharpness": 1.0,       
                    }
                )
                self.camera.configure(config)
                self.camera.start()
                logging.info("PiCamera запущена")

            elif self.USE_USB_CAM:
                print(f"[INFO] Initializing USB Camera (ID: {CAMERA_ID})...")
                self.camera = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)  
                if not self.camera.isOpened():
                    # Пробуем следующий индекс, если текущий не открылся
                    if CAMERA_ID == 0:
                         print("[WARN] Trying fallback USB ID 1...")
                         self.camera = cv2.VideoCapture(1, cv2.CAP_V4L2)
                    
                    if not self.camera.isOpened():
                        logging.error("Ошибка открытия камеры USB")
                        self.send_ssh_message("Ошибка открытия камеры USB")
                        self.camera = None
                        return

                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION_USBCAM[0])
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION_USBCAM[1])
                self.camera.set(cv2.CAP_PROP_FPS, PROCESSING_FPS)
                
                actual_settings = {
                    "Width": self.camera.get(cv2.CAP_PROP_FRAME_WIDTH),
                    "Height": self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    "FPS": self.camera.get(cv2.CAP_PROP_FPS),
                }
                logging.info(f"Настройки USB камеры: {actual_settings}")
                
        except Exception as e:
            logging.error(f"Ошибка запуска камеры: {str(e)}")
            self.send_ssh_message(f"Ошибка запуска камеры: {str(e)}")
            self.camera = None
    

    def process_mgm(self):
        print("[WARN] MGM mode skipped in offline test")
        return

    def get_current_altitude(self):
        msg = self.master.recv_match(type="GLOBAL_POSITION_INT", blocking=False)
        if msg:
            altitude = msg.relative_alt / 1e3 
            if not self.reached_200m and altitude >= 200:
                logging.info(f"Отметка в 200 метров достигнута, высота : {altitude}")
                self.reached_200m = True
            return altitude
        return 0.0

    def descent(self, target_alt):        
        pass
    
    def send_offset_command(self, offset_north, offset_east):
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component, # Теперь этот атрибут существует
            MAV_CMD_USER_1,  
            0,  
            offset_north,    
            offset_east,     
            0, 0, 0, 0, 0   
        )
        logging.info(f"Отправка команды смещения: Север={offset_north:.2f}м, Восток={offset_east:.2f}м")

    def release_camera(self):
        if self.camera:
            try:
                if self.USE_PICAM:
                    # Принудительная остановка потоков Picamera2
                    if hasattr(self.camera, '_stop'):
                        self.camera._stop()
                    self.camera.stop()
                    self.camera.close()
                    self.camera = None
                    logging.info("PiCamera остановлена")
                elif self.USE_USB_CAM:
                    self.camera.release()
                    self.camera = None
                    logging.info("USB-камера остановлена")
            except Exception as e:
                logging.warning(f"Ошибка при остановке камеры: {e}")
                self.camera = None # Все равно обнуляем ссылку
            
            # Пауза для освобождения ресурсов ОС
            time.sleep(0.5)

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
        return offset[0], offset[1]  
    
    def handle_mode_switch(self, new_mode):
        mode_names = {MODE_OFF: "OFF", MODE_LETTERS: "LETTERS", MODE_ARUCO: "ARUCO", MODE_MGM: "MGM"}
        new_mode_str = mode_names.get(new_mode, "UNKNOWN")
    
        if new_mode_str != self.current_mode_str:
            self.send_ssh_message(f" Переключение режима: {self.current_mode_str} -> {new_mode_str}")
            self.current_mode_str = new_mode_str

    def pixels_to_meters(self, pixel_x, pixel_y, ppm):
        return pixel_x / ppm, pixel_y / ppm
    
    def geo_to_3d_xyz(self, lat_deg, lon_deg, h_ellipsoid):
        a = 6378137.0  
        f = 1/298.257223563
        e2 = 2*f - f**2
        
        B = math.radians(lat_deg)
        L = math.radians(lon_deg)
        
        N = a / math.sqrt(1 - e2 * math.sin(B)**2)
        X = (N + h_ellipsoid) * math.cos(B) * math.cos(L)
        Y = (N + h_ellipsoid) * math.cos(B) * math.sin(L)
        Z = ((1 - e2) * N + h_ellipsoid) * math.sin(B)
        return X, Y, Z

    def xyz_to_geo_bowring(self, X, Y, Z):
        a = 6378137.0
        f = 1/298.257223563
        e2 = 2*f - f**2
        b = a * math.sqrt(1 - e2)
        ep2 = e2 / (1 - e2)
        
        Q = math.sqrt(X**2 + Y**2)
        if Q == 0:
            return (90.0 if Z > 0 else -90.0), 0.0, (abs(Z) - b)
            
        r = math.sqrt(Z**2 + Q**2 * (1 - e2))
        num = r**3 + b * ep2 * Z**2
        den = r**3 - b * e2 * (1 - e2) * Q**2
        
        B_rad = math.atan((Z / Q) * (num / den))
        L_rad = math.atan2(Y, X)
        
        N = a / math.sqrt(1 - e2 * math.sin(B_rad)**2)
        H = Q / math.cos(B_rad) - N
        
        return math.degrees(B_rad), math.degrees(L_rad), H

    def add_meters_to_coords_dinam(self, lat, lon, dx_m, dy_m, yaw_deg=0, alt=0):
        yaw_rad = math.radians(yaw_deg)
        north_m = dy_m * math.cos(yaw_rad) - dx_m * math.sin(yaw_rad)
        east_m = dy_m * math.sin(yaw_rad) + dx_m * math.cos(yaw_rad)
        
        h_ellipsoid = alt + ANOMALOUS_HEIGHT
        
        X, Y, Z = self.geo_to_3d_xyz(lat, lon, h_ellipsoid)
        
        lat_rad = math.radians(lat)
        lon_rad = math.radians(lon)
        
        dX = -math.sin(lat_rad)*math.cos(lon_rad)*north_m - math.sin(lon_rad)*east_m
        dY = -math.sin(lat_rad)*math.sin(lon_rad)*north_m + math.cos(lon_rad)*east_m
        dZ =  math.cos(lat_rad)*north_m
        
        new_lat, new_lon, _ = self.xyz_to_geo_bowring(X + dX, Y + dY, Z + dZ)
        
        return new_lat, new_lon

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
        
    def get_yaw(self):
        try:
            msg = self.master.recv_match(type='ATTITUDE', blocking=False)
            if msg:
                yaw = math.degrees(msg.yaw)
                return yaw
        except Exception as e:
            logging.error(f"Ошибка получения курса: {e}")
            return 0.0

    def get_coordinates(self):
        msg = self.master.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
        if msg:
            lat = msg.lat / 1e7
            lon = msg.lon / 1e7
            alt = msg.alt / 1e3
            return lat, lon, alt
        return 0.0, 0.0, 0.0
        
    def process_frame_dinam(self, frame):
        if len(frame.shape) == 2: 
             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        current_lat, current_lon, alt = self.get_coordinates() 
        yaw = self.get_yaw()

        if alt > MAX_ALTITUDE_M:
            return [], None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY) 
        thresh_color = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        alt_safe = max(alt, 1.0) 
        _scale_sq = (30.0 / alt_safe) ** 2
        dynamic_min_area = max(MIN_AREA * _scale_sq, 50)   
        dynamic_max_area = MAX_AREA * _scale_sq

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (dynamic_min_area < area < dynamic_max_area):
                continue

            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            if circularity < 0.4:  
                continue

            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_color = cv2.mean(gray, mask=mask)[0]
            if mean_color < 160:  
                continue

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.intp(box)
            center_x, center_y = int(rect[0][0]), int(rect[0][1])
            
            frame_center_x, frame_center_y = 720/2, 720/2
            dx_px = center_x - frame_center_x
            dy_px = -center_y + frame_center_y

            dynamic_ppm = PIXELS_PER_METER * (30.0 / alt_safe)
            raw_dx_m, raw_dy_m = self.pixels_to_meters(dx_px, dy_px, dynamic_ppm)
            
            dx_m = raw_dx_m
            dy_m = raw_dy_m + alt  
            
            new_lat, new_lon = self.add_meters_to_coords_dinam(current_lat, current_lon, dx_m, dy_m, yaw_deg=yaw, alt=alt)
            
            results.append({
                'center_px': (center_x, center_y),
                'center_m': (dx_m, dy_m),
                'raw_dx_m': raw_dx_m,
                'raw_dy_m': raw_dy_m,
                'coords': (new_lat, new_lon),
                'alt': alt,
                'box': box
            })
        
        return results, thresh_color

    def process_aruco(self):
        TARGET_ALTITUDE = 1.5  
        current_target_id = MARKER_42  
        current_length = LENGTH_42
        
        self.send_ssh_message(f"Старт поиска маркера {current_target_id} ")
        logging.info(f"Старт поиска маркера {current_target_id} ")
        
        last_save_time = time.time()
        last_command_time = 0
        command_interval = 0.25  

        while self.current_mode == MODE_ARUCO:
            current_time = time.time()
            if current_time - last_command_time < command_interval:
                time.sleep(0.01)
                continue
                
            last_command_time = current_time

            new_mode = self.check_mode()
            if new_mode != self.current_mode:
                self.send_ssh_message(" Выход из режима ARUCO")
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
                logging.info("Установлен режим GUIDED")
                self.send_ssh_message("Установлен режим GUIDED")
                self.drone_in_guided = True
            
            if self.land_command_sent:
                 break

            if ids is not None and current_target_id in ids.flatten():
                idx = list(ids.flatten()).index(current_target_id)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[idx]], current_length, camera_matrix, dist_coeffs
                )
                
                altitude = tvecs[0][0][2] 
                offset_north, offset_east = self.calculate_offset(tvecs[0]) 
                # Логирование для отчёта
                self.log_aruco_data(offset_north, offset_east, altitude)
                
                print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] Маркер {current_target_id} | "
                f"Высота: {altitude:.3f}m | "
                f"Размер: {current_length}m")
                
                if time.time() - last_save_time > 5:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(ARUCO_DIR, f"aruco_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    last_save_time = time.time()
                
                if current_target_id == MARKER_42 and altitude <= TARGET_ALTITUDE:
                    current_target_id = MARKER_41
                    current_length = LENGTH_41
                    self.report_data['aruco_landing']['switch_altitude'] = altitude
                    self.send_ssh_message(f"[{datetime.now().strftime('%H:%M:%S')}] ПЕРЕКЛЮЧЕНИЕ: "
                    f"Новый целевой маркер {current_target_id} (достигнута высота {altitude:.2f}m ≤ {TARGET_ALTITUDE}m)")
                    continue
    
                if altitude > self.min_altitude:
                    if abs(offset_north) < self.pos_tolerance and abs(offset_east) < self.pos_tolerance:
                        self.send_ssh_message("Малые смещения -> Снижение") 
                        self.descent(0.6)
                    else:
                        self.send_offset_command(offset_north, offset_east)
                    time.sleep(0.5)
                else:
                    if abs(offset_north) < self.pos_tolerance and abs(offset_east) < self.pos_tolerance and not self.land_command_sent:
                        self.master.set_mode("QLAND")
                        self.land_command_sent = True
                        self.report_data['aruco_landing']['performed'] = True
                        self.send_ssh_message("Команда посадки отправлена, режим QLAND")
                    else:
                        self.send_offset_command(offset_north, offset_east)

            time.sleep(0.01)
        
        self.land_command_sent = False 
        self.drone_in_guided = False

    def process_letters_dinam(self):
        if not self.model_loaded :
            self.load_model()
            self.model_loaded = True

        self.send_ssh_message("=== Активирован режим распознавания букв ===")
        self.last_processing_time = time.time()
        self.last_save_time = time.time()
        
        self.last_inference_time = 0
        self.inference_interval = 0.1  
        
        try:
            while self.current_mode == MODE_LETTERS:
                new_mode = self.check_mode()
                if new_mode != self.current_mode:
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
                    self.send_ssh_message("Кадр не получен")
                    continue
                
                results, thresh_color = self.process_frame_dinam(frame)
                letter_detected = False

                for result in results:
                    cv2.drawContours(frame, [result['box']], 0, (0, 255, 0), 2)
                    cv2.circle(frame, result['center_px'], 5, (0, 0, 255), -1)
                    
                    width, height = map(int, cv2.minAreaRect(result['box'])[1])
                    if width > 40 and height > 40:  
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
                                        label_parts = label.strip().split()
                                        letter_char  = label_parts[1] if len(label_parts) >= 2 else label
                                        letter_idx   = label_parts[0] if len(label_parts) >= 1 else "?"

                                        self.letter_stats[label]['confidences'].append(conf.item())
                                        self.letter_stats[label]['coords'].append(result['coords'])
                                        self.letter_stats[label]['timestamps'].append(current_time)
                                    
                                        self.processing_active = True
                                        text = f"{letter_char} ({conf.item():.2f})"
                                        cv2.putText(frame, text, result['center_px'], 
                                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                        
                                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                        log_msg = (f"[{timestamp_str}] LETTER: {letter_char} | "
                                                   f"CONF: {conf.item():.2f} | "
                                                   f"LAT: {result['coords'][0]:.6f} | "
                                                   f"LON: {result['coords'][1]:.6f} | "
                                                   f"ALT: {result['alt']:.2f} | "
                                                   f"OFFSET_X: {result['raw_dx_m']:.2f} | "
                                                   f"OFFSET_Y: {result['raw_dy_m']:.2f}")
                                        self.send_ssh_message(log_msg)
                                        logging.info(log_msg)

                                        if current_time - self.last_save_time > self.SAVE_INTERVAL:
                                            coords_str = f"{result['coords'][0]:.6f}_{result['coords'][1]:.6f}"
                                            filename = os.path.join(
                                                LETTERS_DIR,
                                                f"armenian_letter_{coords_str}_{letter_idx}.jpg"
                                            )
                                            cv2.imwrite(filename, frame)
                                            self.last_save_time = current_time
                                            
                        except Exception as e:
                            logging.error(f"Ошибка обработки буквы: {e}")

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
                            
                            logging.info(log_msg)
                            self._save_to_csv(best_letter, avg_lat, avg_lon)
                            self.total_letters_detected += 1
                            
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
            self.send_ssh_message(f"ОШИБКА: {str(e)}")
        
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
                
                logging.info(log_msg)
                self._save_to_csv(best_letter, avg_lat, avg_lon)
                self.processing_sessions.append({
                    'letter': best_letter,
                    'count': len(best_data['confidences']),
                    'avg_confidence': avg_conf,
                    'avg_coords': (avg_lat, avg_lon),
                    'timestamp': last_time
                })
        
        if self.processing_sessions:
            log_msg = "\n=== Сводка всех сеансов обработки ==="
            for i, session in enumerate(self.processing_sessions, 1):
                log_msg += f"\n{i}. Буква: {session['letter']}"
                log_msg += f"\n   Обнаружений: {session['count']}"
                log_msg += f"\n   Уверенность: {session['avg_confidence']:.2f}"
                log_msg += f"\n   Координаты: {session['avg_coords']}"
                log_msg += f"\n   Время: {session['timestamp']}"
                log_msg += "\n-----------------------------"
            logging.info(log_msg)

    def _save_to_csv(self, letter_label, avg_lat, avg_lon):
        try:
            parts = letter_label.strip().split()
            letter_index = int(parts[0]) if len(parts) >= 1 else -1

            lat_e7 = int(round(avg_lat * 1e7))
            lon_e7 = int(round(avg_lon * 1e7))

            with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                f.write(f"{letter_index},{lat_e7},{lon_e7}\n")

            letter_char = parts[1] if len(parts) >= 2 else "?"
            log_msg = f"CSV записано: {letter_index},{lat_e7},{lon_e7} (буква: {letter_char})"
            self.send_ssh_message(log_msg)
            logging.info(log_msg)

        except Exception as e:
            logging.error(f"Ошибка записи CSV: {e}")

    def check_mode(self):
        wp = self.get_current_waypoint()
        rc_val = self.get_rc_value()        
        new_mode = None
                
        if rc_val is not None and rc_val != self.last_rc_value:
            new_mode = self.rc_to_mode(rc_val)
            self.last_rc_value = rc_val
            logging.info(f"RC: {rc_val} -> {self.mode_name(new_mode)}")
            self.send_ssh_message(f"RC: {rc_val} -> {self.mode_name(new_mode)}") 
            
        elif wp is not None and wp != self.last_wp_value and wp in self.wp_modes:
            new_mode = self.wp_modes[wp]
            self.last_wp_value = wp
            logging.info(f"WP: {wp} -> {self.mode_name(new_mode)}")
            self.send_ssh_message(f"WP: {wp} -> {self.mode_name(new_mode)}") 
            
        if new_mode is not None and new_mode != self.current_mode:
            self.current_mode = new_mode
            self.handle_mode_switch(new_mode)
        
        return self.current_mode

    def mode_name(self, mode):
        return ["OFF", "LETTERS", "ARUCO", "MGM"][mode]

    def run(self):
        try:
            while True:
                try:  
                    self.current_mode = self.check_mode()
                    
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
                        
                        if self.camera:
                            self.process_letters_dinam()
                            self.process_LETTERS = True
                            self.process_ARUCO = False
                            self.process_MGM = False
                            self.process_OFF = False
                        else:
                            self.send_ssh_message("Ошибка инициализации камеры для LETTERS")
                            self.current_mode = MODE_OFF

                    elif self.current_mode == MODE_ARUCO and not self.process_ARUCO:
                        self.USE_PICAM = False
                        self.USE_USB_CAM = True     
                        self.init_camera()
                        
                        if self.camera and (self.USE_USB_CAM and self.camera.isOpened()):
                            self.process_aruco()
                            self.process_ARUCO = True
                            self.process_LETTERS = False
                            self.process_MGM = False
                            self.process_OFF = False
                        else:
                            self.send_ssh_message("Ошибка инициализации камеры для ARUCO")
                            self.current_mode = MODE_OFF
                    
                    elif self.current_mode == MODE_MGM and not self.process_MGM:
                        self.USE_PICAM = False
                        self.USE_USB_CAM = True  
                        self.process_mgm()
                        self.process_ARUCO = False
                        self.process_LETTERS = False                        
                        self.process_MGM = True
                        self.process_OFF = False
                        self.current_mode = MODE_OFF
                        self.handle_mode_switch(MODE_OFF)                       

                    time.sleep(0.01)
                    
                except Exception as inner_e:
                    logging.error(f"Ошибка в основном цикле: {inner_e}")
                    self.send_ssh_message(f"Внутренняя ошибка: {inner_e}")
                    time.sleep(1)
                
        except KeyboardInterrupt:
            self.send_ssh_message("--- Завершение работы ---")
            self.release_camera()
            self.generate_final_report()
        except Exception as e:
            self.send_ssh_message(f"КРИТИЧЕСКАЯ ОШИБКА: {str(e)}")
            logging.critical(f"Необработанное исключение: {e}")

    def generate_final_report(self):
        """Генерирует финальный отчёт при завершении работы"""
        try:
            self.report_data['mission_end'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.report_data['system_stats']['pixhawk_connected'] = self.is_pixhawk_connected
            self.report_data['system_stats']['model_loaded'] = self.model_loaded
            
            report_dir = "/home/pi/Desktop/reports"
            os.makedirs(report_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"mission_{timestamp}"
            
            self.send_ssh_message("Генерация отчёта...")
            
            # 1. 3D График
            if self.landing_offsets_log:
                graph_result, error = self.generate_3d_trajectory_graph(report_dir, filename_base)
                if graph_result:
                    self.send_ssh_message(f"✓ 3D-график: {graph_result['pdf']}")
                else:
                    self.send_ssh_message(f"⚠ Ошибка графика: {error}")
            
            # 2. GIF (опционально)
            if len(self.landing_offsets_log) >= 10:
                gif_path, error = self.generate_landing_gif(report_dir, filename_base)
                if gif_path:
                    self.send_ssh_message(f"✓ GIF-анимация: {gif_path}")
            
            # 3. PDF отчёт
            pdf_path = self.generate_pdf_report(report_dir, filename_base)
            if pdf_path:
                self.send_ssh_message(f"✓ PDF-отчёт: {pdf_path}")
                self.send_ssh_message(f"Все файлы в: {report_dir}/")
            else:
                self.send_ssh_message("⚠ Не удалось сгенерировать PDF")
            
            # 4. Сохранение данных в JSON (для последующей обработки)
            json_path = f"{report_dir}/{filename_base}_data.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'report_data': self.report_data,
                    'landing_log': self.landing_offsets_log,
                    'processing_sessions': self.processing_sessions
                }, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            logging.error(f"Ошибка генерации финального отчёта: {e}")
            self.send_ssh_message(f"⚠ Ошибка отчёта: {e}")

if __name__ == '__main__':
    controller = DroneController()
    controller.run()
