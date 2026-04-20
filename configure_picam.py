from picamera2 import Picamera2
import cv2
import numpy as np

# Инициализация камеры
picam2 = Picamera2()
config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"},
    controls={
        "FrameRate": 30,
        "AwbEnable": False,
        "AeEnable": False,
        "ExposureTime": 10000,
        "AnalogueGain": 1.0,
        "Brightness": 0.1,
        "Contrast": 1.2,
        "Saturation": 0.8,
        "Sharpness": 1.5
    }
)
picam2.configure(config)
picam2.start()

# Создаем окна для управления параметрами
cv2.namedWindow("Camera Tuning", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Camera Tuning", 800, 600)

# Текущие значения параметров
params = {
    "exposure": 4900,
    "gain": 1.0,
    "brightness": 0.1,
    "contrast": 1.1,
    "saturation": 0,
    "sharpness": 0,
    "awb_mode": 1,
    "ae_enabled": 1,
    "awb_enabled": 1
}

# Функции-обработчики для trackbars
def update_exposure(val):
    picam2.set_controls({"ExposureTime": val})
    params["exposure"] = val

def update_gain(val):
    picam2.set_controls({"AnalogueGain": val/10})
    params["gain"] = val/10

def update_brightness(val):
    picam2.set_controls({"Brightness": val/100})
    params["brightness"] = val/100

def update_contrast(val):
    picam2.set_controls({"Contrast": val/10})
    params["contrast"] = val/10

def update_saturation(val):
    picam2.set_controls({"Saturation": val/10})
    params["saturation"] = val/10

def update_sharpness(val):
    picam2.set_controls({"Sharpness": val/10})
    params["sharpness"] = val/10

def toggle_awb(val):
    picam2.set_controls({"AwbEnable": bool(val)})
    params["awb_enabled"] = val

def toggle_ae(val):
    picam2.set_controls({"AeEnable": bool(val)})
    params["ae_enabled"] = val

def set_awb_mode(val):
    modes = {
        0: 0,   # Auto
        1: 1,   # Daylight
        2: 2,   # Cloudy
        3: 3,   # Shade
        4: 4,   # Tungsten
        5: 5,   # Fluorescent
        6: 6    # Indoor
    }
    picam2.set_controls({"AwbMode": modes[val]})
    params["awb_mode"] = val

# Создаем trackbars
#cv2.createTrackbar("Exposure (us)", "Camera Tuning", 4900, 30000, update_exposure)
#cv2.createTrackbar("Gain (x10)", "Camera Tuning", 36, 50, update_gain)
cv2.createTrackbar("Brightness (x100)", "Camera Tuning", 0, 50, update_brightness)
cv2.createTrackbar("Contrast (x10)", "Camera Tuning", 10, 30, update_contrast)
cv2.createTrackbar("Saturation (x10)", "Camera Tuning", 4, 20, update_saturation)
cv2.createTrackbar("Sharpness (x10)", "Camera Tuning", 0, 50, update_sharpness)
cv2.createTrackbar("AWB Enable", "Camera Tuning", 1, 1, toggle_awb)
cv2.createTrackbar("AE Enable", "Camera Tuning", 1, 1, toggle_ae)
cv2.createTrackbar("AWB Mode", "Camera Tuning", 1, 6, set_awb_mode)

try:
    while True:
        # Получаем кадр
        frame = picam2.capture_array()
        
        # Конвертируем RGB в BGR для OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Отображаем текущие параметры на изображении
        info_text = [
            #f"Exposure: {params['exposure']} us",
            #f"Gain: {params['gain']:.1f}",
            f"Brightness: {params['brightness']:.2f}",
            f"Contrast: {params['contrast']:.1f}",
            f"Saturation: {params['saturation']:.1f}",
            f"Sharpness: {params['sharpness']:.1f}",
            f"AWB: {'ON' if params['awb_enabled'] else 'OFF'}",
            f"AE: {'ON' if params['ae_enabled'] else 'OFF'}",
            f"AWB Mode: {params['awb_mode']}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(frame_bgr, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        # Показываем изображение
        cv2.imshow("Camera Tuning", frame_bgr)
        
        # Выход по 'q' или ESC
        key = cv2.waitKey(1)
        if key in (27, ord('q')):
            break

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    print("\nFinal parameters:")
    for param, value in params.items():
        print(f"{param}: {value}")
