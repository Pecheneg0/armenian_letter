import cv2
import numpy as np

# Инициализация камеры
CAMERA_ID = 1
CAMERA_RESOLUTION = (720, 720)
PROCESSING_FPS = 30

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
if not cap.isOpened():
    print("Ошибка открытия камеры USB")
    exit()

# Установка основных параметров
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
cap.set(cv2.CAP_PROP_FPS, PROCESSING_FPS)

# Создаем окно для управления параметрами
cv2.namedWindow("USB Camera Tuning", cv2.WINDOW_NORMAL)
cv2.resizeWindow("USB Camera Tuning", 800, 600)

# Текущие значения параметров
params = {
    "exposure": 100,
    "gain": 100,
    "brightness": 100,
    "contrast": 45,
    "saturation": 67,
    "sharpness": 80,
    "wb_temperature": 4800,
    "backlight": 1,
    "auto_exposure": 1,
    "auto_wb": 1
}

# Функции-обработчики для trackbars
def update_exposure(val):
    cap.set(cv2.CAP_PROP_EXPOSURE, val)
    params["exposure"] = val

def update_gain(val):
    cap.set(cv2.CAP_PROP_GAIN, val)
    params["gain"] = val

def update_brightness(val):
    cap.set(cv2.CAP_PROP_BRIGHTNESS, val)
    params["brightness"] = val

def update_contrast(val):
    cap.set(cv2.CAP_PROP_CONTRAST, val)
    params["contrast"] = val

def update_saturation(val):
    cap.set(cv2.CAP_PROP_SATURATION, val)
    params["saturation"] = val

def update_sharpness(val):
    cap.set(cv2.CAP_PROP_SHARPNESS, val)
    params["sharpness"] = val

def update_wb_temperature(val):
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, val*100)
    params["wb_temperature"] = val*100

def update_backlight(val):
    cap.set(cv2.CAP_PROP_BACKLIGHT, val)
    params["backlight"] = val

def toggle_auto_exposure(val):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, val)
    params["auto_exposure"] = val

def toggle_auto_wb(val):
    cap.set(cv2.CAP_PROP_AUTO_WB, val)
    params["auto_wb"] = val

# Создаем trackbars
cv2.createTrackbar("Exposure", "USB Camera Tuning", 100, 100, update_exposure)
cv2.createTrackbar("Gain", "USB Camera Tuning", 100, 100, update_gain)
cv2.createTrackbar("Brightness", "USB Camera Tuning", 100, 100, update_brightness)
cv2.createTrackbar("Contrast", "USB Camera Tuning", 45, 100, update_contrast)
cv2.createTrackbar("Saturation", "USB Camera Tuning", 67, 100, update_saturation)
cv2.createTrackbar("Sharpness", "USB Camera Tuning", 80, 100, update_sharpness)
cv2.createTrackbar("WB Temp (x100)", "USB Camera Tuning", 48, 70, update_wb_temperature)
cv2.createTrackbar("Backlight", "USB Camera Tuning", 1, 1, update_backlight)
cv2.createTrackbar("Auto Exposure", "USB Camera Tuning",1 , 1, toggle_auto_exposure)
cv2.createTrackbar("Auto WB", "USB Camera Tuning", 1, 1, toggle_auto_wb)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра")
            break

        # Отображаем текущие параметры на изображении
        info_text = [
            f"Exposure: {params['exposure']}",
            f"Gain: {params['gain']}",
            f"Brightness: {params['brightness']}",
            f"Contrast: {params['contrast']}",
            f"Saturation: {params['saturation']}",
            f"Sharpness: {params['sharpness']}",
            f"WB Temp: {params['wb_temperature']}K",
            f"Backlight: {params['backlight']}",
            f"Auto Exposure: {'ON' if params['auto_exposure'] else 'OFF'}",
            f"Auto WB: {'ON' if params['auto_wb'] else 'OFF'}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        # Показываем изображение
        cv2.imshow("USB Camera Tuning", frame)

        # Выход по 'q' или ESC
        key = cv2.waitKey(1)
        if key in (27, ord('q')):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\nFinal parameters:")
    for param, value in params.items():
        print(f"{param}: {value}")
