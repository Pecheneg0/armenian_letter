import cv2
import torch
import numpy as np
import time
from picamera2 import Picamera2
import math
from PIL import Image
from torchvision import transforms
import pymavlink.mavutil as mavutil
import cv2.aruco as aruco

# 🔹 Загрузка модели
from modeln import ArmenianLetterNet

# Загрузка меток
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

model = ArmenianLetterNet()
model.load_state_dict(torch.load("armenian_letters_model_improved.pth", map_location="cpu"))
model.eval()

# 🔹 Настройка камеры
RESOLUTION = (3280, 2464)  # Разрешение камеры
camera = Picamera2()
config = camera.create_preview_configuration(main={"size": RESOLUTION, "format": "RGB888"})
camera.configure(config)
camera.start()

# 🔹 Подключение к MAVLink
mavlink_connection = mavutil.mavlink_connection("udp:127.0.0.1:14550")
mode = "letter"  # По умолчанию режим поиска букв

# 🔹 Функция обработки изображения для модели
def process_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

# 🔹 Функция предсказания буквы
def predict_letter(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    return labels[predicted_class.item()], confidence.item()

# 🔹 Функция нахождения центра контура
def get_contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        return cx, cy
    return None

# 🔹 Функция обработки ArUco-маркеров
def detect_aruco(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None:
        for corner, marker_id in zip(corners, ids.flatten()):
            int_corners = np.int0(corner)
            cv2.polylines(frame, [int_corners], True, (0, 255, 0), 2)
            center_x = int(np.mean(corner[0][:, 0]))
            center_y = int(np.mean(corner[0][:, 1]))
            cv2.putText(frame, f"ID: {marker_id}", (center_x, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# 🔹 Основной цикл обработки
while True:
    # Проверяем входящие команды MAVLink
    msg = mavlink_connection.recv_match(blocking=False)
    if msg and msg.get_type() == "COMMAND_LONG":
        if msg.command == 300:  # Команда для переключения режимов
            mode = "marker" if mode == "letter" else "letter"
            print(f"Переключение режима: {mode}")
    
    frame = camera.capture_array("main")
    
    if mode == "letter":
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        h, w, _ = frame.shape
        center_x, center_y = w // 2, h // 2
        
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            if len(approx) == 4 and cv2.contourArea(contour) > 1000:
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                letter_center = get_contour_center(contour)
                
                if letter_center:
                    cx, cy = letter_center
                    offset_x = cx - center_x
                    offset_y = cy - center_y
                    offset_x_percent = (offset_x / center_x) * 100
                    offset_y_percent = (offset_y / center_y) * 100
                    
                    cropped = frame[cy-16:cy+16, cx-16:cx+16]
                    cropped_pil = Image.fromarray(cropped) if cropped.shape[0] > 0 and cropped.shape[1] > 0 else None
                    
                    if cropped_pil:
                        image_tensor = process_image(cropped_pil)
                        letter, confidence = predict_letter(image_tensor)
                        
                        cv2.putText(frame, f"{letter} ({confidence:.2f})", (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(frame, f"X: {offset_x} px ({offset_x_percent:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Y: {offset_y} px ({offset_y_percent:.2f}%)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    elif mode == "marker":
        detect_aruco(frame)
    
    cv2.imshow("Video Stream", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    time.sleep(2.5)

cv2.destroyAllWindows()
camera.stop()
