# scan_letters_rp4.py
import cv2
import torch
import numpy as np
import time
from picamera2 import Picamera2
import math
import os

# 🔹 Загрузка модели
from modeln import ArmenianLetterNet

# Загрузка меток
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

model = ArmenianLetterNet()
model.load_state_dict(torch.load("armenian_letters_model_improved.pth", map_location="cpu"))
model.eval()

# 🔹 Параметры системы
HEIGHT = 35  # Высота полёта (30-40 м)
CAMERA_ANGLE = 45  # Угол наклона камеры (градусы)
FOV_H = 62.2  # Горизонтальный угол обзора камеры
FOV_V = 48.8  # Вертикальный угол обзора камеры
RESOLUTION = (3280, 2464)  # Разрешение камеры

# 🔹 Фокусное расстояние камеры (в метрах)
FOCAL_LENGTH = HEIGHT * math.tan(math.radians(FOV_V / 2))

# 🔹 Настройка камеры
camera = Picamera2()
config = camera.create_preview_configuration(main={"size": RESOLUTION, "format": "RGB888"})
camera.configure(config)
camera.start()

# 🔹 Фильтр предсказаний
confidence_threshold_low = 0.75   # 📌 Минимальная уверенность для учёта предсказания
confidence_threshold_high = 0.85  # 📌 Уверенность, при которой сразу фиксируем результат
frame_buffer = []  # Буфер предсказаний для усреднения

# 🔹 Функция для обработки изображения
def process_image(image):
    """ Преобразует изображение в тензор для модели """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Преобразуем в градации серого
        transforms.Resize((32, 32)),  # Изменяем размер до 32x32
        transforms.ToTensor(),  # Преобразуем в тензор
        transforms.Normalize((0.5,), (0.5,))  # Нормализуем
    ])
    image_tensor = transform(image).unsqueeze(0)  # Добавляем batch dimension
    return image_tensor

# 🔹 Функция для предсказания буквы
def predict_letter(image_tensor):
    """ Предсказывает букву на изображении """
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), confidence.item()

# 🔹 Функция для нахождения контура прямоугольника
def find_square_contour(image):
    """ Находит контур белого прямоугольника с черной буквой внутри """
    # Преобразуем изображение в монохромное (черно-белое)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Применяем пороговую обработку для выделения белого прямоугольника
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)  # Порог для выделения белого
    
    # Находим контуры на бинарном изображении
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # Если контур имеет 4 вершины, это может быть прямоугольник
        if len(approx) == 4:
            # Проверяем, что площадь контура достаточно большая
            area = cv2.contourArea(contour)
            if area > 1000:  # Минимальная площадь для исключения мелких объектов
                return approx, binary  # Возвращаем контур и бинарное изображение
    return None, binary  # Если контур не найден, возвращаем только бинарное изображение

# 🔹 Функция для обрезки изображения до прямоугольника
def crop_to_contour(image, contour):
    """ Обрезает изображение так, чтобы остался только прямоугольник """
    x, y, w, h = cv2.boundingRect(contour)
    cropped_image = image[y:y+h, x:x+w]
    return cropped_image

# 🔹 Функция перевода пикселей в координаты
def pixel_to_coordinates(px, py, drone_lat, drone_lon):
    """ Переводит пиксели буквы в реальные координаты """
    cx, cy = RESOLUTION[0] // 2, RESOLUTION[1] // 2  # Центр изображения

    # 🔹 Метры на пиксель
    meters_per_pixel = (2 * HEIGHT * math.tan(math.radians(FOV_H / 2))) / RESOLUTION[0]

    # 🔹 Смещение буквы относительно центра камеры
    dx = (px - cx) * meters_per_pixel
    dy = (py - cy) * meters_per_pixel * math.cos(math.radians(CAMERA_ANGLE))

    # 🔹 Вычисление реальных координат
    lat_offset = dy / 111320  # 1 градус = ~111.32 км
    lon_offset = dx / (111320 * math.cos(math.radians(drone_lat)))

    return drone_lat + lat_offset, drone_lon + lon_offset

# 🔹 Функция сохранения координат
def save_coordinates(letter, lat, lon, confidence):
    """ Записывает найденные буквы и координаты на SD-карту """
    with open("/home/pi/letters_coordinates.txt", "a") as file:
        file.write(f"{letter},{lat},{lon},{confidence:.2f}\n")

# 🔹 Основной цикл
def scan_letters():
    """ Сканирование букв и запись координат """
    global frame_buffer

    while True:
        frame = camera.capture_array("main")  # 📷 Фотографируем
        
        # Находим контур прямоугольника и бинарное изображение
        square_contour, binary_image = find_square_contour(frame)
        if square_contour is None:
            print("⚠️ Ошибка: Не удалось найти прямоугольный контур на изображении.")
            continue  # Пропускаем кадр, если не нашли прямоугольник

        # Обрезаем изображение так, чтобы остался только прямоугольник
        cropped_image = crop_to_contour(binary_image, square_contour)

        # Преобразуем обрезанное изображение в PIL Image для дальнейшей обработки
        cropped_image_pil = Image.fromarray(cropped_image)

        # Обрабатываем изображение и предсказываем букву
        image_tensor = process_image(cropped_image_pil)
        predicted_index, confidence_value = predict_letter(image_tensor)
        letter = labels[predicted_index]  # Преобразуем индекс в букву

        # 🔹 Фильтрация по уверенности
        if confidence_value < confidence_threshold_low:
            print(f"⚠️ Слабое предсказание ({confidence_value:.2f}) - игнорируем")
            continue  # Пропускаем слабые предсказания

        # 🔹 Если уверенность 75-85%, проверяем несколько кадров
        if confidence_threshold_low <= confidence_value < confidence_threshold_high:
            frame_buffer.append((letter, confidence_value))

            if len(frame_buffer) >= 5:  # Анализируем 5 кадров подряд
                avg_confidence = sum([c[1] for c in frame_buffer]) / len(frame_buffer)
                most_common_letter = max(set([c[0] for c in frame_buffer]), key=[c[0] for c in frame_buffer].count)
                
                if avg_confidence >= confidence_threshold_high:
                    print(f"✅ Надёжное предсказание ({most_common_letter}, {avg_confidence:.2f})")
                    frame_buffer.clear()  # Очищаем буфер
                else:
                    print(f"⚠️ Неопределённый результат ({avg_confidence:.2f}) - пропускаем")
                    frame_buffer.clear()
                    continue  # Пропускаем ненадёжные предсказания

        # 🔹 Если уверенность выше 85%, записываем немедленно
        if confidence_value >= confidence_threshold_high:
            drone_lat, drone_lon = 40.1792, 44.4991  # 🔹 Заглушка, заменить на GPS!
            letter_lat, letter_lon = pixel_to_coordinates(RESOLUTION[0]//2, RESOLUTION[1]//2, drone_lat, drone_lon)

            save_coordinates(letter, letter_lat, letter_lon, confidence_value)
            print(f"✅ Найдена буква {letter} ({confidence_value:.2f}) на координатах {letter_lat}, {letter_lon}")

        # 🔹 Энергосбережение – пауза между сканированием
        time.sleep(2)  

# 🚀 Запускаем сканирование
scan_letters()