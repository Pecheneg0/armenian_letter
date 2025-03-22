import cv2
import numpy as np
import torch
import time
import math
from picamera2 import Picamera2
from PIL import Image
from torchvision import transforms
from modeln import ArmenianLetterNet

# 1. Загрузка и оптимизация модели (опционально, можно использовать TorchScript)
model = ArmenianLetterNet()
model.load_state_dict(torch.load("armenian_letters_model_improved.pth", map_location="cpu"))
model.eval()
# Пример оптимизации: 
# scripted_model = torch.jit.trace(model, torch.randn(1, 1, 32, 32))
# Используйте scripted_model вместо model, если потребуется

# Загрузка меток для классификации
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# 2. Параметры системы и камеры
HEIGHT = 35                      # Высота полёта (в метрах)
CAMERA_ANGLE = 45                # Угол наклона камеры (в градусах)
FOV_H = 62.2                     # Горизонтальный угол обзора камеры
FOV_V = 48.8                     # Вертикальный угол обзора камеры
RESOLUTION = (3280, 2464)        # Разрешение камеры
FOCAL_LENGTH = HEIGHT * math.tan(math.radians(FOV_V / 2))

# 3. Инициализация камеры
camera = Picamera2()
config = camera.create_preview_configuration(main={"size": RESOLUTION, "format": "RGB888"})
camera.configure(config)
camera.start()

# 4. Пороговые значения для фильтрации предсказаний
CONFIDENCE_THRESHOLD_LOW = 0.75
CONFIDENCE_THRESHOLD_HIGH = 0.85
frame_buffer = []

# 5. Функция предобработки изображения
def preprocess_image(image):
    # Преобразуем в градации серого
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Применяем гауссово размытие для устранения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Усиливаем контраст (опционально можно использовать CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(blurred)
    # Применяем адаптивное порогование
    binary = cv2.adaptiveThreshold(contrast_enhanced, 255, 
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    return binary

# 6. Поиск квадратного/прямоугольного контура
def find_square_contour(image):
    # Получаем предварительное бинарное изображение
    binary = preprocess_image(image)
    # Используем Canny для выделения краёв
    edges = cv2.Canny(binary, 50, 150)
    # Находим контуры
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        # Аппроксимируем контур
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > 1000:
                return approx, binary
    return None, binary

# 7. Функция для выравнивания и обрезки изображения
def align_and_crop(image, contour):
    points = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # Сортировка точек по сумме координат
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    
    # Сортировка по разности координат
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]
    rect[3] = points[np.argmax(diff)]
    
    # Вычисляем размеры целевого прямоугольника
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

# 8. Преобразование изображения для модели (PIL, ресайз, нормализация)
def process_image_for_model(pil_image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    tensor = transform(pil_image).unsqueeze(0)
    return tensor

# 9. Функция предсказания буквы
def predict_letter(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)  # или scripted_model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    return predicted_class.item(), confidence.item()

# 10. Функция для перевода координат пикселей в реальные координаты
def pixel_to_coordinates(px, py, drone_lat, drone_lon):
    cx, cy = RESOLUTION[0] // 2, RESOLUTION[1] // 2
    meters_per_pixel = (2 * HEIGHT * math.tan(math.radians(FOV_H / 2))) / RESOLUTION[0]
    dx = (px - cx) * meters_per_pixel
    dy = (py - cy) * meters_per_pixel * math.cos(math.radians(CAMERA_ANGLE))
    lat_offset = dy / 111320  # 1 градус ~ 111320 метров
    lon_offset = dx / (111320 * math.cos(math.radians(drone_lat)))
    return drone_lat + lat_offset, drone_lon + lon_offset

# 11. Функция записи координат и предсказанной буквы на SD-карту
def save_coordinates(letter, lat, lon, confidence):
    with open("/home/pi/letters_coordinates.txt", "a") as file:
        file.write(f"{letter},{lat},{lon},{confidence:.2f}\n")

# 12. Основной цикл сканирования с оптимизациями
def scan_letters():
    global frame_buffer
    while True:
        frame = camera.capture_array("main")
        
        # Находим контур и получаем бинарное изображение
        square_contour, binary_image = find_square_contour(frame)
        if square_contour is None:
            print("⚠️ Прямоугольный контур не найден.")
            time.sleep(0.5)
            continue

        # Производим обрезку (используем исходное изображение для лучшей детализации)
        aligned_image = align_and_crop(frame, square_contour)
        
        # Преобразуем результат в PIL Image
        pil_image = Image.fromarray(aligned_image)
        image_tensor = process_image_for_model(pil_image)
        
        # Предсказываем букву
        predicted_index, confidence_value = predict_letter(image_tensor)
        letter = labels[predicted_index]
        
        # Фильтрация по уверенности
        if confidence_value < CONFIDENCE_THRESHOLD_LOW:
            print(f"⚠️ Слабое предсказание ({confidence_value:.2f}) - игнорируем")
            time.sleep(0.5)
            continue
        
        # Если уверенность на границе, аккумулируем данные
        if CONFIDENCE_THRESHOLD_LOW <= confidence_value < CONFIDENCE_THRESHOLD_HIGH:
            frame_buffer.append((letter, confidence_value))
            if len(frame_buffer) >= 5:
                avg_confidence = sum(x[1] for x in frame_buffer) / len(frame_buffer)
                most_common_letter = max(set(x[0] for x in frame_buffer), key=lambda x: [y[0] for y in frame_buffer].count(x))
                frame_buffer.clear()
                if avg_confidence >= CONFIDENCE_THRESHOLD_HIGH:
                    letter = most_common_letter
                    confidence_value = avg_confidence
                else:
                    print(f"⚠️ Неопределённый результат ({avg_confidence:.2f}) - пропускаем")
                    continue
        
        # Если уверенность выше порога, вычисляем координаты и записываем
        if confidence_value >= CONFIDENCE_THRESHOLD_HIGH:
            # Здесь вместо заглушки следует использовать реальные данные GPS
            drone_lat, drone_lon = 40.1792, 44.4991
            # Предполагаем, что буква находится в центре кадра обрезанного изображения
            letter_lat, letter_lon = pixel_to_coordinates(aligned_image.shape[1]//2, aligned_image.shape[0]//2, drone_lat, drone_lon)
            save_coordinates(letter, letter_lat, letter_lon, confidence_value)
            print(f"✅ Найдена буква {letter} ({confidence_value:.2f}) на координатах {letter_lat:.6f}, {letter_lon:.6f}")

        # Небольшая задержка для энергосбережения
        time.sleep(0.5)

if __name__ == "__main__":
    try:
        scan_letters()
    except KeyboardInterrupt:
        print("Прерывание сканирования. Остановка камеры...")
        camera.stop()
        cv2.destroyAllWindows()
