import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# Пути к модели и меткам
MODEL_PATH = "/home/pi/model.tflite"
LABELS_PATH = "/home/pi/labels.txt"

# Загрузка меток
with open(LABELS_PATH, "r") as f:
    labels = [line.strip() for line in f.readlines()]

# Инициализация интерпретатора TensorFlow Lite
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Получение входных и выходных тензоров
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Функция для захвата изображения с камеры
def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Камера не подключена.")
        return None
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Ошибка: Не удалось захватить изображение.")
        return None
    return frame

# Функция для поиска квадратного контура
def find_square_contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # Если контур имеет 4 угла
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.9 < aspect_ratio < 1.1:  # Проверка на квадратность
                return approx
    return None

# Функция для выравнивания контура
def align_contour(image, contour):
    # Получаем координаты углов контура
    points = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")

    # Вычисляем центр контура
    center = np.mean(points, axis=0)

    # Сортируем точки по углам
    diff = points - center
    angles = np.arctan2(diff[:, 1], diff[:, 0])
    sorted_points = points[np.argsort(angles)]

    # Определяем новые координаты для выравнивания
    (tl, tr, br, bl) = sorted_points
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Применяем перспективное преобразование
    matrix = cv2.getPerspectiveTransform(sorted_points, dst)
    aligned = cv2.warpPerspective(image, matrix, (int(width), int(height)))
    return aligned

# Функция для классификации изображения
def classify_image(image):
    input_shape = input_details[0]['shape'][1:3]
    resized_image = cv2.resize(image, input_shape)
    input_data = np.expand_dims(resized_image, axis=0).astype(np.float32)
    input_data = input_data / 255.0  # Нормализация

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    return labels[predicted_class]

# Основной код
def main():
    while True:
        command = input("Нажмите Enter для захвата изображения или 'q' для выхода: ")
        if command.lower() == 'q':
            break

        # Захват изображения
        image = capture_image()
        if image is None:
            continue

        # Поиск квадратного контура
        contour = find_square_contour(image)
        if contour is None:
            print("Квадратный контур не найден.")
            continue

        # Выравнивание контура
        aligned_image = align_contour(image, contour)

        # Классификация изображения
        predicted_letter = classify_image(aligned_image)
        print(f"Распознанная буква: {predicted_letter}")

        # Отображение результата
        cv2.imshow("Aligned Image", aligned_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()