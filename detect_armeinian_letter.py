import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import os
import time

# Пути к модели и меткам
MODEL_PATH = "model.tflite"
LABELS_PATH = "labels.txt"

# Загрузка меток
with open(LABELS_PATH, "r") as f:
    labels = [line.strip().split(" ")[1] for line in f.readlines()]

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

# Функция для обработки изображения
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.9 < aspect_ratio < 1.1:  # Проверка на квадратность
                roi = image[y:y+h, x:x+w]
                return roi, (x, y, w, h)
    return None, None

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

# Основной цикл
def main():
    while True:
        command = input("Нажмите Enter для захвата изображения или 'q' для выхода: ")
        if command.lower() == 'q':
            break

        image = capture_image()
        if image is None:
            continue

        roi, bbox = preprocess_image(image)
        if roi is None:
            print("Квадратный объект не найден.")
            continue

        predicted_letter = classify_image(roi)
        print(f"Распознанная буква: {predicted_letter}")

        # Отображение результата
        if bbox:
            x, y, w, h = bbox
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, f'Letter: {predicted_letter}', (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imshow("Result", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()