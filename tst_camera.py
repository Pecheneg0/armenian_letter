import cv2

# Инициализация камеры
cap = cv2.VideoCapture(0)  # 0 — индекс камеры (CSI-камера)

# Проверка, открыта ли камера
if not cap.isOpened():
    print("Ошибка: Камера не подключена или не найдена.")
    exit()

# Захват изображения
ret, frame = cap.read()

# Проверка, удалось ли захватить изображение
if not ret:
    print("Ошибка: Не удалось захватить изображение.")
    exit()

# Сохранение изображения на диск
output_path = "/home/pi/captured_image.jpg"
cv2.imwrite(output_path, frame)
print(f"Изображение сохранено в {output_path}")

# Отображение изображения в окне
cv2.imshow("Captured Image", frame)
cv2.waitKey(0)  # Ожидание нажатия любой клавиши

# Закрытие окна и освобождение камеры
cv2.destroyAllWindows()
cap.release()