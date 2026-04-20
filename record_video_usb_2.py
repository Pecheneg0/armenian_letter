import cv2

# Укажите индекс вашей USB камеры (обычно 0 или 1)
camera_index = 1

# Создаем объект VideoCapture
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Не удалось открыть камеру")
    exit()

# Получаем FPS, ширину и высоту видео
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Создаем объект VideoWriter для записи видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Используйте другой кодек, если нужно
output_file = 'output.mp4'
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while(True):
    # Читаем кадр из камеры
    ret, frame = cap.read()

    if not ret:
        print("Не удалось получить кадр")
        break

    # Записываем кадр в видеофайл
    out.write(frame)

    # Отображаем кадр (необязательно)
    cv2.imshow('frame', frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()

