import cv2
import numpy as np
import os
from datetime import datetime
import time

# === Конфигурация ===
IMAGE_PATH = '/Users/aleksandr/Desktop/Снимок экрана 2025-05-27 в 17.08.51.png'
SAVE_DIR = '/Users/aleksandr/Desktop/Работа/СКАТ/Contours_Output'
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_INTERVAL = 10  # секунд
last_save_time = time.time()

def extract_letter_image(thresh_img, padding=40, output_size=32):
    # Инвертируем изображение, чтобы чёрная буква стала белым контуром
    inverted_img = cv2.bitwise_not(thresh_img)
    contours, _ = cv2.findContours(inverted_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ Контур не найден.")
        return None, thresh_img

    # Находим контур с максимальной площадью
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Расширяем рамку с учётом границ
    x_exp = max(x - padding, 0)
    y_exp = max(y - padding, 0)
    x_end = min(x + w + padding, inverted_img.shape[1])
    y_end = min(y + h + padding, inverted_img.shape[0])

    # Вырезаем ROI (область вокруг буквы)
    roi = inverted_img[y_exp:y_end, x_exp:x_end]

    # Создаём квадратный холст
    roi_height, roi_width = roi.shape
    size = max(roi_height, roi_width)
    square_img = np.ones((size, size), dtype=np.uint8) * 255  # белый фон
    y_offset = (size - roi_height) // 2
    x_offset = (size - roi_width) // 2
    square_img[y_offset:y_offset+roi_height, x_offset:x_offset+roi_width] = roi

    # Масштабируем до заданного размера
    resized_img = cv2.resize(square_img, (output_size, output_size), interpolation=cv2.INTER_AREA)

    # Рисуем контуры на копии исходного изображения
    img_with_drawings = cv2.cvtColor(thresh_img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_with_drawings, [largest_contour], -1, (0, 0, 255), 2)
    cv2.rectangle(img_with_drawings, (x_exp, y_exp), (x_end, y_end), (0, 255, 0), 2)

    # Инвертируем результат обратно для показа буквы чёрной на белом
    inverted_again_img = cv2.bitwise_not(resized_img)

    return inverted_again_img, img_with_drawings

# === Основная обработка ===
# Загружаем изображение
my_photo = cv2.imread(IMAGE_PATH)
if my_photo is None:
    print("❌ Изображение не найдено!")
    exit()

# Предобработка: фильтр и преобразование в серый
filtered_image = cv2.medianBlur(my_photo, 7)
img_grey = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

# Пороговое преобразование
thresh = 100
_, thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)

# Извлекаем букву
processed_letter, thresh_with_drawings = extract_letter_image(thresh_img, padding=40, output_size=32)

if processed_letter is not None:
    # Показываем изображение с рамкой и букву
    cv2.imshow("Threshold with Contour and Bounding Box", thresh_with_drawings)
    cv2.imshow("Processed Letter", processed_letter)

    # Сохраняем с интервалом
    if time.time() - last_save_time > SAVE_INTERVAL:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename_letter = os.path.join(SAVE_DIR, f'letter_{timestamp}.png')
        filename_drawings = os.path.join(SAVE_DIR, f'thresh_drawings_{timestamp}.png')
        #cv2.imwrite(filename_letter, processed_letter)
        #cv2.imwrite(filename_drawings, thresh_with_drawings)
        print(f'💾 Сохранено: {filename_letter}')
        print(f'💾 Сохранено: {filename_drawings}')
        last_save_time = time.time()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("❌ Буква не найдена.")

