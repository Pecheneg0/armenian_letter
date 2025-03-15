import cv2
import numpy as np
from picamera2 import Picamera2
import time

# 🔹 Настройка камеры
camera = Picamera2()
config = camera.create_preview_configuration(main={"size": (1640, 1232), "format": "RGB888"})
camera.configure(config)
camera.start()

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

# 🔹 Функция для выравнивания и обрезки изображения
def align_and_crop(image, contour):
    """ Выравнивает и обрезает изображение до прямоугольника """
    # Получаем координаты вершин контура
    points = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    
    # Сумма координат будет минимальной у левого верхнего угла и максимальной у правого нижнего
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]  # Левый верхний угол
    rect[2] = points[np.argmax(s)]  # Правый нижний угол
    
    # Разница координат будет минимальной у правого верхнего угла и максимальной у левого нижнего
    diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(diff)]  # Правый верхний угол
    rect[3] = points[np.argmax(diff)]  # Левый нижний угол
    
    # Вычисляем ширину и высоту прямоугольника
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    
    # Выбираем максимальные значения для ширины и высоты
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    
    # Точки для преобразования
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Вычисляем матрицу преобразования и применяем её
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

# 🔹 Основной цикл
def scan_and_display():
    """ Сканирует изображения и выводит обрезанный контур на экран """
    while True:
        # Снимаем кадр с камеры
        frame = camera.capture_array("main")
        
        # Находим контур прямоугольника и бинарное изображение
        square_contour, binary_image = find_square_contour(frame)
        
        if square_contour is not None:
            # Выравниваем и обрезаем изображение
            aligned_image = align_and_crop(binary_image, square_contour)
            
            # Выводим обрезанное изображение на экран
            cv2.imshow("Aligned and Cropped Contour", aligned_image)
        
        # Если контур не найден, выводим сообщение
        else:
            print("⚠️ Прямоугольный контур не найден.")
        
        # Ожидание нажатия клавиши 'q' для выхода
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # Пауза между кадрами (3-4 кадра в 10 секунд)
        time.sleep(2.5)  # 10 секунд / 4 кадра = 2.5 секунды на кадр

    # Освобождаем ресурсы
    cv2.destroyAllWindows()
    camera.stop()

# 🚀 Запускаем сканирование и отображение
scan_and_display()