#!/usr/bin/env python3
import cv2
import numpy as np
import yaml
import os

def calibrate_camera_from_images(image_dir, pattern_size, square_size_m):
    """
    Калибровка камеры из набора изображений шахматной доски.

    Args:
        image_dir (str): Путь к папке с изображениями.
        pattern_size (tuple): Размер шаблона (внутренние углы) (cols, rows), например (8, 6).
        square_size_m (float): Размер квадрата в метрах.

    Returns:
        tuple: (mtx, dist, rvecs, tvecs) - матрица камеры, коэффициенты дисторсии, вращения, смещения.
    """
    # Критерии для уточнения угла
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Подготовка объектных точек (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp = objp * square_size_m # Преобразование в метры

    # Массивы для хранения точек объекта и точек изображения с разных изображений
    objpoints = []  # 3d точка в пространстве мира
    imgpoints = []  # 2d точка в плоскости изображения

    images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print(f"❌ Не найдено изображений в {image_dir}")
        return None, None, None, None

    print(f"🔍 Найдено {len(images)} изображений для калибровки.")

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Находим шахматную доску
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        # Если найдены, добавляем точки
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            print(f"✅ Углы найдены на изображении: {fname}")
        else:
            print(f"❌ Углы НЕ НАЙДЕНЫ на изображении: {fname}")

    if len(objpoints) == 0:
        print("❌ Не удалось найти углы на каком-либо из изображений.")
        return None, None, None, None

    print(f"📊 Использовано изображений для калибровки: {len(objpoints)}")

    # Калибровка
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    if ret:
        print("✅ Калибровка завершена успешно!")
        print(f"   Матрица камеры (mtx):\n{mtx}")
        print(f"   Коэффициенты дисторсии (dist):\n{dist.ravel()}")
        return mtx, dist, rvecs, tvecs
    else:
        print("❌ Калибровка не удалась.")
        return None, None, None, None

def save_calibration_yaml(mtx, dist, output_file, image_size):
    """
    Сохраняет результаты калибровки в формате ROS camera_info_manager.

    Args:
        mtx (numpy.ndarray): Матрица камеры.
        dist (numpy.ndarray): Коэффициенты дисторсии.
        output_file (str): Путь к файлу YAML для сохранения.
        image_size (tuple): (ширина, высота) изображения.
    """
    if mtx is None or dist is None:
        print("❌ Невозможно сохранить: данные калибровки отсутствуют.")
        return

    # Преобразование в список
    camera_matrix_list = mtx.flatten().tolist()
    distortion_coeffs_list = dist.flatten().tolist()

    # Создание словаря YAML
    calibration_data = {
        'image_width': image_size[0],
        'image_height': image_size[1],
        'camera_name': 'cam0',
        'camera_matrix': {
            'rows': 3,
            'cols': 3,
            'data': camera_matrix_list
        },
        'distortion_model': 'rational_polynomial', # или 'plumb_bob' для старых версий
        'distortion_coefficients': {
            'rows': 1,
            'cols': len(distortion_coeffs_list),
            'data': distortion_coeffs_list
        },
        'rectification_matrix': {
            'rows': 3,
            'cols': 3,
            'data': [1, 0, 0, 0, 1, 0, 0, 0, 1]
        },
        'projection_matrix': {
            'rows': 3,
            'cols': 4,
            'data': [
                camera_matrix_list[0], 0., camera_matrix_list[2], 0.,
                0., camera_matrix_list[4], camera_matrix_list[5], 0.,
                0., 0., 1., 0.
            ]
        }
    }

    with open(output_file, 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=False, sort_keys=False)

    print(f"💾 Результаты калибровки сохранены в: {output_file}")


if __name__ == '__main__':
    # ПАРАМЕТРЫ КАЛИБРОВКИ
    IMAGE_DIR = "/home/pi/tests/p1/p3/armenian_letter/calibration_images"
    PATTERN_SIZE = (9, 6)  # Размер шаблона (внутренние углы)
    SQUARE_SIZE_M = float(input("Введите размер квадрата в метрах (например, 0.024 для 2.4 см): "))
    OUTPUT_FILE = "/home/pi/tests/p1/p3/armenian_letter/calibration_result.yaml"

    # Запуск калибровки
    mtx, dist, rvecs, tvecs = calibrate_camera_from_images(IMAGE_DIR, PATTERN_SIZE, SQUARE_SIZE_M)

    # Сохранение результата
    if mtx is not None and dist is not None:
        # Предположим, что все изображения одного размера (берём первое)
        sample_img_path = [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))][0]
        sample_img = cv2.imread(sample_img_path)
        image_size = (sample_img.shape[1], sample_img.shape[0]) # (ширина, высота)
        save_calibration_yaml(mtx, dist, OUTPUT_FILE, image_size)

        # Вывод параметров для estimator_config.yaml
        print("\n" + "="*50)
        print("📋 ПАРАМЕТРЫ ДЛЯ estimator_config.yaml:")
        print("="*50)
        print(f"intrinsics: [{mtx[0,0]}, {mtx[1,1]}, {mtx[0,2]}, {mtx[1,2]}]")
        print(f"distortion_coeffs: [{dist[0,0]}, {dist[0,1]}, {dist[0,2]}, {dist[0,3]}] # Взяты первые 4")
        print("="*50)
    else:
        print("\n❌ Калибровка не удалась. Проверьте изображения и параметры.")

