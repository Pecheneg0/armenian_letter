#!/usr/bin/env python3
"""
Скрипт для захвата изображений шахматной доски для калибровки камеры.
Использует Picamera2 для захвата изображений с камеры Raspberry Pi CSI.
Сохраняет изображения в формате chessboard_000.jpg, chessboard_001.jpg и т.д.
"""

import cv2
from picamera2 import Picamera2
import time
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Capture calibration images for camera calibration.')
    parser.add_argument('--output_dir', type=str, default='~/calibration_images', help='Directory to save images')
    parser.add_argument('--width', type=int, default=320, help='Width of the captured image')
    parser.add_argument('--height', type=int, default=240, help='Height of the captured image')
    parser.add_argument('--count', type=int, default=50, help='Number of images to capture')
    parser.add_argument('--delay', type=float, default=2.0, help='Delay between captures in seconds')
    args = parser.parse_args()

    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"🚀 Захват изображений для калибровки камеры.")
    print(f"📸 Разрешение: {args.width}x{args.height}")
    print(f"📁 Папка для сохранения: {output_dir}")
    print(f"📸 Количество изображений: {args.count}")
    print(f"⏱️  Задержка между снимками: {args.delay} секунд")
    print("\n💡 Подсказки:")
    print("   - Поместите шахматную доску перед камерой.")
    print("   - Медленно перемещайте доску, покрывая всё поле зрения.")
    print("   - Наклоняйте доску под разными углами.")
    print("   - Нажмите 's' для захвата изображения, 'q' или 'ESC' для выхода.\n")

    # Инициализация камеры
    picam2 = Picamera2()
    
    # Конфигурация камеры
    config = picam2.create_preview_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"},
        controls={
            "FrameRate": 15,  # Уменьшаем частоту кадров для снижения нагрузки
            "AwbEnable": True, # Включаем AWB для лучшего цвета
            "AeEnable": True,  # Включаем AE для адаптации к освещению
            # "ExposureTime": 10000, # Можно раскомментировать для фиксации
            # "AnalogueGain": 1.0,   # Можно раскомментировать для фиксации
            "Brightness": 0.0,
            "Contrast": 1.0,
            "Saturation": 1.0,
            "Sharpness": 1.0
        }
    )
    picam2.configure(config)
    picam2.start()

    # Инициализация OpenCV окна
    cv2.namedWindow("Calibration Image Capture", cv2.WINDOW_AUTOSIZE)

    count = 0
    last_capture_time = time.time()

    try:
        while count < args.count:
            # Получение кадра
            frame = picam2.capture_array()
            # Конвертация из RGB в BGR для OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Отображение информации на изображении
            #info_text = [
             #   f"Captured: {count}/{args.count}",
              #  f"Resolution: {args.width}x{args.height}",
               # f"Press 's' to save, 'q' to quit"
           # ]
            #y_offset = 30
            #for text in info_text:
             #   cv2.putText(frame_bgr, text, (10, y_offset),
              #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
               # y_offset += 30

            # Отображение кадра
            cv2.imshow("Calibration Image Capture", frame_bgr)

            # Обработка клавиш
            key = cv2.waitKey(1) & 0xFF
            current_time = time.time()

            if key == ord('s'):
                if current_time - last_capture_time >= args.delay:
                    filename = os.path.join(output_dir, f"chessboard_{count:03d}.jpg")
                    cv2.imwrite(filename, frame_bgr)
                    print(f"📸 Сохранено: {filename}")
                    count += 1
                    last_capture_time = current_time
                else:
                    remaining = args.delay - (current_time - last_capture_time)
                    print(f"⏳ Подождите ещё {remaining:.1f} секунд перед следующим снимком.")
            elif key in [ord('q'), 27]: # 'q' или ESC
                print("\n⏹️  Захват остановлен пользователем.")
                break

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

    print(f"\n✅ Захват завершён. Сохранено {count} изображений в {output_dir}.")

if __name__ == '__main__':
    main()

