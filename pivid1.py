from picamera2 import Picamera2
from libcamera import controls
import cv2
import time
import os

# Конфигурация
VIDEO_RESOLUTION = (720, 720)
FPS = 30  # Можно увеличить до 60, если камера поддерживает
PREVIEW_WINDOW_TITLE = "Live Camera Feed"

def live_preview():
    # Инициализация камеры
    picam2 = Picamera2()
    
    # Настройка параметров
    config = picam2.create_preview_configuration(
        main={"size": VIDEO_RESOLUTION, "format": "XRGB8888"},
        controls={
            "FrameRate": FPS,
            "AwbEnable": True,  # Автобаланс белого
            "AeEnable": True,    # Автоэкспозиция
        }
    )
    picam2.configure(config)
    
    # Запуск камеры
    picam2.start()
    print(f"Запущен предпросмотр с разрешением {VIDEO_RESOLUTION} и {FPS} FPS")
    print("Нажмите 'q' для выхода...")
    
    try:
        while True:
            # Получение кадра
            frame = picam2.capture_array()
            
            # Конвертация из XRGB в BGR (для OpenCV)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Отображение кадра
            cv2.imshow(PREVIEW_WINDOW_TITLE, frame_bgr)
            
            # Выход по нажатию 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nПрервано пользователем")
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        # Освобождение ресурсов
        picam2.stop()
        cv2.destroyAllWindows()
        print("Камера остановлена")

if __name__ == "__main__":
    live_preview()
