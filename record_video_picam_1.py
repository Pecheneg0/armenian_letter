from picamera2 import Picamera2
from libcamera import controls
import time
import os

# Конфигурация
VIDEO_RESOLUTION = (720, 720)
FPS = 60
OUTPUT_DIR = "/home/pi/Desktop/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def record_video():
    # Инициализация камеры
    picam2 = Picamera2()
    
    # Настройка параметров
    config = picam2.create_video_configuration(
        main={"size": VIDEO_RESOLUTION, "format": "XRGB8888"},
        controls={
            "FrameRate": FPS,
            "AwbEnable": True,
            "AeEnable": True,
            "Sharpness": 1.0
        }
    )
    picam2.configure(config)
    
    # Создаем кодировщик H.264
    encoder = picam2.create_video_encoder()
    
    # Генерация имени файла
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")
    
    try:
        # Старт камеры
        picam2.start()
        print(f"Камера запущена. Начало записи: {output_file}")
        
        # Запуск записи с указанием кодировщика и файла
        encoder.output = output_file
        encoder.start()
        print("Запись начата...")
        
        # Запись в течение 10 секунд
        time.sleep(10)
        
        # Остановка записи
        encoder.stop()
        print("Запись остановлена.")
        
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        picam2.stop()
        picam2.close()

if __name__ == "__main__":
    record_video()
