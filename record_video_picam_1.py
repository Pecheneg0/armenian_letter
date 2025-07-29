from picamera2 import Picamera2
from libcamera import controls
import time
import os

# Конфигурация (аналогично вашему проекту)
VIDEO_RESOLUTION = (720, 720)
FPS = 60
OUTPUT_DIR = "/home/pi/Desktop/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def record_video():
    # Инициализация камеры
    picam2 = Picamera2()
    
    # Настройка параметров (как в вашем коде)
    config = picam2.create_video_configuration(
        main={"size": VIDEO_RESOLUTION, "format": "RGB888"},
        controls={
            "FrameRate": FPS,
            "AwbEnable": True,  # Автобаланс белого
            "AeEnable": True,   # Автоэкспозиция
            "Sharpness": 1.0    # Резкость
        }
    )
    picam2.configure(config)
    
    # Генерация имени файла с временной меткой
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")
    
    try:
        # Старт записи
        picam2.start()
        print(f"Запись начата: {output_file} (Разрешение: {VIDEO_RESOLUTION}, FPS: {FPS})")
        
        # Запись в течение 10 секунд (для теста)
        picam2.start_recording(output_file)
        time.sleep(10)
        
        # Остановка
        picam2.stop_recording()
        print("Запись завершена.")
        
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        picam2.stop()
        picam2.close()

if __name__ == "__main__":
    record_video()