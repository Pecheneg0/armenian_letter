from picamera2 import Picamera2
from libcamera import controls
import time
import os
import logging



# Конфигурация
VIDEO_RESOLUTION = (720, 720)
FPS = 60
OUTPUT_DIR = "/home/pi/Desktop/videos"
LOG_DIR = "/home/pi/Desktop"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok = True)


logging.basicConfig(
    filename=os.path.join(LOG_DIR, "letters_detect.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def record_video():
    # Инициализация камеры
    picam2 = Picamera2()
    
    # Настройка параметров (как в вашем коде)
    config = picam2.create_video_configuration(
        main={"size": VIDEO_RESOLUTION, "format": "XRGB8888"},
        controls={
            #"FrameRate": FPS,
            #"AwbEnable": True,  # Автобаланс белого
            #"AeEnable": True,   # Автоэкспозиция
            #"Sharpness": 1.0    # Резкость
            "FrameRate": FPS,
            
            "AwbEnable": True,  # Отключаем авто баланс белого
            "AwbMode": True,       # Режим "Auto" (0) или конкретный режим освещения
            "AeEnable": True,   # Отключаем авто экспозицию
            "ExposureTime": 4900,  # В микросекундах (начните с 10000 и регулируйте)
            "AnalogueGain": 3.6,    # Минимальное усиление
            "Brightness": 0,      # 0-1 (рекомендуется 0.1-0.3)
            "Contrast": 1,        # 1.0 = нормальный, >1.0 увеличивает контраст
            "Saturation": 0,      # 0.0-2.0 (меньше = менее насыщенные цвета)
            "Sharpness": 1,       # 0.0-16.0 (умеренная резкость)
        }
    )
    picam2.configure(config)
    
    # Генерация имени файла с временной меткой
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")
    
    try:
        # Используем метод, который точно работает
        logging.info(f"Начало записи видео: {output_file}")
        picam2.start_and_record_video(output_file, duration=300)
        #print("Запись успешно завершена!")
        
    except Exception as e:
        logging.info(f"Ошибка при записи видео: {e}")
    finally:
        # Всегда освобождаем ресурсы камеры
        picam2.stop()
        picam2.close()
        logging.info("Ресурсы камеры освобождены")

if __name__ == "__main__":
    record_video()
