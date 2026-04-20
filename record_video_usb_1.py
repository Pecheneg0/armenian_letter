import cv2
import time
import os
from datetime import datetime

# Конфигурация
VIDEO_RESOLUTION = (720, 720)
FPS = 60  # Для USB-камер обычно 30 FPS стабильнее
OUTPUT_DIR = "/home/pi/Desktop/videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CAMERA_ID = 0  # Обычно 0 для встроенной/первой USB-камеры

def record_video():
    # Инициализация камеры
    cap = cv2.VideoCapture(CAMERA_ID)
    
    # Установка параметров
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, FPS)
    
    # Проверка открытия камеры
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру!")
        return
    
    # Настройка кодека и создание VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # или 'X264' если доступен
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(OUTPUT_DIR, f"usb_video_{timestamp}.mp4")
    
    out = cv2.VideoWriter(output_file, fourcc, FPS, VIDEO_RESOLUTION)
    
    print(f"Начало записи видео: {output_file}")
    start_time = time.time()
    recording_duration = 10  # Длительность записи в секундах
    
    try:
        while (time.time() - start_time) < recording_duration:
            ret, frame = cap.read()
            if not ret:
                print("Ошибка: Не удалось получить кадр!")
                break
            
            # Конвертация цвета (если нужно)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Запись кадра
            out.write(frame)
            
            # Для отображения превью (опционально)
            cv2.imshow('Recording...', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Ошибка при записи видео: {e}")
    finally:
        # Освобождение ресурсов
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Запись завершена. Видео сохранено как: {output_file}")

if __name__ == "__main__":
    record_video()
