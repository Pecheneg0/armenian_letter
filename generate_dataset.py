import cv2
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random

# Создаём папку для датасета
dataset_path = "/Users/aleksandr/Desktop/Работа/СКАТ/new_model/dataset_generated_1"
os.makedirs(dataset_path, exist_ok=True)

# Армянский алфавит (буквы от U+0531 до U+0556)
armenian_letters = [chr(code) for code in range(0x0531, 0x0557)]

# Параметры генерации
img_size = 128  # Размер оригинального изображения
final_size = 32  # Размер после сжатия

# Создаём изображения для каждой буквы
for letter in armenian_letters:
    letter_folder = os.path.join(dataset_path, letter)
    os.makedirs(letter_folder, exist_ok=True)

    for i in range(300):  # 300 изображений на букву
        img = Image.new('L', (img_size, img_size), 255)  # Белый фон
        draw = ImageDraw.Draw(img)

        # Используем системный шрифт (замени на армянский, если есть)
        font_path = "/System/Library/Fonts/Supplemental/Mshtakan.ttc"  # Укажи путь к шрифту
        font = ImageFont.truetype(font_path, random.randint(80, 120))

        # 📌 Используем textbbox() вместо textsize()
        bbox = draw.textbbox((0, 0), letter, font=font)  
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Рисуем букву в центре
        draw.text(((img_size - w) // 2, (img_size - h) // 2), letter, font=font, fill=0)

        # ✅ Добавляем случайное вращение, но без expand=True (чтобы не менялся размер)
        img = img.rotate(random.uniform(-30, 30), fillcolor=255)

        # ✅ Обрезаем изображение обратно до 128x128
        img = img.crop((0, 0, img_size, img_size))

        # ✅ Добавляем размытие (имитация движения дрона)
        img = img.filter(ImageFilter.GaussianBlur(random.uniform(0, 2)))

        # ✅ Добавляем шум (исправляем размер)
        noise = np.random.normal(0, 10, (img_size, img_size))
        img_np = np.array(img)
        if img_np.shape != noise.shape:
            noise = cv2.resize(noise, (img_np.shape[1], img_np.shape[0]))  # Исправляем размер шума
        img_np = img_np + noise
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_np)

        # ✅ Уменьшаем размер до 32x32 (убрали `ANTIALIAS`)
        img = img.resize((final_size, final_size))

        # Сохраняем
        img.save(os.path.join(letter_folder, f"{i}.png"))

print("✅ Датасет успешно создан!")
