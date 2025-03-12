import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from model import ArmenianLetterNet  # Импортируем модель

# 🔹 Порог уверенности
confidence_threshold = 0.55

# 🔹 Армянский алфавит (от Юникода 1329 до 1366)
armenian_alphabet = [chr(code) for code in range(0x0531, 0x0557)]

# 🔹 Загружаем модель
model = ArmenianLetterNet()
model.load_state_dict(torch.load("armenian_letters_model_2.pth", map_location="cpu"))
model.eval()

# 🔹 Загружаем тестовое изображение
image_path = "/Users/aleksandr/Desktop/Снимок экрана 2025-03-09 в 23.18.55.png"  # Укажи путь к тестовому изображению
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (32, 32))
image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

# 🔹 Предсказание модели
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    confidence, predicted_class = torch.max(probabilities, 1)

confidence_value = confidence.item()
predicted_index = predicted_class.item()

# 🔹 Проверяем, предсказан ли корректный индекс
if 0 <= predicted_index < len(armenian_alphabet):
    predicted_letter = armenian_alphabet[predicted_index]
else:
    predicted_letter = "❌ Ошибка индекса"

# 🔹 Определяем результат
if confidence_value >= confidence_threshold:
    text_color = "green"
    result_text = f"✅ Распознанная буква: {predicted_letter} (Индекс: {predicted_index}) | Уверенность: {confidence_value:.2f}"
else:
    text_color = "red"
    result_text = f"⚠️ Буква не распознана с достаточной уверенностью! ({confidence_value:.2f})"

# 🔹 Визуализация результата
image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Преобразуем в цветное изображение
plt.figure(figsize=(5, 5))
plt.imshow(image_rgb, cmap="gray")
plt.axis("off")
plt.title(result_text, color=text_color, fontsize=12, weight="bold")
plt.show()




def predict_multiple_frames(model, images):
    """ Усредняет предсказания нескольких изображений. """
    predictions = []
    for image in images:
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_class = torch.max(output, 1)
            predictions.append(predicted_class.item())

    final_prediction = max(set(predictions), key=predictions.count)  # ✅ Берём наиболее частый класс
    return final_prediction





# 🔹 Вывод в консоль
print("📌 **Результат предсказания:**")
print(f"🔢 Индекс буквы: {predicted_index} | 🔤 Распознанная буква: {predicted_letter}")
print(f"📊 Уверенность модели: {confidence_value:.2f}")

# 🔹 Предупреждение, если индекс выходит за границы алфавита
if predicted_letter == "❌ Ошибка индекса":
    print("⚠️ Ошибка! Индекс буквы выходит за пределы допустимого диапазона (0-35).")
