import cv2
import torch
import numpy as np
import time
from picamera2 import Picamera2
from PIL import Image
from torchvision import transforms

# Загрузка модели
from modeln import ArmenianLetterNet
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

model = ArmenianLetterNet()
model.load_state_dict(torch.load("armenian_letters_model_improved.pth", map_location="cpu"))
model.eval()

# Настройка камеры
camera = Picamera2()
camera.configure(camera.create_preview_configuration(main={"size": (3280, 2464), "format": "RGB888"}))
camera.start()

def process_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return transform(image).unsqueeze(0)

def predict_letter(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
    return labels[predicted_class.item()], confidence.item()

while True:
    frame = camera.capture_array("main")
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                offset_x, offset_y = cx - center_x, cy - center_y

                cropped = frame[max(cy-16,0):cy+16, max(cx-16,0):cx+16]
                if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    cropped_pil = Image.fromarray(cropped)
                    image_tensor = process_image(cropped_pil)
                    letter, confidence = predict_letter(image_tensor)
                    print(f"Распознана буква: {letter} ({confidence:.2f}), Смещение: X={offset_x}px, Y={offset_y}px")

    time.sleep(2.5)
