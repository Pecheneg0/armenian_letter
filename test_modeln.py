# test.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from modeln import ArmenianLetterNet

# Настройка предобработки изображений
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Загружаем тестовый датасет
test_dataset = datasets.ImageFolder(root="test_dataset", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Загружаем модель
model = ArmenianLetterNet()
model.load_state_dict(torch.load("armenian_letters_model_improved.pth", map_location="cpu"))
model.eval()

# Оценка модели
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Точность на тестовом наборе: {accuracy:.2f}%")