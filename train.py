import torch
import torch.nn as nn  # ✅ Импортируем nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import ArmenianLetterNet  # Импортируем модель

# Настройка предобработки изображений
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomRotation(30),  # ✅ Повороты
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # ✅ Яркость/контраст
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # ✅ Смещение
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Загружаем датасет
train_dataset = datasets.ImageFolder(root="dataset_generated_2", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Инициализируем модель
model = ArmenianLetterNet()
criterion = nn.CrossEntropyLoss()  # ✅ Теперь nn определён
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Запускаем обучение
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"📊 Эпоха {epoch+1}/{num_epochs} | Потеря: {running_loss:.4f} | Точность: {accuracy:.2f}%")

print("✅ Обучение завершено!")
torch.save(model.state_dict(), "armenian_letters_model_2.pth")
print("✅ Модель сохранена!")
