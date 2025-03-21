# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR  # Добавлен Learning Rate Scheduler
from modeln import ArmenianLetterNet

# Настройка предобработки изображений
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomRotation(30),  # Повороты
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Яркость/контраст
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Смещение
    transforms.RandomHorizontalFlip(),  # Горизонтальное отражение
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Нормализация
])

# Загружаем датасет
train_dataset = datasets.ImageFolder(root="/Users/aleksandr/Downloads/Скат/dataset_generated_2", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Увеличен размер батча

# Инициализируем модель
model = ArmenianLetterNet()
criterion = nn.CrossEntropyLoss()  # Функция потерь
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Добавлена L2-регуляризация
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Learning Rate Scheduler

# Запускаем обучение
num_epochs = 30  # Увеличено количество эпох
for epoch in range(num_epochs):
    model.train()  # Переводим модель в режим обучения
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()  # Обнуляем градиенты

        # Прямой проход
        outputs = model(images)
        loss = criterion(outputs, labels)  # Вычисляем потерю
        loss.backward()  # Обратный проход
        optimizer.step()  # Обновляем веса

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Обновляем learning rate
    scheduler.step()

    accuracy = 100 * correct / total
    print(f"📊 Эпоха {epoch+1}/{num_epochs} | Потеря: {running_loss:.4f} | Точность: {accuracy:.2f}%")

print("✅ Обучение завершено!")
torch.save(model.state_dict(), "armenian_letters_model_x_TF.pth")
print("✅ Модель сохранена!")