# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ArmenianLetterNet(nn.Module):
    def __init__(self):
        super(ArmenianLetterNet, self).__init__()
        # Свёрточные слои
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  # Исправлено на 128 * 4 * 4
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 36)  # 36 классов
        
        # Пулинг и дропаут
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Применяем свёрточные слои, BatchNorm и пулинг
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Выравниваем тензор для полносвязных слоёв
        x = x.view(x.size(0), -1)
        
        # Применяем полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x