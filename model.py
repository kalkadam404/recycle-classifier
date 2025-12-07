"""
Архитектура модели для классификации отходов
Использует предобученную ResNet18 в качестве backbone
"""
import torch
import torch.nn as nn
import torchvision.models as models


class WasteClassifier(nn.Module):
    """Модель классификатора отходов на основе ResNet18"""
    
    def __init__(self, num_classes=5, pretrained=True):
        """
        Args:
            num_classes: количество классов для классификации
            pretrained: использовать ли предобученные веса
        """
        super(WasteClassifier, self).__init__()
        
        # Загрузка предобученной ResNet18
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Замена последнего слоя
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class SimpleCNN(nn.Module):
    """Простая CNN архитектура (альтернативный вариант)"""
    
    def __init__(self, num_classes=5):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Первый блок свертки
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Второй блок свертки
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Третий блок свертки
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Четвертый блок свертки
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Тестирование модели
    model = WasteClassifier(num_classes=5)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Входной размер: {x.shape}")
    print(f"Выходной размер: {output.shape}")
    print(f"Количество параметров: {sum(p.numel() for p in model.parameters()):,}")

