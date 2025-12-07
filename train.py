"""
Скрипт для обучения модели классификации отходов
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from data_loader import load_dataset
from model import WasteClassifier


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Обучение модели на одной эпохе"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Статистика
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Валидация модели"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def train_model(data_dir="garbage-dataset", num_epochs=20, lr=0.001, batch_size=32):
    """Основная функция обучения"""
    
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Загрузка данных
    print("Загрузка данных...")
    train_loader, val_loader, test_loader, class_names = load_dataset(data_dir)
    num_classes = len(class_names)
    
    # Создание модели
    model = WasteClassifier(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)
    
    # История обучения
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0
    
    print(f"\nНачало обучения на {num_epochs} эпох...")
    print(f"Классы: {class_names}\n")
    
    # Обучение
    for epoch in range(num_epochs):
        print(f"\nЭпоха {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Обучение
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Валидация
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Обновление learning rate
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': class_names,
            }, "models/best_model.pth")
            print(f"✓ Сохранена лучшая модель (Val Acc: {val_acc:.2f}%)")
    
    # Визуализация истории обучения
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.title('График потерь')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy (%)')
    plt.title('График точности')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/training_history.png", dpi=300, bbox_inches='tight')
    print(f"\nГрафик обучения сохранен в results/training_history.png")
    
    print(f"\nОбучение завершено!")
    print(f"Лучшая точность на валидации: {best_val_acc:.2f}%")
    
    return model, class_names


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение модели классификации отходов')
    parser.add_argument('--data_dir', type=str, default='garbage-dataset',
                        help='Путь к директории с данными')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Количество эпох обучения')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Размер батча')
    
    args = parser.parse_args()
    
    train_model(
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size
    )

