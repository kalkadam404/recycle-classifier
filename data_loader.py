"""
Модуль для загрузки и предобработки данных с использованием OpenCV
"""
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch


class WasteDataset(Dataset):
    """Класс для загрузки изображений отходов"""
    
    def __init__(self, image_paths, labels, transform=None):
        """
        Args:
            image_paths: список путей к изображениям
            labels: список меток (индексы классов)
            transform: трансформации для аугментации данных
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Загрузка изображения с помощью OpenCV
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        
        # Конвертация BGR в RGB (OpenCV использует BGR по умолчанию)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Применение трансформаций
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def load_dataset(data_dir, categories=None, test_size=0.2, val_size=0.1):
    """
    Загружает датасет из директории
    
    Args:
        data_dir: путь к директории с данными
        categories: список категорий для использования (None = все категории)
        test_size: доля тестовой выборки
        val_size: доля валидационной выборки (от обучающей)
    
    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    if categories is None:
        # Используем основные категории для классификации отходов
        categories = ['plastic', 'glass', 'paper', 'metal', 'biological']
    
    image_paths = []
    labels = []
    class_names = []
    
    # Собираем все изображения и метки
    for class_idx, category in enumerate(categories):
        category_dir = os.path.join(data_dir, category)
        if not os.path.exists(category_dir):
            print(f"Предупреждение: директория {category_dir} не найдена, пропускаем")
            continue
        
        class_names.append(category)
        files = [f for f in os.listdir(category_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for file in files:
            image_paths.append(os.path.join(category_dir, file))
            labels.append(class_idx)
    
    print(f"Загружено {len(image_paths)} изображений из {len(class_names)} категорий")
    for i, name in enumerate(class_names):
        count = labels.count(i)
        print(f"  {name}: {count} изображений")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=test_size, random_state=42, stratify=labels
    )
    
    # Разделение train на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42, stratify=y_train
    )
    
    # Трансформации для обучения (с аугментацией)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Трансформации для валидации и теста (без аугментации)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Создание датасетов
    train_dataset = WasteDataset(X_train, y_train, transform=train_transform)
    val_dataset = WasteDataset(X_val, y_val, transform=val_transform)
    test_dataset = WasteDataset(X_test, y_test, transform=val_transform)
    
    # Создание DataLoader'ов
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, class_names


if __name__ == "__main__":
    # Тестирование загрузки данных
    data_dir = "garbage-dataset"
    train_loader, val_loader, test_loader, class_names = load_dataset(data_dir)
    
    print(f"\nКатегории: {class_names}")
    print(f"Размер обучающей выборки: {len(train_loader.dataset)}")
    print(f"Размер валидационной выборки: {len(val_loader.dataset)}")
    print(f"Размер тестовой выборки: {len(test_loader.dataset)}")

