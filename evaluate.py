"""
Скрипт для оценки модели с визуализацией результатов
Включает confusion matrix, classification report и примеры предсказаний
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import os

from data_loader import load_dataset
from model import WasteClassifier


def evaluate_model(model_path="models/best_model.pth", data_dir="garbage-dataset"):
    """Оценка модели на тестовой выборке"""
    
    # Устройство
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")
    
    # Загрузка чекпоинта
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint['class_names']
    num_classes = len(class_names)
    
    # Создание модели
    model = WasteClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Загрузка данных
    print("Загрузка тестовых данных...")
    _, _, test_loader, _ = load_dataset(data_dir)
    
    # Предсказания
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("Выполнение предсказаний...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Метрики
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\n{'='*60}")
    print(f"Точность на тестовой выборке: {accuracy*100:.2f}%")
    print(f"{'='*60}\n")
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Визуализация Confusion Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Истинные метки', fontsize=12)
    plt.xlabel('Предсказанные метки', fontsize=12)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/confusion_matrix.png", dpi=300, bbox_inches='tight')
    print("Confusion matrix сохранена в results/confusion_matrix.png")
    
    # Нормализованная Confusion Matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('Истинные метки', fontsize=12)
    plt.xlabel('Предсказанные метки', fontsize=12)
    plt.tight_layout()
    plt.savefig("results/confusion_matrix_normalized.png", dpi=300, bbox_inches='tight')
    print("Нормализованная confusion matrix сохранена в results/confusion_matrix_normalized.png")
    
    # Точность по классам
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_accuracies * 100, color='steelblue')
    plt.title('Точность по классам', fontsize=16, fontweight='bold')
    plt.ylabel('Точность (%)', fontsize=12)
    plt.xlabel('Класс', fontsize=12)
    plt.ylim([0, 100])
    plt.grid(axis='y', alpha=0.3)
    
    # Добавление значений на столбцы
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc*100:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/class_accuracy.png", dpi=300, bbox_inches='tight')
    print("График точности по классам сохранен в results/class_accuracy.png")
    
    # Количество примеров по классам
    class_counts = np.bincount(all_labels)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(class_names, class_counts, color='coral')
    plt.title('Распределение классов в тестовой выборке', fontsize=16, fontweight='bold')
    plt.ylabel('Количество изображений', fontsize=12)
    plt.xlabel('Класс', fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    # Добавление значений на столбцы
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("results/class_distribution.png", dpi=300, bbox_inches='tight')
    print("График распределения классов сохранен в results/class_distribution.png")
    
    print(f"\nВсе результаты сохранены в директории results/")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'class_accuracies': dict(zip(class_names, class_accuracies)),
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Оценка модели классификации отходов')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                        help='Путь к сохраненной модели')
    parser.add_argument('--data_dir', type=str, default='garbage-dataset',
                        help='Путь к директории с данными')
    
    args = parser.parse_args()
    
    evaluate_model(model_path=args.model_path, data_dir=args.data_dir)

