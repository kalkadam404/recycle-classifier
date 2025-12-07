"""
Скрипт для предсказания класса отходов на новых изображениях
"""
import torch
import cv2
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import os
import argparse

from model import WasteClassifier


def load_image(image_path):
    """Загрузка и предобработка изображения"""
    # Загрузка с помощью OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")
    
    # Конвертация BGR в RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def preprocess_image(image):
    """Предобработка изображения для модели"""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Добавляем batch dimension
    return image_tensor


def predict_image(model_path, image_path, class_names):
    """Предсказание класса для одного изображения"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загрузка модели
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = len(class_names)
    
    model = WasteClassifier(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Загрузка и предобработка изображения
    image = load_image(image_path)
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)
    
    # Предсказание
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        prob, pred = torch.max(probs, 1)
    
    predicted_class = class_names[pred.item()]
    confidence = prob.item() * 100
    
    return predicted_class, confidence, probs[0].cpu().numpy(), image


def predict_single(image_path, model_path="models/best_model.pth"):
    """Предсказание для одного изображения с визуализацией"""
    # Загрузка классов из модели
    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint['class_names']
    
    # Предсказание
    predicted_class, confidence, all_probs, image = predict_image(
        model_path, image_path, class_names
    )
    
    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Изображение с предсказанием
    axes[0].imshow(image)
    axes[0].axis('off')
    axes[0].set_title(f'Предсказание: {predicted_class}\nУверенность: {confidence:.2f}%', 
                     fontsize=14, fontweight='bold')
    
    # График вероятностей
    y_pos = np.arange(len(class_names))
    axes[1].barh(y_pos, all_probs * 100, color='steelblue')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(class_names)
    axes[1].set_xlabel('Вероятность (%)', fontsize=12)
    axes[1].set_title('Вероятности по классам', fontsize=14, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    # Добавление значений на столбцы
    for i, prob in enumerate(all_probs):
        axes[1].text(prob * 100, i, f'{prob*100:.1f}%',
                    va='center', ha='left', fontsize=10)
    
    plt.tight_layout()
    
    # Сохранение результата
    os.makedirs("results/predictions", exist_ok=True)
    output_path = f"results/predictions/{os.path.basename(image_path)}"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Результат сохранен в {output_path}")
    
    plt.show()
    
    print(f"\nПредсказание: {predicted_class}")
    print(f"Уверенность: {confidence:.2f}%")
    print("\nВсе вероятности:")
    for i, (class_name, prob) in enumerate(zip(class_names, all_probs)):
        marker = " <--" if i == np.argmax(all_probs) else ""
        print(f"  {class_name}: {prob*100:.2f}%{marker}")


def predict_batch(image_dir, model_path="models/best_model.pth", num_images=10):
    """Предсказание для нескольких изображений"""
    # Загрузка классов
    checkpoint = torch.load(model_path, map_location='cpu')
    class_names = checkpoint['class_names']
    
    # Поиск изображений
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(image_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if len(image_files) == 0:
        print(f"Изображения не найдены в {image_dir}")
        return
    
    image_files = image_files[:num_images]
    
    print(f"Обработка {len(image_files)} изображений...")
    
    # Предсказания
    results = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        try:
            predicted_class, confidence, _, _ = predict_image(
                model_path, img_path, class_names
            )
            results.append({
                'image': img_file,
                'prediction': predicted_class,
                'confidence': confidence
            })
            print(f"{img_file}: {predicted_class} ({confidence:.2f}%)")
        except Exception as e:
            print(f"Ошибка при обработке {img_file}: {e}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Предсказание класса отходов на изображении')
    parser.add_argument('--image', type=str, help='Путь к изображению')
    parser.add_argument('--image_dir', type=str, help='Путь к директории с изображениями')
    parser.add_argument('--model_path', type=str, default='models/best_model.pth',
                        help='Путь к модели')
    parser.add_argument('--num_images', type=int, default=10,
                        help='Количество изображений для обработки (при использовании --image_dir)')
    
    args = parser.parse_args()
    
    if args.image:
        predict_single(args.image, args.model_path)
    elif args.image_dir:
        predict_batch(args.image_dir, args.model_path, args.num_images)
    else:
        print("Укажите --image для одного изображения или --image_dir для директории")

