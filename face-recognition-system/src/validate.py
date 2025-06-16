import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from dataset import load_dataset, load_difficulty_levels
from train import FaceRecognitionModel

# Configuration
VALID_DIR = 'c:/Users/Igor/Documents/Trabalho de ALN/dataset/valid'
IMG_SIZE = (100, 100)  # Same as training
MODEL_PATH = 'facial_recognition_model.pth'
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate_model():
    # Load validation data
    if any(os.path.isdir(os.path.join(VALID_DIR, d)) and d in ['easy', 'medium', 'hard', 'very-easy', 'extras'] 
           for d in os.listdir(VALID_DIR)):
        print("Loading data from difficulty level folders...")
        val_loader, val_dataset = load_difficulty_levels(VALID_DIR, IMG_SIZE, batch_size=BATCH_SIZE)
    else:
        print("Loading data from flat directory structure...")
        val_loader, val_dataset = load_dataset(VALID_DIR, IMG_SIZE, batch_size=BATCH_SIZE)
    
    num_classes = val_dataset.num_classes
    
    if len(val_dataset) == 0:
        print("No validation data found!")
        return
    
    # Load the trained model
    model = FaceRecognitionModel(num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    
    # Evaluate the model
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    
    # Calculate per-class accuracy
    classes = val_dataset.class_names
    for i, class_name in enumerate(classes):
        class_indices = np.where(np.array(all_labels) == i)[0]
        if len(class_indices) > 0:
            class_preds = np.array(all_preds)[class_indices]
            class_labels = np.array(all_labels)[class_indices]
            class_accuracy = np.mean(class_preds == class_labels) * 100
            print(f"Class: {class_name}, Accuracy: {class_accuracy:.2f}%, Samples: {len(class_indices)}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Detailed classification report
    report = classification_report(all_labels, all_preds, target_names=classes)
    print("Classification Report:")
    print(report)
    
    # Save report to file
    with open('classification_report.txt', 'w') as f:
        f.write(f"Validation Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)

if __name__ == "__main__":
    validate_model()