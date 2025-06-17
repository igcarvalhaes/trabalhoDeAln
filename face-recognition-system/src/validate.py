import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report  # métricas de avaliação (confusion matrix, classification report)
import seaborn as sns  # biblioteca de visualização para melhorar o gráfico de confusão
from dataset import load_dataset, load_difficulty_levels   # carregamento dos dados de validação
from train import FaceRecognitionModel  # importando o modelo de reconhecimento facial definido no arquivo train.py

# Configurações
VALID_DIR = 'c:/Users/Igor/Documents/Trabalho de ALN/dataset/valid' # Diretório de validação
IMG_SIZE = (100, 100)  # Tamanho das imagens
MODEL_PATH = 'facial_recognition_model.pth'   # Caminho do modelo treinado
BATCH_SIZE = 32                               # Tamanho do lote (quantas imagens processamos de cada vez)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def validate_model(): # Essa é a função que faz toda a avaliação do modelo
    if any(os.path.isdir(os.path.join(VALID_DIR, d)) and d in ['easy', 'medium', 'hard', 'very-easy', 'extras']    # verifica se os dados estão organizados em subpastas
           for d in os.listdir(VALID_DIR)):
        print("Loading data from difficulty level folders...")
        val_loader, val_dataset = load_difficulty_levels(VALID_DIR, IMG_SIZE, batch_size=BATCH_SIZE)  # Se sim, carrega os dados organizados por dificuldade
    else:
        print("Loading data from flat directory structure...")
        val_loader, val_dataset = load_dataset(VALID_DIR, IMG_SIZE, batch_size=BATCH_SIZE) # Se não, carrega os dados organizados em uma única pasta
    
    num_classes = val_dataset.num_classes
    
    if len(val_dataset) == 0:  # Verifica se o dataset de validação não está vazio.
        print("No validation data found!")
        return
    
    # Carrega o modelo treinado
    model = FaceRecognitionModel(num_classes)  # Inicializa o modelo com o número de classes do dataset de validação
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))  # Carrega os pesos do modelo treinado
    model = model.to(DEVICE)           # Move o modelo para o dispositivo (GPU ou CPU)
    model.eval()                       # Coloca o modelo em modo de avaliação (desativa dropout, batch normalization, etc)
    
    # Avaliação do modelo
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():     # Desativa o cálculo de gradientes para economizar memória e acelerar a avaliação
        for inputs, labels in val_loader:  # Itera para cada lote de imagens e labels:
            inputs = inputs.to(DEVICE)     # Move as imagens para o dispositivo (GPU ou CPU)
            labels = labels.to(DEVICE)     # Move os labels para o dispositivo
            
            outputs = model(inputs)        # Passa as imagens pelo modelo para obter as previsões
            _, predicted = torch.max(outputs.data, 1) 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Conta quantas previsões estão corretas
            
            all_preds.extend(predicted.cpu().numpy())  # Armazena as previsões
            all_labels.extend(labels.cpu().numpy())    # Armazena os labels verdadeiros
    
    # Calcula e imprime a acurácia total do modelo na validação.
    accuracy = 100 * correct / total    
    print(f"Validation Accuracy: {accuracy:.2f}%")
    
    # Percorre cada classe e calcula a acurácia individual.
    classes = val_dataset.class_names
    for i, class_name in enumerate(classes):
        class_indices = np.where(np.array(all_labels) == i)[0]
        if len(class_indices) > 0:
            class_preds = np.array(all_preds)[class_indices]
            class_labels = np.array(all_labels)[class_indices]
            class_accuracy = np.mean(class_preds == class_labels) * 100
            print(f"Class: {class_name}, Accuracy: {class_accuracy:.2f}%, Samples: {len(class_indices)}")
    
    # Calcula a matriz de confusão: mostra onde o modelo errou e acertou.
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Gera e salva o gráfico da matriz de confusão. Visualiza como o modelo está se saindo em cada classe.
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    
    # Gera o relatório de classificação: fornece métricas detalhadas como precisão, recall e F1-score para cada classe.
    report = classification_report(all_labels, all_preds, target_names=classes)
    print("Classification Report:")
    print(report)
    
    # Salva o relatório de classificação e a acurácia geral em arquivo de texto.
    with open('classification_report.txt', 'w') as f:
        f.write(f"Validation Accuracy: {accuracy:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)

# Executa a função de validação quando o arquivo é chamado diretamente.
if __name__ == "__main__":
    validate_model()