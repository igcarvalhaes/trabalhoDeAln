import os           
import torch                 # deep learning (PyTorch)
import torch.nn as nn        # construção de redes neurais
import torch.optim as optim  # algoritmos de otimização
import numpy as np
import matplotlib.pyplot as plt  # gerar gráficos de evolução do treinamento
from torch.utils.data import DataLoader, random_split  # dividir dataset em treino e validação
from torchvision import transforms, models  # modelos pré-treinados da torchvision (usamos o ResNet18)
from dataset import FaceDataset, load_dataset, load_difficulty_levels # carregamento e processamento de dados já feito no arquivo

# Configuration
TRAIN_DIR = 'c:/Users/Igor/Documents/Trabalho de ALN/dataset/train'  #Pasta de treinamento.
IMG_SIZE = (100, 100)  # Define o tamanho das imagens de entrada (já pré-processada no dataset.py)
BATCH_SIZE = 32        #Tamanho do lote (quantas imagens processamos de cada vez).
EPOCHS = 50            #Número de épocas (quantas vezes o modelo vai passar por todo o dataset).
LEARNING_RATE = 0.001  #Taxa de aprendizado (quão rápido o modelo aprende).
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #Se possível, usar GPU (CUDA), senão usa CPU

class FaceRecognitionModel(nn.Module): # Definimos a arquitetura do modelo de deep learning propriamente dita
    def __init__(self, num_classes):
        super(FaceRecognitionModel, self).__init__()
        # Usamos um modelo pré-treinado ResNet18 como base
        self.model = models.resnet18(pretrained=True)
        
        # Modificamos a última camada para se adequar ao número de classes do nosso dataset
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),   
            nn.ReLU(),          
            nn.Dropout(0.5),                # Dropout para evitar que o modelo se sobrecarregue com os dados de treinamento
            nn.Linear(256, num_classes)     # Camada final com o número de classes do dataset
        )
        
    def forward(self, x):
        return self.model(x)


def train_model(): # Função principal para treinar o modelo de reconhecimento facial
    # Checa se os dados estão organizados em subpastas (easy, medium, hard, etc).
    if any(os.path.isdir(os.path.join(TRAIN_DIR, d)) and d in ['easy', 'medium', 'hard', 'very-easy', 'extras'] 
           for d in os.listdir(TRAIN_DIR)):
        print("Loading data from difficulty level folders...")
        train_loader, dataset = load_difficulty_levels(TRAIN_DIR, IMG_SIZE, batch_size=BATCH_SIZE)  # Se sim: carrega os dados organizados por dificuldade
    else:
        print("Loading data from flat directory structure...")
        train_loader, dataset = load_dataset(TRAIN_DIR, IMG_SIZE, batch_size=BATCH_SIZE) # Se não: carrega os dados organizados em uma única pasta
    
    num_classes = dataset.num_classes
    print(f"Loaded {len(dataset)} training samples across {num_classes} classes")
    
    # Divide dataset em treino e validação
    train_size = int(0.8 * len(dataset)) # Dividimos 80% dos dados para treino
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Inicializa o modelo
    model = FaceRecognitionModel(num_classes)
    model = model.to(DEVICE)
    
    # Função de perda e otimizador
    criterion = nn.CrossEntropyLoss() # Função de perda para classificação multiclasse
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # Otimizador Adam para atualizar os pesos do modelo
    
    # Listas para armazenar as perdas e acurácias durante o treinamento
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    # Loop de treinamento
    best_val_acc = 0.0
    for epoch in range(EPOCHS): 
        # Início da época de treinamento
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:  # Carrega os dados de treinamento (lotes de imagens e rótulos)
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            
            optimizer.zero_grad()
            
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Calcula as predições e a perda
            loss.backward()
            optimizer.step()
            
            # Atualiza as métricas de perda e acurácia
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)
        
        # Início da época de validação
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad(): 
            for inputs, labels in val_loader: # Carrega os dados de validação
                inputs = inputs.to(DEVICE)    # Move os dados para o dispositivo (GPU ou CPU)
                labels = labels.to(DEVICE)    # Move os rótulos para o dispositivo
                
                outputs = model(inputs)       # Faz a predição com o modelo
                loss = criterion(outputs, labels)       # Calcula a perda
                
                val_loss += loss.item()              # Acumula a perda de validação
                _, predicted = torch.max(outputs.data, 1)  # Obtém as predições do modelo
                total += labels.size(0)                 # Total de amostras de validação
                correct += (predicted == labels).sum().item()   # Conta quantas predições estão corretas
        
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = correct / total
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        
        print(f'Epoch {epoch+1}/{EPOCHS}, '
              f'Train Loss: {epoch_loss:.4f}, '
              f'Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, '
              f'Val Acc: {epoch_val_acc:.4f}')
        
        # Salva o modelo se a acurácia de validação melhorar
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_facial_model.pth')
            print(f'Model saved with validation accuracy: {best_val_acc:.4f}')
    
    # Salva o modelo final
    torch.save(model.state_dict(), 'facial_recognition_model.pth')
    
    # Gera gráficos de perda e acurácia
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_accs)
    plt.plot(val_accs)
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    return model

if __name__ == "__main__":  # Inicia o treinamento ao rodar o script
    train_model()