import os           # manipulacao de diretorios e arquivos
import cv2          # leitura e manipulacao de imagens
import numpy as np  # operacoes numericas nos arrays
import torch        # framework de deep learning (PyTorch)
import re           # expressões regulares para extrair IDs de arquivos (e.g., "11-1.jpg" -> "11")
from torch.utils.data import Dataset, DataLoader # permitem organizar os dados para o treinamento
from torchvision import transforms    # transformações de imagens para normalização e pré-processamento

class FaceDataset(Dataset):
    # Aqui é criada uma classe que herda de Dataset do PyTorch, o que permite integrar as imagens diretamente com o treinamento.
    def __init__(self, directory, img_size=(100, 100), transform=None):
        # Inicializa o dataset com a pasta que contém imagens, tamanho da imagem, transformações e outras variáveis necessárias.
        self.directory = directory
        self.img_size = img_size
        self.transform = transform
        self.images = []  # Lista para armazenar os caminhos das imagens
        self.labels = []  # Lista para armazenar os rótulos (labels) correspondentes
        self.class_names = set() # conjunto com todos os IDs únicos de pessoas.
        self.class_map = {}  # mapeamento dos IDs para índices numéricos.
        
        # Chamada para carregar os dados do diretório.
        self._load_data()
        self.class_names = sorted(list(self.class_names))
        
    def _load_data(self):
        if os.path.isdir(self.directory): # Verifica se o diretório existe
            files = os.listdir(self.directory) # Lista todos os arquivos no diretório
            
            # Percorre todos os arquivos de imagem
            for img_file in files:
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(self.directory, img_file)
                    
                    # Extrai o ID da pessoa a partir do nome da imagem: (e.g., "11-1.jpg" -> "11")
                    match = re.match(r'(\d+)-', img_file)
                    if match:
                        person_id = match.group(1)
                        
                        
                        self.class_names.add(person_id) # Adiciona o ID no conjunto de classes
                        
                        # Verifica se o ID já está mapeado, se não, adiciona ao mapeamento
                        if person_id not in self.class_map:
                            self.class_map[person_id] = len(self.class_map)
                        
                        # Adiciona o caminho da imagem e o rótulo correspondente
                        # ao dataset
                        self.images.append(img_path)
                        self.labels.append(self.class_map[person_id])

    def __len__(self): # Retorna o número total de imagens no dataset
        return len(self.images)
    
    def __getitem__(self, idx): # Retorna uma imagem e seu rótulo correspondente dado um índice
        img_path = self.images[idx]  # Verifica se o índice está dentro dos limites do dataset
        label = self.labels[idx] 
        
        try:
            # Lê a imagem usando OpenCV
            img = cv2.imread(img_path)
            if img is None:
                # Se a imagem não puder ser lida, cria uma imagem vazia (preta)
                img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converte a imagem BGR para RGB
                img = cv2.resize(img, self.img_size)  # Redimensiona a imagem para 100x100 pixels
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        
        # Converte a imagem para o formato esperado pelo PyTorch
        img = img.transpose((2, 0, 1))  # Transforma a imagem para o formato do PyTorch (Canais, Altura, Largura)
        img = img / 255.0  # Normaliza os pixels entre 0 e 1
        img = torch.from_numpy(img).float()
        
        # Aplica transformações adicionais (se existirem)
        if self.transform:
            img = self.transform(img)
            
        return img, label  # Retorna o par imagem e rótulo
        #  Aqui é onde o dado final fica pronto para ser enviado para o modelo.
    
    @property
    def num_classes(self): # Retorna o número de classes (IDs únicos de pessoas) no dataset
        return len(self.class_names)


def load_dataset(directory, img_size=(100, 100), batch_size=32, shuffle=True):
    """
    Cria o dataset com as imagens de um diretório simples (sem subpastas de dificuldade)
    """
    # Define uma normalização nos dados
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Cria o dataset com as imagens do diretório
    # e as transformações definidas
    dataset = FaceDataset(directory, img_size=img_size, transform=transform)
    
    # Cria o DataLoader do PyTorch para facilitar o treinamento
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader, dataset


def load_difficulty_levels(base_dir, img_size=(100, 100), batch_size=32, difficulty_levels=None):
    """
    Cria um dataset combinando imagens de diferentes níveis de dificuldade
    a partir de subpastas dentro de um diretório base.
    Cada subpasta deve conter imagens nomeadas no formato "ID-numero.jpg",
    onde "ID" é o ID da pessoa (e.g., "11-1.jpg" -> "11").
    """
    if difficulty_levels is None:
        difficulty_levels = ['easy', 'medium', 'hard', 'very-easy', 'extras']
    
    all_images = []
    all_labels = []
    class_map = {}
    class_names = set()
    
    # Itera sobre cada nível de dificuldade
    for level in difficulty_levels:
        level_dir = os.path.join(base_dir, level)
        if os.path.isdir(level_dir):
            print(f"Scanning {level} dataset for classes...")
            for img_file in os.listdir(level_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    # Extract person ID from filename (e.g., "11-1.jpg" -> "11")
                    match = re.match(r'(\d+)-', img_file)
                    if match:
                        person_id = match.group(1)
                        class_names.add(person_id)

    # Cria um mapeamento de classes para índices numéricos
    sorted_class_names = sorted(list(class_names))
    for i, name in enumerate(sorted_class_names):
        class_map[name] = i
                        
    # Carrega as imagens e rótulos de cada nível de dificuldade
    for level in difficulty_levels:
        level_dir = os.path.join(base_dir, level)
        if os.path.isdir(level_dir):
            print(f"Loading images from {level} dataset...")
            for img_file in os.listdir(level_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(level_dir, img_file)
                    
                    # Extrai o ID da pessoa a partir do nome da imagem: (e.g., "11-1.jpg" -> "11")
                    match = re.match(r'(\d+)-', img_file)
                    if match:
                        person_id = match.group(1)
                        label = class_map[person_id]
                        
                        all_images.append(img_path)
                        all_labels.append(label)
    
    # Cria o dataset combinado com todas as imagens e rótulos
    # e as classes ordenadas
    combined_dataset = FaceDataset(base_dir, img_size=img_size)
    combined_dataset.images = all_images
    combined_dataset.labels = all_labels
    combined_dataset.class_names = sorted_class_names
    combined_dataset.class_map = class_map
    
    # Cria o DataLoader do PyTorch
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Loaded {len(all_images)} images across {len(sorted_class_names)} classes")
    
    return dataloader, combined_dataset