# Sistema de Reconhecimento Facial

## Visão Geral

Este projeto implementa um sistema de reconhecimento facial utilizando técnicas de aprendizado profundo. O sistema foi projetado para treinar um modelo de rede neural em um conjunto de dados de imagens faciais e validar seu desempenho em diferentes níveis de dificuldade. O projeto utiliza PyTorch como framework principal de deep learning e incorpora técnicas modernas de processamento de imagem e classificação.

## Estrutura do Projeto

```
face-recognition-system/
├── src/                           # Código-fonte do projeto
│   ├── train.py                   # Script para treinamento do modelo
│   ├── validate.py                # Script para validação e avaliação do modelo
│   ├── dataset.py                 # Gerenciamento de dados e carregamento de imagens
│   └── model_types/               # Definições de arquiteturas de modelo
├── best_facial_model.pth          # Modelo treinado com melhor desempenho
├── facial_recognition_model.pth   # Versão atual do modelo treinado
├── confusion_matrix.png           # Visualização da matriz de confusão
├── classification_report.txt      # Relatório de métricas de classificação
├── training_history.png           # Gráfico do histórico de treinamento
└── requirements.txt               # Dependências do projeto
```

## Conjunto de Dados

O conjunto de dados está organizado em duas pastas principais:

```
train/                             # Conjunto de treinamento
├── easy/                          # Imagens de dificuldade fácil
├── medium/                        # Imagens de dificuldade média
├── hard/                          # Imagens de dificuldade difícil
├── very-easy/                     # Imagens de dificuldade muito fácil
└── extras/                        # Imagens adicionais

valid/                             # Conjunto de validação
├── easy/
├── medium/
├── hard/
├── very-easy/
└── extras/
```

Cada subdiretório contém imagens faciais no formato "ID-número.jpg", onde "ID" é o identificador único da pessoa (por exemplo, "11-1.jpg", "11-2.jpg" para a pessoa ID 11).

## Componentes Principais

### dataset.py

Este módulo gerencia o carregamento, pré-processamento e organização das imagens faciais:

- `FaceDataset`: Classe que herda de `torch.utils.data.Dataset` para integração com PyTorch
- `load_dataset()`: Carrega dados de um diretório simples
- `load_difficulty_levels()`: Carrega dados de múltiplos níveis de dificuldade

Funcionalidades:

- Redimensionamento de imagens para tamanho padrão (100x100 pixels)
- Normalização de pixel para valores entre 0 e 1
- Extração automática de IDs de pessoa a partir dos nomes dos arquivos
- Mapeamento de IDs para índices numéricos de classe

### train.py

Implementa o processo de treinamento do modelo:

- Define a arquitetura do modelo (`FaceRecognitionModel`)
- Configura hiperparâmetros (taxa de aprendizagem, épocas, etc.)
- Implementa o loop de treinamento e validação
- Salva o modelo treinado e registra métricas de desempenho

### validate.py

Gerencia a avaliação do modelo treinado:

- Carrega o modelo salvo
- Avalia o desempenho em conjuntos de dados de validação
- Calcula métricas como acurácia, precisão, recall e F1-score
- Gera uma matriz de confusão e relatório de classificação
- Salva resultados e visualizações para análise

## Modelo de Rede Neural

O projeto utiliza uma arquitetura baseada em Redes Neurais Convolucionais (CNN), específicamente:

- Camadas convolucionais para extração de características faciais
- Normalização em lote para estabilidade de treinamento
- Camadas de pooling para redução de dimensionalidade
- Dropout para prevenção de overfitting
- Camadas totalmente conectadas para classificação final

## Instruções de Configuração

1. **Clone o repositório**:

   ```
   git clone <repository-url>
   cd face-recognition-system
   ```

2. **Instale as dependências**:

   ```
   pip install -r requirements.txt
   ```

   Principais dependências:

   - torch
   - torchvision
   - numpy
   - opencv-python
   - matplotlib
   - seaborn
   - scikit-learn

3. **Configure os caminhos no código**:
   Atualize os caminhos para os diretórios de treinamento e validação nos arquivos `train.py` e `validate.py` conforme sua configuração local.

## Uso

### Treinamento do Modelo

Para treinar o modelo do zero:

```
python src/train.py
```

Opções e parâmetros:

- O treinamento salvará automaticamente o modelo em `facial_recognition_model.pth`
- O melhor modelo (com menor perda na validação) será salvo em `best_facial_model.pth`
- O histórico de treinamento será visualizado e salvo em `training_history.png`

### Validação do Modelo

Para avaliar o desempenho do modelo treinado:

```
python src/validate.py
```

Resultados:

- Acurácia global e por classe
- Matriz de confusão salva em `confusion_matrix.png`
- Relatório detalhado de classificação em `classification_report.txt`

## Resultados e Avaliação

O sistema gera vários artefatos de avaliação:

1. **Matriz de Confusão**: Visualização gráfica mostrando quais classes estão sendo confundidas entre si.

2. **Relatório de Classificação**: Documento detalhado com métricas por classe:

   - Precisão: capacidade do modelo de não classificar incorretamente outras faces
   - Recall: capacidade do modelo de encontrar todas as instâncias de uma face
   - F1-score: média harmônica entre precisão e recall
   - Support: número de ocorrências de cada classe

3. **Gráfico de Histórico de Treinamento**: Visualização da perda e acurácia ao longo do treinamento.

## Expandindo o Projeto

Para adaptar o projeto para seus próprios dados:

1. Organize suas imagens seguindo a estrutura de pastas mostrada acima
2. Ajuste o tamanho das imagens em `dataset.py` se necessário
3. Modifique a arquitetura do modelo em `train.py` para seus requisitos específicos
4. Ajuste hiperparâmetros diretamente no código-fonte dos arquivos respectivos

## Créditos

Este projeto é um trabalho para a disciplina Algebra Linear Numérica, ministrada pelo prof. Ricardo Fabbri.

## Licença

Este projeto está licenciado sob a Licença MIT - consulte o arquivo LICENSE
