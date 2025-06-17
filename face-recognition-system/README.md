# Sistema de Reconhecimento Facial

Este projeto implementa um sistema de reconhecimento facial utilizando técnicas de aprendizado profundo. O sistema foi projetado para treinar um modelo de rede neural em um conjunto de dados de imagens faciais e validar seu desempenho em um conjunto de validação separado.

## Estrutura do Projeto

```
face-recognition-system
├── src
│   ├── train.py          # Contém a lógica de treinamento do modelo
│   ├── validate.py       # Gerencia o processo de validação e cálculo de precisão
│   ├── dataset.py        # Administra o carregamento e pré-processamento dos conjuntos de dados
│   └── types
│       └── __init__.py   # Define tipos ou interfaces personalizadas
├── requirements.txt       # Lista as dependências do projeto
├── README.md              # Documentação do projeto
└── config.yaml            # Configurações para caminhos e parâmetros
```

## Instruções de Configuração

1. **Clone o repositório**:

   ```
   git clone <repository-url>
   cd face-recognition-system
   ```

2. **Instale as dependências**:
   Use o seguinte comando para instalar os pacotes necessários:

   ```
   pip install -r requirements.txt
   ```

3. **Configure os caminhos**:
   Atualize o arquivo `config.yaml` com os caminhos corretos para seus conjuntos de dados de treinamento e validação.

## Uso

- Para treinar o modelo, execute:

  ```
  python src/train.py
  ```

- Para validar o modelo, execute:
  ```
  python src/validate.py
  ```

## Conjunto de Dados

O conjunto de dados é organizado em pastas de treinamento e validação, com subpastas para diferentes níveis de dificuldade (fácil, médio, difícil, etc.). Certifique-se de que o conjunto de dados esteja estruturado conforme especificado no projeto.

## Modelo

A arquitetura do modelo é definida em `src/model.py`. Você pode modificar a arquitetura e os parâmetros de treinamento no arquivo `config.yaml` para experimentar diferentes configurações.

## Licença

Este projeto está licenciado sob a Licença MIT - consulte o arquivo LICENSE para obter detalhes.
