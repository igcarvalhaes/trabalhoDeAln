# Face Recognition System

This project implements a facial recognition system using deep learning techniques. The system is designed to train a neural network model on a dataset of facial images and validate its performance on a separate validation set.

## Project Structure

```
face-recognition-system
├── src
│   ├── train.py          # Contains the training logic for the model
│   ├── validate.py       # Handles the validation process and accuracy calculation
│   ├── dataset.py        # Manages loading and preprocessing of datasets
│   ├── model.py          # Defines the neural network architecture
│   ├── utils.py          # Contains utility functions for preprocessing and metrics
│   └── types
│       └── __init__.py   # Defines custom types or interfaces
├── requirements.txt       # Lists project dependencies
├── README.md              # Documentation for the project
└── config.yaml            # Configuration settings for paths and parameters
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd face-recognition-system
   ```

2. **Install dependencies**:
   Use the following command to install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Configure paths**:
   Update the `config.yaml` file with the correct paths to your training and validation datasets.

## Usage

- To train the model, run:
  ```
  python src/train.py
  ```

- To validate the model, run:
  ```
  python src/validate.py
  ```

## Dataset

The dataset is organized into training and validation folders, with subfolders for different difficulty levels (easy, medium, hard, etc.). Ensure that the dataset is structured as specified in the project.

## Model

The model architecture is defined in `src/model.py`. You can modify the architecture and training parameters in the `config.yaml` file to experiment with different configurations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.