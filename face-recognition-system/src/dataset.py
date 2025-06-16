import os
import cv2
import numpy as np
import torch
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class FaceDataset(Dataset):
    def __init__(self, directory, img_size=(100, 100), transform=None):
        self.directory = directory
        self.img_size = img_size
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_names = set()
        self.class_map = {}
        
        # Load data
        self._load_data()
        self.class_names = sorted(list(self.class_names))
        
    def _load_data(self):
        # Check if files in directory are directly images
        if os.path.isdir(self.directory):
            files = os.listdir(self.directory)
            
            # Process all files in the directory
            for img_file in files:
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(self.directory, img_file)
                    
                    # Extract person ID from filename (e.g., "11-1.jpg" -> "11")
                    match = re.match(r'(\d+)-', img_file)
                    if match:
                        person_id = match.group(1)
                        
                        # Add to class names set
                        self.class_names.add(person_id)
                        
                        # Map person ID to numerical label if not already mapped
                        if person_id not in self.class_map:
                            self.class_map[person_id] = len(self.class_map)
                        
                        # Add image and label
                        self.images.append(img_path)
                        self.labels.append(self.class_map[person_id])

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            # Read and resize image
            img = cv2.imread(img_path)
            if img is None:
                # Return a black image and the label if image can't be read
                img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                img = cv2.resize(img, self.img_size)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
        
        # Convert numpy array to PyTorch tensor
        img = img.transpose((2, 0, 1))  # Convert from HWC to CHW format
        img = img / 255.0  # Normalize to [0, 1]
        img = torch.from_numpy(img).float()
        
        # Apply additional transforms if specified
        if self.transform:
            img = self.transform(img)
            
        return img, label
    
    @property
    def num_classes(self):
        return len(self.class_names)


def load_dataset(directory, img_size=(100, 100), batch_size=32, shuffle=True):
    """
    Load images from directory and create a PyTorch DataLoader
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = FaceDataset(directory, img_size=img_size, transform=transform)
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader, dataset


def load_difficulty_levels(base_dir, img_size=(100, 100), batch_size=32, difficulty_levels=None):
    """
    Load images from different difficulty levels and combine them into a single dataset
    """
    if difficulty_levels is None:
        difficulty_levels = ['easy', 'medium', 'hard', 'very-easy', 'extras']
    
    all_images = []
    all_labels = []
    class_map = {}
    class_names = set()
    
    # First scan to identify all unique classes across difficulty levels
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

    # Sort class names and create a consistent mapping
    sorted_class_names = sorted(list(class_names))
    for i, name in enumerate(sorted_class_names):
        class_map[name] = i
                        
    # Now load all images with consistent class mappings
    for level in difficulty_levels:
        level_dir = os.path.join(base_dir, level)
        if os.path.isdir(level_dir):
            print(f"Loading images from {level} dataset...")
            for img_file in os.listdir(level_dir):
                if img_file.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(level_dir, img_file)
                    
                    # Extract person ID
                    match = re.match(r'(\d+)-', img_file)
                    if match:
                        person_id = match.group(1)
                        label = class_map[person_id]
                        
                        all_images.append(img_path)
                        all_labels.append(label)
    
    # Create a combined dataset
    combined_dataset = FaceDataset(base_dir, img_size=img_size)
    combined_dataset.images = all_images
    combined_dataset.labels = all_labels
    combined_dataset.class_names = sorted_class_names
    combined_dataset.class_map = class_map
    
    # Create data loader
    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Loaded {len(all_images)} images across {len(sorted_class_names)} classes")
    
    return dataloader, combined_dataset