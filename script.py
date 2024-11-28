import os
import random
import shutil
import logging
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

# Constants
CLASS_NAMES = ['normal', 'viral', 'covid']
ROOT_DIR = './COVID-19RadiographyDatabase'
SOURCE_DIRS = ['NORMAL', 'Viral Pneumonia', 'COVID-19']
TEST_SIZE = 30

# Dataset Preparation
def prepare_dataset():
    """Prepare dataset by reorganizing and splitting into train and test sets."""
    try:
        if os.path.isdir(os.path.join(ROOT_DIR, SOURCE_DIRS[0])):
            test_dir = os.path.join(ROOT_DIR, 'test')
            os.makedirs(test_dir, exist_ok=True)

            # Rename source directories to class names
            for i, d in enumerate(SOURCE_DIRS):
                os.rename(os.path.join(ROOT_DIR, d), os.path.join(ROOT_DIR, CLASS_NAMES[i]))

            # Create test directories for each class
            for class_name in CLASS_NAMES:
                os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

            # Move TEST_SIZE images to the test set
            for class_name in CLASS_NAMES:
                images = [x for x in os.listdir(os.path.join(ROOT_DIR, class_name)) if x.lower().endswith('png')]
                selected_images = random.sample(images, TEST_SIZE)
                for image in selected_images:
                    source_path = os.path.join(ROOT_DIR, class_name, image)
                    target_path = os.path.join(test_dir, class_name, image)
                    shutil.move(source_path, target_path)

            logging.info("Dataset prepared successfully.")
    except Exception as e:
        logging.error(f"Error preparing dataset: {e}")


class ChestXRayDataset(Dataset):
    def __init__(self, image_dirs: Dict[str, str], transform: transforms.Compose):
        """Initialize dataset with image directories and transformations."""
        def get_images(class_name: str) -> List[str]:
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('png')]
            logging.info(f"Found {len(images)} {class_name} examples.")
            return images

        self.images = {class_name: get_images(class_name) for class_name in CLASS_NAMES}
        self.image_dirs = image_dirs
        self.transform = transform

    def __len__(self) -> int:
        return sum(len(images) for images in self.images.values())

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        class_name = random.choice(CLASS_NAMES)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), CLASS_NAMES.index(class_name)


# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Data Directories
train_dirs = {name: os.path.join(ROOT_DIR, name) for name in CLASS_NAMES}
test_dirs = {name: os.path.join(ROOT_DIR, 'test', name) for name in CLASS_NAMES}

# Datasets and DataLoaders
train_dataset = ChestXRayDataset(train_dirs, train_transform)
test_dataset = ChestXRayDataset(test_dirs, test_transform)

batch_size = 6
dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
dl_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

logging.info(f'Number of training batches: {len(dl_train)}')
logging.info(f'Number of test batches: {len(dl_test)}')

# Visualization
def show_images(images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor):
    plt.figure(figsize=(12, 6))
    for i, image in enumerate(images):
        plt.subplot(1, batch_size, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green' if preds[i] == labels[i] else 'red'
        plt.xlabel(f'{CLASS_NAMES[int(labels[i])]}')
        plt.ylabel(f'{CLASS_NAMES[int(preds[i])]}', color=col)
    plt.tight_layout()
    plt.show()


# Model Setup
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Linear(in_features=512, out_features=len(CLASS_NAMES))
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet18.parameters(), lr=3e-5)


def show_preds():
    resnet18.eval()
    images, labels = next(iter(dl_test))
    outputs = resnet18(images)
    _, preds = torch.max(outputs, 1)
    show_images(images, labels, preds)


def train(epochs: int):
    logging.info('Starting training...')
    for e in range(epochs):
        logging.info(f'Epoch {e + 1}/{epochs}')
        train_loss = 0.

        resnet18.train()
        for train_step, (images, labels) in enumerate(dl_train):
            optimizer.zero_grad()
            outputs = resnet18(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if train_step % 20 == 0:
                validate()

        train_loss /= (train_step + 1)
        logging.info(f'Training Loss: {train_loss:.4f}')
    logging.info('Training complete.')
    # Saving the model after training
    torch.save(resnet18.state_dict(), 'model.pth')
    logging.info('Model saved to model.pth')

def validate():
    val_loss = 0.
    accuracy = 0.
    resnet18.eval()

    with torch.no_grad():
        for val_step, (images, labels) in enumerate(dl_test):
            outputs = resnet18(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            accuracy += (preds == labels).sum().item()

    val_loss /= len(dl_test)
    accuracy /= len(test_dataset)
    logging.info(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

    show_preds()

# Prepare and Train
prepare_dataset()
train(epochs=1)
