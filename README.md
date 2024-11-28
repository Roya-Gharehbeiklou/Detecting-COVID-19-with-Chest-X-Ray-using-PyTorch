# Chest X-Ray Classification using ResNet18

This repository contains a project for classifying chest X-ray images into three categories: `normal`, `viral pneumonia`, and `COVID-19`. It leverages PyTorch and a pretrained ResNet18 model for transfer learning. The dataset used is organized and processed from the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).

---

## Features
- **Dataset Preparation**: Automates the organization and splitting of the dataset into training and test sets.
- **Transformations**: Applies image preprocessing and augmentation using PyTorch's `torchvision.transforms`.
- **Training**: Implements a training loop with logging for losses and saves the model.
- **Validation**: Includes validation during training with visual feedback on predictions.
- **Inference**: Provides a script for predicting the class of a single chest X-ray image.

---

## Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.9.0+
- torchvision
- matplotlib
- PIL (Pillow)


## Usage

### Dataset Preparation
The dataset should be downloaded and placed in the root directory of the project as `COVID-19RadiographyDatabase`. Run the `script.py` file to prepare the dataset:

```bash
python script.py
```

This script:
- Renames subdirectories for better compatibility.
- Splits the dataset into training and test sets.

### Training
To train the model, run:
```bash
python script.py
```

This will:
1. Train the ResNet18 model on the training set.
2. Validate the model and log results.
3. Save the trained model as `model.pth`.

### Prediction
To predict the class of a single X-ray image, use the `test.py` script:

```bash
python test.py
```

Edit the script's `predict` function or provide a path to the desired image file for prediction.

---

## Files Overview

- **`script.py`**: Main script for dataset preparation, training, and validation.
- **`test.py`**: Script for loading a saved model and predicting the class of a single image.

---

## Dataset Preparation Details

- Classes:
  - `normal`
  - `viral`
  - `covid`

- Transformations:
  - Resize images to `224x224`.
  - Normalize using ImageNet's mean and standard deviation.
  - Apply random horizontal flips (training only).

---

- The dataset is sourced from [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).

---

For any issues, please open an issue or contact [your-email@example.com].

