# Brain Tumor Classification Using Convolutional Neural Networks

## Project Overview
This project implements a deep learning solution for automated brain tumor classification using MRI scans. The system utilizes a Convolutional Neural Network (CNN) architecture to classify brain MRI images into four categories: no tumor, glioma tumor, meningioma tumor, and pituitary tumor. This project was developed as part of the CS 453 Data Mining course at the University of Oregon.

## Problem Statement
Brain tumors can have devastating consequences if not identified swiftly, causing various medical conditions including loss of sensation, headaches, nausea, seizures, and even death. Traditional detection relies heavily on manual analysis by radiologists, which can be time-consuming and subject to human error. Our research leverages CNNs to develop an efficient automated classification system that can assist medical professionals in the diagnosis process, potentially reducing diagnosis time while maintaining high accuracy.

## Dataset
The project uses the "Brain Tumor MRI Dataset" from Kaggle, which contains 7,023 MRI scan images organized into four categories:
- No Tumor
- Glioma Tumor (cancerous, originating from glial cells)
- Meningioma Tumor (non-cancerous, originating from the membrane surrounding the brain)
- Pituitary Tumor

The dataset is a combination of three other datasets: figshare, SARTAJ dataset, and Br35H. Our implementation uses 5,712 images for training and 1,311 images for testing and validation.

## Technical Requirements
- Python 3.8+
- PyTorch
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
  <git clone git@github.com:WillMarceau/cs453_project.git>
2. Install required packages:
  <pip install -r requirements.txt>

## Project Structure
brain-tumor-classification/
│
├── data/                     # Dataset directory
│   ├── training/            # Training data
│   │   ├── glioma/         # Glioma tumor images
│   │   ├── meningioma/     # Meningioma tumor images
│   │   ├── notumor/        # No tumor images
│   │   └── pituitary/      # Pituitary tumor images
│   ├── validation/          # Validation data (same subfolders as training)
│   └── testing/             # Test data (same subfolders as training)
│
├── src/                     # Source code
│   ├── architectureCNN.py  # CNN model architecture definition
│
├── results/                 # Trained models and results
├── runs/                    # Run logs
│
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
└── .gitignore              # Git ignore file

## Model Architecture
The project implements a Convolutional Neural Network (CNN) with the following key features:

### CNN Layers:
- Input: RGB images (3 channels)
- First Convolutional Layer: 8 filters with 3×3 kernel
- Batch Normalization and ReLU Activation
- Dropout (0.3) for regularization
- Max Pooling (3×3, stride 3)
- Second Convolutional Layer: 16 filters with 3×3 kernel
- Batch Normalization and ReLU Activation
- Dropout (0.3) for regularization
- Max Pooling (3×3, stride 4)

### Dense Layers:
- Flatten Layer: Converts 2D feature maps to 1D feature vectors
- First Dense Layer: 5184 → 512 nodes with ReLU activation
- Dropout (0.4) for regularization
- Output Layer: 512 → 4 nodes (one for each class)

### Training Configuration:
- Loss Function: Weighted Cross-Entropy Loss (adjusted for class imbalance)
- Optimizer: Adam with learning rate 0.001 and weight decay 1e-4
- Data Augmentation: Applied using PyTorch transformers for better generalization

## Data Preprocessing
The preprocessing pipeline includes:
- Image resizing to ensure consistent dimensions
- Data augmentation techniques (flipping, scaling, etc.)
- Normalization to improve training stability
- Class weighting to handle imbalanced class distribution

## Performance Metrics
The model's performance is evaluated using:
- Overall Accuracy
- Recall (particularly important to minimize false negatives)
- Precision
- Confusion Matrix

## Advantages of Our Approach
- Balances model sophistication with practical utility
- Employs data augmentation to improve generalization with limited data
- Implements regularization techniques to prevent overfitting
- Designed for deployment in resource-constrained clinical settings

## Team Members
Amanda Hoelting, Ellison Largent, & Will Marceau

## License
This project is for educational purposes as part of the CS 453 Data Mining course at the University of Oregon. All rights reserved.

## Acknowledgments
- Dataset source: Kaggle user Masoud Nickparvar
- Course Instructor: Yu Wang
- Data Mining Course, University of Oregon
