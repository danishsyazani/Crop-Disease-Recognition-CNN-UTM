# CROP DISEASE RECOGNITION IN MALAYSIAN AGRICULTURE BY USING CNN MODELS (VGG-16, RESNET-50, EfficientNet)

**Researcher:**
Danish Syazani bin Mohd Zakir

**Matric ID:** A21MJ8023

**Program:** SECJH

**Subject name:** SECJ4134-15 (SOFTWARE ENGINEERING PROJECT II)

**University:** Universiti Teknologi Malaysia

**Faculty:** MJIIT

**Date finished:** 26/5/2024

------------------------------------------------------------------------------------------------

## Project Overview

This research project aims to recognize crop diseases in Malaysian agriculture using Convolutional Neural Network (CNN) models, specifically VGG-16, ResNet-50, and EfficientNet. The project leverages the PlantVillage dataset to train and evaluate the models, and a web application is developed to allow users to upload images of potato and tomato leaves for disease prediction.

## Dataset

The dataset used for this project is the PlantVillage dataset, credited to SP Mohanty. The dataset contains images of healthy and diseased plant leaves. The dataset can be found at [this link](https://github.com/spMohanty/PlantVillage-Dataset).

## Preprocessing

The data preprocessing involves:
- Splitting the dataset into training (85%), validation (10%), and testing (5%) sets.
- Renaming the images for consistency.
- Saving the preprocessed images in the `Preprocessing` directory.

## Training

Three CNN models were trained using the Adam optimizer with a learning rate of 0.001:
1. **ResNet-50**
   - Batch size: 64
   - Epochs: 10
2. **VGG-16**
   - Batch size: 16
   - Epochs: 10
3. **EfficientNet**
   - Batch size: 16
   - Epochs: 10

Each model's training progress, loss, and accuracy were recorded and visualized.

## Evaluation

The models were evaluated on the test dataset to gauge their generalization ability. The following metrics were calculated:
- Accuracy
- Precision
- Recall
- F1-Score

Confusion matrices were also generated for a detailed analysis of true positives, false positives, true negatives, and false negatives.

## Web Application

A web application was developed using Flask. The application allows users to:
- Select a model (ResNet-50, VGG-16, or EfficientNet).
- Upload an image of a potato or tomato leaf.
- Receive a prediction of the disease and the health condition of the leaf.

### How to Run the Application

1. **Install Required Packages:**

   ```bash
   pip install -r requirements.txt

2. **Run the Flask Application:**

    python app.py

3. **Access the Application:**

    Open a web browser and go to http://127.0.0.1:8080/.

## Imports Used

The following imports were used in the project:

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from waitress import serve

## Credits
Dataset: PlantVillage dataset by SP Mohanty (GitHub Link)

## Conclusion
This project successfully developed a system for crop disease recognition in Malaysian agriculture using CNN models. The system was evaluated thoroughly and implemented as a user-friendly web application. This work contributes to the field of agricultural technology by providing an effective tool for early disease detection, potentially improving crop yields and sustainability.

Researcher: Danish Syazani bin Mohd Zakir
