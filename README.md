# Exercise 2 of "Intro To Computer Vision" (83551) course in BIU - CNNs, Regularization & RNNs

This repository contains my implementation of advanced neural network components and architectures for the second assignment of the course. 
The project bridges the gap between manual implementations using **NumPy** and modern deep learning using the **PyTorch** framework, covering both Image Classification and Image Captioning.

## Features:
* **Advanced Layers (From Scratch):** Manual implementation of **Batch Normalization** and **Dropout** (forward and backward propagation) to improve training stability and prevent overfitting.
* **CNN Implementation:** Built Convolutional Neural Networks (CNNs) from scratch, including Convolutional layers, Max Pooling, and a 3-layer ConvNet using pure NumPy.
* **PyTorch Integration:** Migrated to PyTorch, implementing models using three levels of abstraction: Barebones tensors, `nn.Module` API, and `nn.Sequential` API.
* **Deep Architecture Design:** Designed and trained a custom deep CNN (VGG-style) to maximize performance on CIFAR-10.
* **Image Captioning (RNN):** Implemented Vanilla Recurrent Neural Networks (RNNs) and word embeddings to generate natural language descriptions for images on the COCO dataset.

## Results:
* **Datasets:** CIFAR-10 (Classification) and MS-COCO (Image Captioning).
* **Performance:** Achieved **83.9% accuracy** on the CIFAR-10 test set using a custom deep PyTorch architecture utilizing Batch Norm, Dropout, and Max Pooling.
* **Captioning:** Successfully trained an RNN model to overfit a small sample of the COCO dataset, generating meaningful captions for images.

## Used:
* Python 3
* NumPy
* PyTorch
* Matplotlib (for visualization)
