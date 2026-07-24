# CPELX230-Week02-Assessment---Cuison
# Three-Layer ANN/MLP for Fashion-MNIST Classification

A PyTorch implementation of a Three-Layer Artificial Neural Network (Multi-Layer Perceptron) for image classification on the Fashion-MNIST dataset. This project demonstrates the complete machine learning workflow, including data preprocessing, model design, training, evaluation, hyperparameter comparison, and computational constraint analysis.

---

## 📖 Project Overview

This project was developed as part of **CPELX230 – CPE Elective 2**.

The objective is to design and implement a fully connected neural network consisting of **three trainable linear layers** to classify grayscale clothing images from the Fashion-MNIST dataset. In addition to model implementation, the project evaluates different hyperparameter configurations and analyzes the trade-offs between model performance, computational cost, and training efficiency.

---

## 🎯 Learning Objectives

- Build a Three-Layer Multi-Layer Perceptron (MLP) using PyTorch.
- Understand how image data is flattened before entering a neural network.
- Calculate trainable parameters and memory requirements.
- Train and evaluate an MLP for image classification.
- Compare multiple hyperparameter configurations.
- Analyze computational constraints such as training time, inference time, and parameter count.
- Interpret learning curves and confusion matrices.

---

## 🧠 Model Architecture

The implemented model follows the required architecture below:

```
Input Image (28 × 28)

        │
        ▼

Flatten

784 Features

        │
        ▼

Linear Layer
784 → 128

        │
      ReLU

        ▼

Linear Layer
128 → 64

        │
      ReLU

        ▼

Linear Layer
64 → 10

        │

Raw Logits

        ▼

CrossEntropyLoss
```

### Network Summary

| Layer | Input | Output |
|--------|------:|-------:|
| Flatten | 28 × 28 | 784 |
| Linear 1 | 784 | 128 |
| ReLU | 128 | 128 |
| Linear 2 | 128 | 64 |
| ReLU | 64 | 64 |
| Linear 3 | 64 | 10 |

---

## 📂 Dataset

The project uses the **Fashion-MNIST** dataset from `torchvision.datasets`.

Fashion-MNIST contains **70,000 grayscale images** of clothing items.

- 60,000 Training Images
- 10,000 Testing Images

### Classes

| Label | Class |
|-------:|----------------|
| 0 | T-shirt / Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

---

## ⚙️ Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab
- Jupyter Notebook

---

## 📊 Features

This project includes:

- Data preprocessing
- Fashion-MNIST visualization
- Three-Layer MLP implementation
- Forward propagation
- Backpropagation
- Model training
- Accuracy evaluation
- Loss visualization
- Confusion matrix
- Sample predictions
- Hyperparameter comparison
- Computational constraint analysis

---

## 🧪 Hyperparameter Experiments

Four controlled experiments were performed.

| Run | Optimizer | Activation | Hidden Layer | Learning Rate |
|-----|-----------|------------|--------------|--------------|
| A | SGD | ReLU | 128 | 0.01 |
| B | Adam | ReLU | 128 | 0.001 |
| C | Adam | ReLU | 256 | 0.001 |
| D | Adam | Tanh | 128 | 0.001 |

Each experiment keeps the same:

- Dataset
- Random Seed
- Number of Epochs
- Evaluation Method

Only one hyperparameter changes per experiment to ensure a fair comparison.

---

## 📈 Results

The notebook generates the following outputs:

- Sample Fashion-MNIST images
- Model architecture
- Parameter count
- Tensor shapes
- Training Loss Curve
- Validation Loss Curve
- Accuracy Curve
- Final Test Accuracy
- Macro F1 Score
- Confusion Matrix
- Sample Predictions
- Hyperparameter Comparison Table

---

## 🚀 Installation

Clone the repository.

```bash
git clone https://github.com/yourusername/fashion-mnist-three-layer-mlp.git
```

Move into the project folder.

```bash
cd fashion-mnist-three-layer-mlp
```

Install the required packages.

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Open the notebook Google Colab.

Then run all notebook cells from top to bottom.

---

## 📚 Key Concepts Demonstrated

This project demonstrates several fundamental machine learning concepts:

- Artificial Neural Networks (ANN)
- Multi-Layer Perceptrons (MLP)
- Forward Propagation
- Backpropagation
- CrossEntropy Loss
- Gradient Descent
- Adam Optimizer
- ReLU and Tanh Activation Functions
- Hyperparameter Tuning
- Model Evaluation
- Confusion Matrix Analysis

---

## 💡 Engineering Insights

This project highlights the trade-offs involved in neural network design. Increasing the number of hidden neurons generally improves model capacity but also increases computational cost, memory usage, and training time. Hyperparameter comparisons demonstrate how optimizer selection, activation functions, and network size influence convergence speed, stability, and classification performance.

These analyses provide practical insights into selecting efficient neural network configurations for environments with limited computational resources.

---

## 📌 Future Improvements

Possible enhancements include:

- Adding Dropout layers
- Implementing Batch Normalization
- Testing deeper neural network architectures
- Comparing MLP with Convolutional Neural Networks (CNNs)
- Performing automated hyperparameter optimization
- Deploying the trained model as a web application

