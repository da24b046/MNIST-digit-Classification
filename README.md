[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/R05VM8Rg)

# MNIST Digit Classification

## 📌 Purpose

This repository contains a complete implementation of **MNIST digit classification** using ensemble learning techniques. The project implements and combines three machine learning algorithms - Softmax Regression, XGBoost, and k-Nearest Neighbors (kNN) with PCA dimensionality reduction - to achieve high classification accuracy on the MNIST dataset.

---

## 🔍 Project Overview

This project classifies handwritten digits (0-9) from the MNIST dataset using an ensemble approach. The solution combines predictions from multiple models using a voting mechanism to improve classification performance.

**Key Algorithms Implemented:**
- **Softmax Regression** - Multi-class logistic regression with gradient descent
- **XGBoost** - Gradient boosting with decision stumps for multi-class classification
- **kNN with PCA** - k-Nearest Neighbors classifier with Principal Component Analysis for feature reduction

---

## 📁 Repository Structure

```
MNIST-digit-Classification/
├── main.py                          # Main execution script for ensemble prediction
├── algorithms.py                    # Implementation of all ML algorithms
├── testcases.ipynb                  # Jupyter notebook with test cases and analysis
├── MNIST_train.csv                  # Training dataset (47,000 samples)
├── MNIST_validation.csv             # Validation dataset (10,000 samples)
├── REPORT.pdf                       # Detailed project report with results and analysis
└── README.md                         # This file
```

### File Descriptions

- **main.py**: Orchestrates the training and prediction pipeline. Trains all three models on the MNIST data, generates predictions, and computes evaluation metrics.
- **algorithms.py**: Contains implementations of:
  - `linear_regression_ovr()` - Linear regression with One-vs-Rest approach
  - `softmax_regression()` - Softmax regression with mini-batch gradient descent
  - `DecisionStumpXGB` - Decision stump class for XGBoost
  - `XGBoostMulticlass` - Multi-class XGBoost classifier
  - `PCAModel` - Principal Component Analysis for dimensionality reduction
  - `KNNClassifier` - k-Nearest Neighbors classifier
- **testcases.ipynb**: Comprehensive Jupyter notebook containing exploratory data analysis, model testing, and visualization of results.

---

## 📦 Installation & Dependencies

### Requirements
- Python 3.7+
- NumPy
- Pandas
- scikit-learn
- Jupyter Notebook (for testcases.ipynb)

### Setup

```bash
# Clone the repository
git clone https://github.com/da24b046/MNIST-digit-Classification.git
cd MNIST-digit-Classification

# Install required packages
pip install numpy pandas scikit-learn jupyter
```

---

## ▶️ Running the Code

### A. Command-line (Recommended for Grading)

Run the main ensemble classification:

```bash
python main.py
```

This will:
1. Load the MNIST training and validation datasets
2. Train Softmax Regression model
3. Train XGBoost classifier
4. Train kNN classifier with PCA dimensionality reduction
5. Combine predictions using voting ensemble
6. Output evaluation metrics: F1-Score, Accuracy, Precision, Recall, and Training Time

### B. Jupyter Notebook

For interactive exploration and testing:

```bash
jupyter notebook testcases.ipynb
```

---

## 📊 Model Details

### Softmax Regression
- **Mini-batch size**: 56
- **Learning rate**: 0.01
- **Epochs**: 30
- **Classes**: 10 (digits 0-9)

### XGBoost
- **Number of estimators**: 40
- **Learning rate**: 0.5
- **Max depth**: 3
- **L2 regularization**: 1.0
- **Gamma**: 0.1

### kNN with PCA
- **PCA components**: 112
- **k neighbors**: 6

### Ensemble Method
Uses **majority voting** across the three models to generate final predictions.

---

## 🧾 Authors

**Roll No.** (2025–26), IIT Madras

---

## 📝 Best Practices Followed

- ✅ Meaningful commit messages documenting code evolution
- ✅ Modularized code with clear separation of algorithms
- ✅ Well-commented functions and classes
- ✅ Reproducible results with fixed random seeds
- ✅ Comprehensive evaluation metrics
- ✅ Detailed documentation and project report

---

## 📄 Results

The ensemble model achieves strong performance metrics on the MNIST validation set. Detailed results and analysis are available in `REPORT.pdf`.

