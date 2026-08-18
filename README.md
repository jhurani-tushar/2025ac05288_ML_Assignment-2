# Handwritten Digit Classification — ML Assignment 2

## a. Problem Statement

Can a machine correctly identify handwritten digits (0 through 9) from a small 8×8 grayscale image? This project tackles that question by training and comparing five classical machine learning classifiers on the **Optical Recognition of Handwritten Digits** dataset from the UCI Machine Learning Repository.

The motivation is straightforward — digit recognition is one of the foundational problems in pattern recognition, and comparing multiple models on the same data gives a practical sense of which algorithms suit which kinds of feature spaces.

---

## b. Dataset Description

| Property | Details |
|---|---|
| **Name** | Optical Recognition of Handwritten Digits |
| **Source** | UCI Machine Learning Repository |
| **URL** | https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits |
| **Samples** | 1 797 |
| **Features** | 64 (each an integer 0–16, representing pixel intensity in an 8×8 grid) |
| **Target** | `digit_label` — integer 0 to 9 |
| **Classes** | 10 (approximately 180 samples per class — well balanced) |
| **Missing values** | None |

Each sample is an 8×8 image of a handwritten digit. The 64 pixel values were originally obtained by dividing 32×32 bitmaps from NIST into non-overlapping 4×4 blocks and counting the number of on-pixels in each block. This gives integer values between 0 and 16.

**Train/Test Split:** 75 / 25 stratified (random_state = 101)
- Training: 1 347 samples
- Test: 450 samples

---

## c. GitHub Repository Link

**Repo:** [https://github.com/jhurani-tushar/2025ac05288_ML_Assignment-2](https://github.com/jhurani-tushar/2025ac05288_ML_Assignment-2)

```
digit-classification-ml/
├── app.py                          # Streamlit web application
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── test_data.csv                   # 450-row test split
└── model/
    ├── train_models.py             # Standalone training & evaluation script
    ├── scaler.pkl                  # Fitted MinMaxScaler
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest_ensemble.pkl
    └── model_results.csv
```

---

## d. Models Used

All five models were trained on the same 75 % training split. Features were normalised to [0, 1] with `MinMaxScaler`. Because this is a 10-class problem, Precision, Recall, and F1 are **weighted averages** and AUC uses the **one-vs-rest** strategy.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9733 | 0.9994 | 0.9740 | 0.9733 | 0.9734 | 0.9704 |
| Decision Tree | 0.8400 | 0.9368 | 0.8446 | 0.8400 | 0.8406 | 0.8226 |
| kNN | 0.9889 | 0.9999 | 0.9891 | 0.9889 | 0.9889 | 0.9877 |
| Naive Bayes | 0.8578 | 0.9798 | 0.8850 | 0.8578 | 0.8614 | 0.8447 |
| Random Forest (Ensemble) | 0.9711 | 0.9995 | 0.9725 | 0.9711 | 0.9710 | 0.9681 |

### Hyperparameters

| Model | Key settings |
|---|---|
| Logistic Regression | C = 0.8, solver = lbfgs, max_iter = 3 000 |
| Decision Tree | max_depth = 12, min_samples_leaf = 4, criterion = gini |
| kNN | n_neighbors = 3, weights = distance, p = 2 (Euclidean) |
| Naive Bayes | Gaussian, var_smoothing = 1e-8 |
| Random Forest | n_estimators = 200, max_depth = 20, min_samples_leaf = 2 |

---

### Observations

| ML Model Name | Observation |
|---|---|
| **Logistic Regression** | Performs remarkably well for a linear model on what is essentially an image recognition task. The 64-dimensional pixel space is sparse enough that a multinomial logistic boundary separates the classes with just 12 misclassifications out of 450 test samples. Feature normalisation is important here — without it convergence was slow and accuracy dropped by ~3 %. |
| **Decision Tree** | Weakest of the five by a wide margin. Even with depth capped at 12 to control overfitting, the tree makes 72 errors. The core issue is that axis-aligned splits on pixel values are brittle — a digit shifted even slightly produces different pixel patterns the tree hasn't learned. The low AUC (0.94 vs 0.99+ for others) confirms poor probability calibration. |
| **kNN** | Best overall performer on every metric. With k = 3 and distance-based weighting, kNN leverages the fact that similar-looking digits lie close together in 64-D pixel space. Only 5 out of 450 test samples were wrong. The near-perfect AUC (0.9999) means almost every sample is ranked correctly by its class probability. The practical downside is slow inference since it compares each new sample against all training data. |
| **Naive Bayes** | Second weakest with 64 errors. The Gaussian assumption is a poor fit for pixel data, which tends to be bimodal (mostly zero background or high-intensity foreground). Despite this, its precision (0.885) is notably higher than its recall (0.858), meaning when it does commit to a class it's more often correct — but it frequently confuses visually similar pairs like 3/9 and 1/8. |
| **Random Forest (Ensemble)** | Nearly matches Logistic Regression (13 errors vs 12) and hugely outperforms the single Decision Tree (72 errors → 13). The ensemble of 200 trees eliminates the variance that cripples a lone tree. AUC of 0.9995 shows it is almost perfectly calibrated. A solid choice when you want a reliable "just works" classifier with no tuning. |
| **Overall Winner** | **kNN** is the clear winner across all six metrics. The handwritten-digit data has well-separated clusters in the normalised pixel space, which plays directly to an instance-based learner's strengths. In a production setting where inference latency matters, Logistic Regression or Random Forest (both ~97 % accuracy) would be more practical alternatives. |

---

## Live Streamlit App

**Deployed at:** [https://2025ac05288mlassignment-2-e7o7pwp2rra6zr9dt62msf.streamlit.app](https://2025ac05288mlassignment-2-e7o7pwp2rra6zr9dt62msf.streamlit.app)

---

## Running Locally

```bash
git clone https://github.com/jhurani-tushar/2025ac05288_ML_Assignment-2.git
cd digit-classification-ml
pip install -r requirements.txt
streamlit run app.py
```

To retrain models independently:
```bash
cd model
python train_models.py
```

---

## Tech Stack

Python 3.10 · Streamlit · scikit-learn · pandas · NumPy · matplotlib · seaborn
