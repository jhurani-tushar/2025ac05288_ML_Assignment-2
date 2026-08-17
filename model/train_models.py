#!/usr/bin/env python3
# -------------------------------------------------------
# train_models.py
# Trains 5 classification models on the Handwritten Digits
# dataset (UCI) and saves results + test data for the app.
# -------------------------------------------------------

import pandas as pd
import numpy as np
import os, joblib, warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)


# ---------- CONFIG ----------
TEST_RATIO = 0.25
SEED = 101
# ----------------------------


def prepare_dataset():
    """
    Loads the digits dataset from sklearn (originally from UCI).
    64 features (8x8 pixel grid), 10 classes (0-9), 1797 samples.
    """
    digits = load_digits()
    
    cols = [f"pixel_{i}" for i in range(64)]
    X_df = pd.DataFrame(digits.data, columns=cols)
    y_series = pd.Series(digits.target, name="digit_label")
    
    print(f"Loaded digits dataset: {X_df.shape[0]} samples, {X_df.shape[1]} features")
    print(f"Number of classes: {len(np.unique(y_series))}")
    print(f"Samples per class:\n{y_series.value_counts().sort_index().to_string()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series,
        test_size=TEST_RATIO,
        random_state=SEED,
        stratify=y_series
    )
    
    # export test csv
    export_df = X_test.copy()
    export_df["digit_label"] = y_test.values
    export_df.to_csv("../test_data.csv", index=False)
    print(f"\nSaved test_data.csv  ({len(X_test)} rows)")
    
    return X_train, X_test, y_train, y_test, cols


def normalise(X_train, X_test):
    """Scale pixel values to [0, 1] using MinMaxScaler."""
    scaler = MinMaxScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")
    return Xtr, Xte, scaler


def build_models():
    """Return dict of model name -> untrained estimator."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, solver="lbfgs",
            C=0.8, random_state=SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=4,
            criterion="gini", random_state=SEED
        ),
        "kNN": KNeighborsClassifier(
            n_neighbors=3, weights="distance", p=2
        ),
        "Naive Bayes": GaussianNB(
            var_smoothing=1e-8
        ),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, max_depth=20,
            min_samples_leaf=2, random_state=SEED
        ),
    }


def score_model(clf, X_te, y_te):
    """
    Compute all 6 evaluation metrics.
    Uses weighted averaging for multi-class precision/recall/F1.
    Uses OVR strategy for multi-class AUC.
    """
    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)
    
    acc  = accuracy_score(y_te, preds)
    auc  = roc_auc_score(y_te, probs, multi_class="ovr", average="weighted")
    prec = precision_score(y_te, preds, average="weighted", zero_division=0)
    rec  = recall_score(y_te, preds, average="weighted", zero_division=0)
    f1   = f1_score(y_te, preds, average="weighted", zero_division=0)
    mcc  = matthews_corrcoef(y_te, preds)
    
    return {
        "Accuracy":  round(acc, 4),
        "AUC":       round(auc, 4),
        "Precision": round(prec, 4),
        "Recall":    round(rec, 4),
        "F1":        round(f1, 4),
        "MCC":       round(mcc, 4),
    }, preds, probs


def main():
    print("=" * 55)
    print("  Handwritten Digit Classification — Model Training")
    print("=" * 55)
    
    X_train, X_test, y_train, y_test, feature_cols = prepare_dataset()
    Xtr_norm, Xte_norm, scaler = normalise(X_train, X_test)
    
    models = build_models()
    results_all = {}
    
    for mname, clf in models.items():
        print(f"\n--- {mname} ---")
        clf.fit(Xtr_norm, y_train)
        
        metrics, preds, probs = score_model(clf, Xte_norm, y_test)
        results_all[mname] = metrics
        
        for k, v in metrics.items():
            print(f"  {k:>12s} : {v:.4f}")
        
        cm = confusion_matrix(y_test, preds)
        print(f"  Misclassified: {(preds != y_test).sum()} / {len(y_test)}")
        
        # save trained model
        fname = mname.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".pkl"
        joblib.dump(clf, fname)
        print(f"  -> saved {fname}")
    
    # summary table
    print("\n" + "=" * 80)
    print(f"{'Model':<28s} {'Acc':>8s} {'AUC':>8s} {'Prec':>8s} {'Rec':>8s} {'F1':>8s} {'MCC':>8s}")
    print("-" * 80)
    for mname, met in results_all.items():
        print(f"{mname:<28s} {met['Accuracy']:8.4f} {met['AUC']:8.4f} {met['Precision']:8.4f} {met['Recall']:8.4f} {met['F1']:8.4f} {met['MCC']:8.4f}")
    
    # determine winner by F1
    winner = max(results_all, key=lambda m: results_all[m]["F1"])
    print(f"\nBest model (by F1): {winner}  (F1 = {results_all[winner]['F1']:.4f})")
    
    # save summary
    pd.DataFrame(results_all).T.to_csv("model_results.csv")
    print("Results written to model/model_results.csv")


if __name__ == "__main__":
    main()
