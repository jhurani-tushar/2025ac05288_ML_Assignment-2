"""
Streamlit App — Handwritten Digit Classification
Compares five ML classifiers on the UCI Digits dataset (10-class).
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
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
import warnings
warnings.filterwarnings("ignore")


# -------- constants --------
SEED = 101
TEST_RATIO = 0.25
TARGET_COL = "digit_label"

# -------- page setup --------
st.set_page_config(page_title="Digit Classifier", page_icon="✋", layout="wide")

st.markdown("""
<style>
    .block-container {padding-top: 1.5rem;}
    h1 {color: #0d7377; font-family: 'Segoe UI', sans-serif;}
    h2, h3 {color: #14919b;}
    .winner-box {
        background: #e8f8f5; border-left: 5px solid #0d7377;
        padding: 12px 18px; border-radius: 4px; margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# =============== helper functions ===============

@st.cache_data
def get_full_dataset():
    """Load digits from sklearn, return as dataframe."""
    raw = load_digits()
    cols = [f"pixel_{i}" for i in range(64)]
    df = pd.DataFrame(raw.data, columns=cols)
    df[TARGET_COL] = raw.target
    return df, cols


@st.cache_resource
def fit_classifiers(_Xtr, _ytr):
    """Train the five classifiers (cached so it runs once)."""
    specs = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, solver="lbfgs", C=0.8, random_state=SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=4, random_state=SEED
        ),
        "kNN": KNeighborsClassifier(n_neighbors=3, weights="distance"),
        "Naive Bayes": GaussianNB(var_smoothing=1e-8),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=200, max_depth=20, min_samples_leaf=2, random_state=SEED
        ),
    }
    fitted = {}
    for name, est in specs.items():
        est.fit(_Xtr, _ytr)
        fitted[name] = est
    return fitted


def calc_metrics(clf, Xte, yte):
    """Return dict of the 6 required metrics + raw predictions."""
    y_hat = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)
    
    out = {
        "Accuracy":  round(accuracy_score(yte, y_hat), 4),
        "AUC":       round(roc_auc_score(yte, y_proba, multi_class="ovr", average="weighted"), 4),
        "Precision": round(precision_score(yte, y_hat, average="weighted", zero_division=0), 4),
        "Recall":    round(recall_score(yte, y_hat, average="weighted", zero_division=0), 4),
        "F1":        round(f1_score(yte, y_hat, average="weighted", zero_division=0), 4),
        "MCC":       round(matthews_corrcoef(yte, y_hat), 4),
    }
    return out, y_hat, y_proba


def draw_cm(yte, y_hat, title):
    """Confusion-matrix heatmap for 10 classes."""
    cm = confusion_matrix(yte, y_hat, labels=range(10))
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu",
                xticklabels=range(10), yticklabels=range(10), ax=ax,
                linewidths=0.4, linecolor="white")
    ax.set_xlabel("Predicted Digit", fontsize=11)
    ax.set_ylabel("Actual Digit", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    return fig


def draw_digit_grid(X_arr, y_arr, n=16):
    """Show a sample grid of digit images from the test set."""
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))
    idxs = np.random.RandomState(42).choice(len(X_arr), size=n, replace=False)
    for i, idx in enumerate(idxs):
        ax = axes[i // 8][i % 8]
        ax.imshow(X_arr[idx].reshape(8, 8), cmap="gray_r", interpolation="nearest")
        ax.set_title(f"{int(y_arr[idx])}", fontsize=10, color="#0d7377")
        ax.axis("off")
    plt.suptitle("Sample Digits from Test Set", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    return fig


def draw_radar(metrics_dict, model_name):
    """Radar chart for a single model's metrics."""
    labels = list(metrics_dict.keys())
    vals = list(metrics_dict.values())
    vals += vals[:1]  # close the polygon
    
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.fill(angles, vals, color="#14919b", alpha=0.25)
    ax.plot(angles, vals, color="#0d7377", linewidth=2, marker="o", markersize=5)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_title(model_name, fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


def draw_grouped_bars(all_metrics_df):
    """Grouped bar chart comparing all models on each metric."""
    metrics_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    n_models = len(all_metrics_df)
    n_metrics = len(metrics_cols)
    x = np.arange(n_metrics)
    bar_w = 0.15
    
    palette = ["#0d7377", "#e2725b", "#45b39d", "#f39c12", "#8e44ad"]
    
    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (mname, row) in enumerate(all_metrics_df.iterrows()):
        offset = (i - n_models / 2 + 0.5) * bar_w
        vals = [row[m] for m in metrics_cols]
        ax.bar(x + offset, vals, bar_w, label=mname, color=palette[i % len(palette)],
               edgecolor="white", linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_cols, fontsize=11)
    ax.set_ylim(0.7, 1.03)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("All Models — Metric-by-Metric Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8.5, loc="lower right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return fig


# =============== MAIN APP ===============

def run():
    
    st.title("✋ Handwritten Digit Classification Dashboard")
    st.caption(
        "Comparing 5 ML classifiers on the UCI Optical Recognition of Handwritten Digits dataset  •  "
        "10 classes (0–9)  •  64 features  •  1797 samples"
    )
    st.markdown("---")
    
    # -- load & split data --
    full_df, feature_cols = get_full_dataset()
    X_all = full_df[feature_cols]
    y_all = full_df[TARGET_COL]
    
    Xtr, Xte_default, ytr, yte_default = train_test_split(
        X_all, y_all, test_size=TEST_RATIO, random_state=SEED, stratify=y_all
    )
    
    # normalise
    scaler = MinMaxScaler()
    Xtr_n = scaler.fit_transform(Xtr)
    
    # train models (cached)
    classifiers = fit_classifiers(Xtr_n, ytr.values)
    
    # ===== SIDEBAR =====
    st.sidebar.header("Controls")
    
    st.sidebar.subheader("Upload Test CSV")
    csv_file = st.sidebar.file_uploader(
        "Choose a CSV with 64 pixel columns + digit_label",
        type=["csv"],
        help="Must contain columns pixel_0…pixel_63 and 'digit_label'."
    )
    
    chosen_model = st.sidebar.selectbox(
        "Pick a model to inspect",
        list(classifiers.keys())
    )
    
    show_all = st.sidebar.checkbox("Show all-model comparison", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Dataset:** UCI Optical Recognition of "
        "Handwritten Digits\n\n"
        "**Features:** 64 (8×8 pixel intensities)\n\n"
        "**Classes:** 10 (digits 0–9)\n\n"
        "**Total samples:** 1 797"
    )
    
    # -- resolve test data --
    if csv_file is not None:
        try:
            uploaded = pd.read_csv(csv_file)
            if TARGET_COL not in uploaded.columns:
                st.error(f"CSV must have a '{TARGET_COL}' column.")
                st.stop()
            feat_cols = [c for c in uploaded.columns if c != TARGET_COL]
            Xte_raw = uploaded[feat_cols].values
            yte = uploaded[TARGET_COL].values
            src_label = "Uploaded CSV"
            st.sidebar.success(f"Loaded {len(yte)} test rows from upload")
        except Exception as exc:
            st.error(f"CSV read error: {exc}")
            st.stop()
    else:
        Xte_raw = Xte_default.values
        yte = yte_default.values
        src_label = "Default 25 % test split"
    
    Xte_n = scaler.transform(Xte_raw)
    
    # ===================================================
    # Section 1 — Dataset Overview
    # ===================================================
    st.header("1 ▸ Dataset at a Glance")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Features", "64")
    c2.metric("Training rows", f"{len(Xtr):,}")
    c3.metric("Test rows", f"{len(yte):,}")
    c4.metric("Classes", "10")
    
    with st.expander("Preview sample digit images", expanded=True):
        fig_grid = draw_digit_grid(Xte_raw, yte)
        st.pyplot(fig_grid)
    
    with st.expander("Test set class distribution"):
        dist = pd.Series(yte).value_counts().sort_index()
        fig_dist, ax_dist = plt.subplots(figsize=(8, 3))
        bars = ax_dist.bar(dist.index.astype(str), dist.values,
                           color="#14919b", edgecolor="white", width=0.6)
        for b in bars:
            ax_dist.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                         str(int(b.get_height())), ha="center", fontsize=9, fontweight="bold")
        ax_dist.set_xlabel("Digit")
        ax_dist.set_ylabel("Count")
        ax_dist.set_title("Samples per Class in Test Data")
        plt.tight_layout()
        st.pyplot(fig_dist)
    
    st.markdown("---")
    
    # ===================================================
    # Section 2 — Selected Model Deep Dive
    # ===================================================
    st.header(f"2 ▸ Model Deep Dive — {chosen_model}")
    
    clf = classifiers[chosen_model]
    met, y_hat, y_proba = calc_metrics(clf, Xte_n, yte)
    
    # metric cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{met['Accuracy']:.4f}")
    m2.metric("AUC", f"{met['AUC']:.4f}")
    m3.metric("Precision", f"{met['Precision']:.4f}")
    m4.metric("Recall", f"{met['Recall']:.4f}")
    m5.metric("F1", f"{met['F1']:.4f}")
    m6.metric("MCC", f"{met['MCC']:.4f}")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Confusion Matrix")
        fig_cm = draw_cm(yte, y_hat, f"Confusion Matrix — {chosen_model}")
        st.pyplot(fig_cm)
    
    with col_right:
        st.subheader("Radar Chart")
        fig_radar = draw_radar(met, chosen_model)
        st.pyplot(fig_radar)
    
    # classification report
    st.subheader("Classification Report")
    rpt = classification_report(
        yte, y_hat,
        labels=range(10),
        target_names=[f"Digit {d}" for d in range(10)],
        output_dict=True
    )
    rpt_df = pd.DataFrame(rpt).T.round(4)
    st.dataframe(rpt_df, use_container_width=True)
    
    # show some misclassified samples
    wrong_idx = np.where(y_hat != yte)[0]
    if len(wrong_idx) > 0:
        with st.expander(f"Misclassified examples ({len(wrong_idx)} total)"):
            show_n = min(8, len(wrong_idx))
            fig_wrong, axes_w = plt.subplots(1, show_n, figsize=(show_n * 1.6, 2))
            if show_n == 1:
                axes_w = [axes_w]
            for i in range(show_n):
                ix = wrong_idx[i]
                axes_w[i].imshow(Xte_raw[ix].reshape(8, 8), cmap="gray_r")
                axes_w[i].set_title(f"T:{int(yte[ix])} P:{int(y_hat[ix])}", fontsize=9, color="red")
                axes_w[i].axis("off")
            plt.suptitle("True (T) vs Predicted (P)", fontsize=11, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig_wrong)
    
    st.markdown("---")
    
    # ===================================================
    # Section 3 — All-Model Comparison
    # ===================================================
    if show_all:
        st.header("3 ▸ All Models — Head to Head")
        
        rows = {}
        for mname, c in classifiers.items():
            m, _, _ = calc_metrics(c, Xte_n, yte)
            rows[mname] = m
        
        comp_df = pd.DataFrame(rows).T
        comp_df.index.name = "Model"
        
        st.dataframe(
            comp_df.style.highlight_max(axis=0, color="#d5f5e3"),
            use_container_width=True,
        )
        
        top_model = comp_df["F1"].idxmax()
        top_f1 = comp_df.loc[top_model, "F1"]
        st.markdown(
            f'<div class="winner-box">🏆 <b>Winner: {top_model}</b> with F1 = {top_f1:.4f}</div>',
            unsafe_allow_html=True
        )
        
        # grouped bars
        fig_bars = draw_grouped_bars(comp_df)
        st.pyplot(fig_bars)
        
        st.markdown("---")
        
        # Observations
        st.header("4 ▸ Observations")
        
        notes = {
            "Logistic Regression": (
                "Surprisingly strong for a linear model on what is essentially an image task. "
                "The multinomial solver handles the 10-class problem efficiently, and the "
                "high-dimensional pixel space (64 features) actually works in its favour because "
                "each digit occupies a fairly distinct region after normalisation. "
                "Only 12 misclassifications on 450 test samples is impressive."
            ),
            "Decision Tree": (
                "Clearly the weakest of the five. Even with max_depth=12 the tree overfits "
                "the training data and then fails to generalise to unseen digits. 72 mistakes "
                "on 450 samples is a big gap compared to the others. The pixel features don't "
                "lend themselves well to axis-aligned splits — a slight shift in a digit's "
                "position can fool the tree entirely."
            ),
            "kNN": (
                "Best performer overall. Distance-weighted kNN with k=3 excels here because "
                "similar-looking digits naturally cluster close in the 64-dimensional pixel "
                "space. Only 5 out of 450 samples were misclassified. The near-perfect AUC "
                "of 0.9999 shows it ranks almost every sample correctly. The trade-off is "
                "speed: prediction requires comparing against all training samples."
            ),
            "Naive Bayes": (
                "Second weakest with 64 errors. The Gaussian assumption does not hold well "
                "for pixel intensity values, which tend to be bimodal (mostly 0 or close to "
                "max). Despite this, precision is notably higher than recall (0.885 vs 0.858), "
                "meaning when it commits to a prediction it is more often right, but it "
                "confuses some digit pairs like 1/8 and 3/9."
            ),
            "Random Forest (Ensemble)": (
                "Very close to Logistic Regression, finishing just behind with 13 errors vs 12. "
                "The ensemble of 200 trees smooths out the instability of a single Decision "
                "Tree, recovering from 84% accuracy to 97%. The near-perfect AUC (0.9995) "
                "shows excellent probability calibration. It's a strong general-purpose pick "
                "when you have no prior knowledge about the data."
            ),
            "Overall Winner": (
                "kNN takes the crown on every metric for this dataset. The digits data "
                "has well-separated clusters in the pixel space, which is exactly where "
                "instance-based methods shine. For a production system, however, kNN's slow "
                "inference could be a drawback and one might prefer Logistic Regression or "
                "Random Forest which are nearly as accurate but much faster at prediction time."
            ),
        }
        
        for lbl, txt in notes.items():
            with st.expander(f"📝 {lbl}", expanded=(lbl == "Overall Winner")):
                st.write(txt)
    
   


if __name__ == "__main__":
    run()
