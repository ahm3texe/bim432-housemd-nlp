"""
src/train.py — 4 sınıflama modeli eğit ve kaydet.

Görev: Intent / diagnosis_stage / Emotion / Sarcasm için ayrı LogisticRegression
modelleri eğit. TF-IDF char_wb (2-5) + word (1-2) n-gram + ekstra feature'lar
(q_count, ex_count) birleştirilir.

Veri sızıntısını önlemek için bölüm bazlı (season+episode) GroupShuffleSplit kullanılır:
aynı bölümün replikleri ya tamamen train'de ya tamamen test'te kalır.

Girdi : data/HouseMD_clean.csv
Çıktı : models/<target>_model.joblib    (model + vectorizer + label_encoder)
        outputs/metrics_<target>.txt    (classification report)
        outputs/confusion_<target>.png  (confusion matrix heatmap)
"""
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "HouseMD_clean.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)

TARGETS = ["Intent", "diagnosis_stage", "Emotion", "Sarcasm"]
SEED = 42


def vectorize(X_text_train, X_text_test, X_extra_train, X_extra_test):
    """char + word TF-IDF + extra feature → sparse matrix."""
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    Xc_tr = char_vec.fit_transform(X_text_train)
    Xc_te = char_vec.transform(X_text_test)
    Xw_tr = word_vec.fit_transform(X_text_train)
    Xw_te = word_vec.transform(X_text_test)
    Xe_tr = csr_matrix(X_extra_train.values)
    Xe_te = csr_matrix(X_extra_test.values)
    X_train = hstack([Xc_tr, Xw_tr, Xe_tr]).tocsr()
    X_test = hstack([Xc_te, Xw_te, Xe_te]).tocsr()
    return X_train, X_test, char_vec, word_vec


def save_confusion(y_true, y_pred, labels, target, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    fig_size = max(6, len(labels) * 0.6)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        cbar=False,
    )
    plt.title(f"Confusion Matrix — {target}")
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def train_one(df: pd.DataFrame, target: str, summary: dict) -> None:
    print(f"\n{'=' * 60}\n=== {target} ===\n{'=' * 60}")

    X_text = df["text"].astype(str).fillna("")
    X_extra = df[["q_count", "ex_count"]].astype(float)
    y_raw = df[target].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Sınıf sayısı: {len(le.classes_)}")
    print(f"Sınıflar    : {list(le.classes_)}")

    # Bölüm bazlı split (data leakage'sız)
    groups = (df["season"].astype(str) + "_" + df["episode"].astype(str)).values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(X_text, y, groups))

    X_text_tr = X_text.iloc[train_idx]
    X_text_te = X_text.iloc[test_idx]
    X_extra_tr = X_extra.iloc[train_idx]
    X_extra_te = X_extra.iloc[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    print(f"Train: {len(y_tr)} satır | Test: {len(y_te)} satır")
    print(f"Train bölüm: {len(set(groups[train_idx]))} | Test bölüm: {len(set(groups[test_idx]))}")

    X_train, X_test, char_vec, word_vec = vectorize(
        X_text_tr, X_text_te, X_extra_tr, X_extra_te
    )
    print(f"Feature boyutu: {X_train.shape[1]} (char + word + extra)")

    # Sarcasm dengesiz → balanced, diğerleri varsayılan (accuracy odaklı)
    cw = "balanced" if target == "Sarcasm" else None
    model = LogisticRegression(
        class_weight=cw,
        max_iter=3000,
        solver="lbfgs",
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(X_train, y_tr)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_te, y_pred)
    f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_te, y_pred, average="weighted", zero_division=0)

    print(f"\nAccuracy      : {acc:.4f}")
    print(f"F1 (macro)    : {f1_macro:.4f}")
    print(f"F1 (weighted) : {f1_weighted:.4f}")

    report = classification_report(
        y_te, y_pred, target_names=le.classes_, zero_division=0
    )
    print("\nClassification Report:\n" + report)

    summary[target] = {
        "n_classes": len(le.classes_),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "feature_dim": int(X_train.shape[1]),
    }

    # Confusion matrix kaydet
    cm_path = OUTPUTS_DIR / f"confusion_{target}.png"
    save_confusion(y_te, y_pred, le.classes_, target, cm_path)
    print(f"Saved: {cm_path.relative_to(ROOT)}")

    # Metrik raporu kaydet
    metrics_path = OUTPUTS_DIR / f"metrics_{target}.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"=== {target} ===\n")
        f.write(f"Accuracy      : {acc:.4f}\n")
        f.write(f"F1 (macro)    : {f1_macro:.4f}\n")
        f.write(f"F1 (weighted) : {f1_weighted:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Saved: {metrics_path.relative_to(ROOT)}")

    # Model + vectorizer + encoder tek bundle olarak kaydet
    bundle = {
        "model": model,
        "char_vec": char_vec,
        "word_vec": word_vec,
        "label_encoder": le,
        "target": target,
        "extra_cols": ["q_count", "ex_count"],
    }
    model_path = MODELS_DIR / f"{target}_model.joblib"
    joblib.dump(bundle, model_path)
    print(f"Saved: {model_path.relative_to(ROOT)}")


def split_and_save(df: pd.DataFrame) -> pd.DataFrame:
    """Veriyi bölüm bazlı %80/%20 train/test'e ayır ve diske yaz.

    Aynı (season+episode) bölümünün replikleri ya tamamen train'e ya tamamen
    test'e düşer — modelin ezbere değil gerçekten öğrenip öğrenmediğini
    ölçmek için (data leakage önlenir).
    """
    groups = (df["season"].astype(str) + "_" + df["episode"].astype(str)).values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(gss.split(df, groups=groups))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    train_path = ROOT / "data" / "train.csv"
    test_path = ROOT / "data" / "test.csv"
    df_train.to_csv(train_path, sep=";", index=False)
    df_test.to_csv(test_path, sep=";", index=False)

    print(f"\n=== Split ===")
    print(f"Train: {len(df_train)} satır · {df_train[['season','episode']].drop_duplicates().shape[0]} bölüm")
    print(f"Test : {len(df_test)} satır · {df_test[['season','episode']].drop_duplicates().shape[0]} bölüm")
    print(f"Saved: {train_path.relative_to(ROOT)}")
    print(f"Saved: {test_path.relative_to(ROOT)}")
    # Bilgi: bölümler hiç çakışmıyor mu doğrula
    train_eps = set(zip(df_train["season"], df_train["episode"]))
    test_eps = set(zip(df_test["season"], df_test["episode"]))
    overlap = train_eps & test_eps
    print(f"Çakışan bölüm: {len(overlap)}  (0 olmalı)")
    return df_train, df_test


def main() -> None:
    print(f"Loading {DATA_FILE.name} ...")
    df = pd.read_csv(DATA_FILE, sep=";")
    print(f"Loaded: {len(df)} satır × {len(df.columns)} kolon")

    df_train, df_test = split_and_save(df)

    summary = {}
    for target in TARGETS:
        train_one(df, target, summary)

    # Özet tablo
    print(f"\n\n{'=' * 60}\n=== ÖZET ===\n{'=' * 60}")
    print(f"{'Hedef':<22} {'Sınıf':>6} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>13}")
    print("-" * 65)
    for target, m in summary.items():
        print(f"{target:<22} {m['n_classes']:>6} "
              f"{m['accuracy']:>10.4f} {m['f1_macro']:>10.4f} {m['f1_weighted']:>13.4f}")

    # Özeti JSON olarak kaydet
    summary_path = OUTPUTS_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {summary_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
