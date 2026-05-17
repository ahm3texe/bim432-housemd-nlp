"""
src/analytics.py — Bonus analitik grafikleri üretir.

Sınıflama metriklerine ek olarak veri keşfi grafikleri:
    1) Karakter bazlı sarkazm oranı (top 12)
    2) Sezon × Emotion heatmap
    3) Top 6 karakter × Intent stacked bar
    4) Sezon bazlı diagnosis_stage dağılımı
    5) Karakter konuşma sıklığı (top 12)

Çıktı: outputs/chart_*.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "HouseMD_clean.csv"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.titlesize"] = 13


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, sep=";")
    df["Sarcasm"] = df["Sarcasm"].astype(int)
    df["season"] = df["season"].astype(int)
    return df


def chart_sarcasm_by_speaker(df: pd.DataFrame) -> None:
    """Top 12 karakter için sarkazm oranı (%)."""
    top = df["speaker"].value_counts().head(12).index
    sub = df[df["speaker"].isin(top)]
    rates = sub.groupby("speaker")["Sarcasm"].agg(["mean", "count"])
    rates["mean"] *= 100
    rates = rates.sort_values("mean", ascending=False)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(rates.index, rates["mean"], color="#8A2432", edgecolor="black", linewidth=0.5)
    for bar, n in zip(bars, rates["count"]):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.2,
                f"%{h:.1f}\n(n={n})",
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Karakter Bazlı Sarkazm Oranı (Top 12)")
    ax.set_ylabel("Sarkastik Replik Oranı (%)")
    ax.set_xlabel("Karakter")
    ax.set_ylim(0, rates["mean"].max() * 1.25)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    out = OUT_DIR / "chart_sarcasm_by_speaker.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def chart_emotion_heatmap(df: pd.DataFrame) -> None:
    """Sezon × Emotion sayım heatmap'i."""
    pivot = pd.crosstab(df["season"], df["Emotion"])
    # Sütunları toplam sayıya göre sırala
    pivot = pivot[pivot.sum().sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd",
                cbar_kws={"label": "Replik Sayısı"}, ax=ax)
    ax.set_title("Sezon × Duygu Dağılımı")
    ax.set_ylabel("Sezon")
    ax.set_xlabel("Duygu")
    plt.xticks(rotation=20, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out = OUT_DIR / "chart_emotion_by_season.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def chart_intent_by_speaker(df: pd.DataFrame) -> None:
    """Top 6 karakter için Intent stacked bar (yüzde)."""
    top = df["speaker"].value_counts().head(6).index
    sub = df[df["speaker"].isin(top)]
    pivot = pd.crosstab(sub["speaker"], sub["Intent"], normalize="index") * 100
    pivot = pivot.loc[top]  # konuşma sıklığına göre sırala

    fig, ax = plt.subplots(figsize=(11, 6))
    pivot.plot(kind="bar", stacked=True, ax=ax,
               colormap="tab20", edgecolor="white", width=0.7)
    ax.set_title("Karakter Bazlı Konuşma Niyeti (Intent) Dağılımı")
    ax.set_ylabel("Yüzde (%)")
    ax.set_xlabel("Karakter")
    ax.legend(title="Intent", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = OUT_DIR / "chart_intent_by_speaker.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def chart_stage_by_season(df: pd.DataFrame) -> None:
    """Sezon bazlı diagnosis_stage line chart (yüzde)."""
    pivot = pd.crosstab(df["season"], df["diagnosis_stage"], normalize="index") * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot.plot(kind="line", marker="o", ax=ax, linewidth=2)
    ax.set_title("Sezona Göre Tanı Aşaması (diagnosis_stage) Dağılımı")
    ax.set_ylabel("Yüzde (%)")
    ax.set_xlabel("Sezon")
    ax.set_xticks(pivot.index)
    ax.legend(title="Stage", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.tight_layout()
    out = OUT_DIR / "chart_stage_by_season.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def chart_top_speakers(df: pd.DataFrame) -> None:
    """En çok konuşan 12 karakter — bar chart."""
    top = df["speaker"].value_counts().head(12)

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = sns.color_palette("viridis", n_colors=len(top))
    bars = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor="black", linewidth=0.4)
    for bar, n in zip(bars, top.values[::-1]):
        ax.text(n + 20, bar.get_y() + bar.get_height() / 2,
                str(n), va="center", fontsize=10)
    ax.set_title("Karakter Konuşma Sıklığı (Top 12)")
    ax.set_xlabel("Replik Sayısı")
    ax.set_ylabel("Karakter")
    plt.tight_layout()
    out = OUT_DIR / "chart_top_speakers.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {out.name}")


def main() -> None:
    print(f"Loading {DATA_FILE.name} ...")
    df = load()
    print(f"Loaded: {len(df)} satır\n")

    print("Generating charts:")
    chart_sarcasm_by_speaker(df)
    chart_emotion_heatmap(df)
    chart_intent_by_speaker(df)
    chart_stage_by_season(df)
    chart_top_speakers(df)
    print(f"\nAll charts saved to: {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
