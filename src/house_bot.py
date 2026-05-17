"""
src/house_bot.py — Retrieval-based House Bot.

LLM YOK. Sadece TF-IDF + cosine similarity:
    1) House'un tüm replikleri (speaker='house') TF-IDF vektör matrisi olur
    2) Kullanıcı text'i aynı vectorizer ile vektörlenir
    3) En yüksek cosine similarity skorlu House repliği döndürülür

Kaydedilen bot dosyası Streamlit demosunda kullanılır.

Girdi  : data/HouseMD_clean.csv
Çıktı  : models/house_bot.joblib
"""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "HouseMD_clean.csv"
BOT_FILE = ROOT / "models" / "house_bot.joblib"
BOT_FILE.parent.mkdir(exist_ok=True)


@dataclass
class HouseReply:
    text: str
    season: int
    episode: int
    similarity: float


class HouseBot:
    """Retrieval-based replik botu."""

    def __init__(self, replies: pd.DataFrame, vectorizer: TfidfVectorizer, matrix):
        self.replies = replies.reset_index(drop=True)
        self.vec = vectorizer
        self.matrix = matrix

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "HouseBot":
        replies = df[df["speaker"] == "house"][["text", "season", "episode"]].copy()
        replies = replies.dropna(subset=["text"])
        replies = replies[replies["text"].str.len() > 0]
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
        )
        matrix = vec.fit_transform(replies["text"].astype(str))
        return cls(replies, vec, matrix)

    def reply(self, user_text: str, top_k: int = 1) -> list[HouseReply]:
        if not user_text.strip():
            return []
        user_vec = self.vec.transform([user_text])
        sims = cosine_similarity(user_vec, self.matrix)[0]
        # En yüksek top_k indeks
        top_idx = sims.argsort()[-top_k:][::-1]
        out = []
        for i in top_idx:
            row = self.replies.iloc[i]
            out.append(HouseReply(
                text=str(row["text"]),
                season=int(row["season"]),
                episode=int(row["episode"]),
                similarity=float(sims[i]),
            ))
        return out


def main() -> None:
    print(f"Loading {DATA_FILE.name} ...")
    df = pd.read_csv(DATA_FILE, sep=";")
    print(f"Loaded: {len(df)} satır")

    bot = HouseBot.fit(df)
    print(f"House replik sayısı : {len(bot.replies)}")
    print(f"TF-IDF feature sayısı: {bot.matrix.shape[1]}\n")

    # CLI test örnekleri
    samples = [
        "lupus olabilir mi?",
        "tomografi çekin hemen",
        "hasta sarhoş muydu?",
        "wilson sen ne düşünüyorsun",
        "hepatit c olabilir",
        "şaka yapmıyorum",
    ]
    print("=== Örnek tahminler ===\n")
    for q in samples:
        print(f"USER:  {q}")
        for r in bot.reply(q, top_k=2):
            print(f"  HOUSE: {r.text[:90]}{'...' if len(r.text) > 90 else ''}")
            print(f"         (S{r.season}E{r.episode}, sim={r.similarity:.3f})")
        print()

    # Kaydet
    joblib.dump(bot, BOT_FILE)
    size_kb = BOT_FILE.stat().st_size / 1024
    print(f"Saved: {BOT_FILE.relative_to(ROOT)}  ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
