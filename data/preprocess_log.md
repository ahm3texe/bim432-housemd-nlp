# Preprocess Log

```
HouseMD_DataSet.csv  ──preprocess.py──▶  HouseMD_clean.csv
   (7282 × 16)                              (6954 × 17)
```

**Veri kaybı:** 328 satır (%4.50)
**Pipeline:** [`data/scripts/preprocess.py`](scripts/preprocess.py) — tek script, 7 aşama

---

## 7 Aşama (kısaca)

1. **Lowercase** — Türkçe-uyumlu (`İ→i`, `I→ı`), tüm hücreler
2. **Sarcasm fix** — binary'e indirgendi: `hayır→0`, `alaycı→1`; 4 anomali atıldı
3. **Speaker clean** — `dr. X → X`, `doktor → doctor`, typo/yan etki düzelt; 14 belirsiz speaker (35 satır) atıldı
4. **Data quality** — apostrof öncesi yan etki düzelt (16 kelime) + eksik etiket at (203) + tam duplike (23) + çelişen text→etiket (33)
5. **Label clean** — Intent/Stage/Emotion için yakın etiket merge + rare → `diğer` sınıfı
6. **Aux cols** — Organ `-` → `''`, `model_prediction` kolonu silindi
7. **Text clean** — `q_count`/`ex_count` feature çıkar, noktalama temizle (apostrof korundu), kısa text filter (≤10 char, 30 satır)

---

## Final Çıktı

**`data/HouseMD_clean.csv`** — 6954 × 17

### Sınıflama hedefleri
| Kolon | Sınıf | Top |
|-------|------:|-----|
| Intent | 13 | açıklama (%49) |
| diagnosis_stage | 11 | değerlendirme (%24) |
| Emotion | 10 | nötr (%54) |
| Sarcasm | 2 | `0` (%94) → dengesiz, `class_weight='balanced'` |

### Yeni feature'lar
- `q_count`, `ex_count` — sarkazm için noktalama sinyali

### Modelleme aşamasında
- TF-IDF char_wb n-gram (2-5) — Türkçe ekleri stemmer'sız çözer
- StratifiedGroupKFold — bölüm bazlı split (data leakage'sız)
- LogisticRegression + `class_weight='balanced'`

### Bilerek atlandı
- Sayı normalize (yaş/yüzde anlamlı)
- Stopword (TF-IDF IDF zaten kötüler)
- Stemming (char n-gram alternatifi var, bağımlılık eklenmedi)
