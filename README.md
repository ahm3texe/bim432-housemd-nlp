# BIM432 — House MD Türkçe Diyalog Analizi

Doğal Dil İşleme dersi (BIM432) dönem projesi.
House MD dizisinin Türkçe repliklerinden **sınıflandırma**, **duygu analizi** ve **karakter botu** geliştirildi.

## Ekip

- **Ahmet Şafak YILDIRIM**
- **Ogün ŞAHİN**
- **Muhammet Berşan KURTCEPHE**

İstanbul Sabahattin Zaim Üniversitesi · Mayıs 2026

---

## Ne yaptık?

Tek bir replik üzerinden 4 farklı tahmin + en yakın House cevabı:

| Hedef | Sınıf sayısı | Doğruluk |
|-------|-------------:|---------:|
| Konuşma niyeti (Intent) | 8 | 0.5100 |
| Tanı aşaması (diagnosis_stage) | 7 | 0.3815 |
| Duygu (Emotion) | 6 | 0.4759 |
| Alaycılık (Sarcasm) | 2 | 0.8842 |

> **Not:** Alaycılık verisi çok dengesiz (94/6). Model hep "0" dese %94 alırdı; F1 macro 0.57 daha gerçekçi göstergedir.

### Bonus
- **House BOT** — 2463 House repliği üzerinde TF-IDF + cosine similarity ile retrieval
- **5 keşif grafiği** — karakter sarkazm oranı, sezon × duygu, kim hangi tarzda konuşur, vb.
- **Streamlit demo** — tek sayfada 4 sınıflama + bot, ~50 ms tahmin

---

## Proje yapısı

```
final/
├── data/
│   ├── HouseMD_DataSet.csv      # ham veri
│   ├── HouseMD_clean.csv        # temizlenmiş veri (6954 × 17)
│   ├── preprocess_log.md        # 7 aşamalı temizleme raporu
│   └── scripts/preprocess.py    # tek script pipeline
├── src/
│   ├── train.py                 # 4 sınıflama modelini eğitir
│   ├── analytics.py             # bonus keşif grafikleri
│   └── house_bot.py             # retrieval-tabanlı House botu
├── models/                      # eğitilmiş model dosyaları (.joblib)
├── outputs/                     # confusion matrix + grafik + metrik raporları
├── sunum/                       # PowerPoint sunum dosyası
├── app.py                       # Streamlit demo
├── requirements.txt             # Python bağımlılıkları
└── BIM432 - Proje.pdf           # ders gereksinim PDF'i
```

---

## Kurulum

```bash
python3 -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

---

## Çalıştırma

### 1) Veriyi temizle

```bash
venv/bin/python data/scripts/preprocess.py
# data/HouseMD_DataSet.csv → data/HouseMD_clean.csv
```

### 2) Modelleri eğit

```bash
venv/bin/python src/train.py
# models/*.joblib + outputs/{metrics,confusion}_*
```

### 3) Bonus grafikleri üret

```bash
venv/bin/python src/analytics.py
# outputs/chart_*.png
```

### 4) House BOT'u kur

```bash
venv/bin/python src/house_bot.py
# models/house_bot.joblib
```

### 5) Streamlit demosunu aç

```bash
venv/bin/python -m streamlit run app.py
# http://localhost:8501
```

---

## Yaklaşım özeti

- **Veri**: 7282 replik → 7 aşamalı temizleme → 6954 satır (%4.5 kayıp)
- **Özellik çıkarımı**: TF-IDF char n-gram (2-5) + word n-gram (1-2) + ekstra feature (soru/ünlem işareti sayısı)
- **Model**: Logistic Regression × 4 hedef
- **Adil değerlendirme**: GroupShuffleSplit (aynı bölüm hem train hem test'te değil) — veri sızıntısı önlendi
- **Dengesizlik**: Sarcasm için `class_weight='balanced'`

Detaylar için `data/preprocess_log.md` ve `outputs/metrics_*.txt`.

---

## Sınırlılıklar & Etik

- **Tıbbi tavsiye VERMEZ.** Sistem kurgusal dizi repliklerini analiz eder.
- Veri seti küçük (~7K) + çok sınıf → doğruluk %38-51 aralığında, gerçekçi sonuç.
- BERT/transformer modelleri zaman ve donanım kısıtı nedeniyle kullanılmadı (future work).
- House MD telifi yapımcısına aittir; bu çalışma akademik fair-use kapsamındadır.

---

## Lisans

Akademik proje — yeniden kullanım için izin gerekir.
