"""
preprocess.py — HouseMD verisinin tek adımda tam preprocess pipeline'ı.

Tüm preprocess adımları sırasıyla tek script içinde:

    1) Lowercase (Türkçe-uyumlu, tüm hücreler)
    2) Sarcasm fix → binary (0/1)
    3) Speaker clean → map (dr. x → x, doktor → doctor, ...) + drop belirsizler
    4) Data quality:
        a) Lowercase yan etkisi düzeltme (ıv → iv, ıpecac → ipecac, ...)
        b) Eksik kritik etiket olan satırları at (Intent/Stage/Emotion/text)
        c) Tam duplike satır temizliği
        d) Çelişen text→etiket duplikatları (aynı text farklı etiket → at)
    5) Label clean → Intent / diagnosis_stage / Emotion için:
        - Yakın anlamlı etiketleri birleştir (merge)
        - Top N dışındaki rare sınıfları "diğer" sınıfına grupla
    6) Aux cols:
        - Organ kolonu '-' fake null değerlerini boşalt
        - model_prediction kolonunu sil (kullanılmıyor)
    7) Text clean:
        - q_count + ex_count feature kolonları (sarkazm sinyali)
        - Noktalama temizleme (Türkçe harfler korunur)
        - Whitespace normalize

Girdi : data/HouseMD_DataSet.csv  (ham veri)
Çıktı : data/HouseMD_clean.csv    (modele girecek temiz veri)
"""
import csv
import re
from collections import defaultdict
from pathlib import Path


# === [1] Türkçe lowercase ===
def tr_lower(s: str) -> str:
    return s.translate(str.maketrans({"İ": "i", "I": "ı"})).lower()


# === [2] Sarcasm map ===
SARCASM_MAP = {"0": "0", "1": "1", "hayır": "0", "alaycı": "1"}


# === [3] Speaker ===
SPEAKER_MAP = {
    # "dr. X" → "X"
    "dr. house": "house", "dr. foreman": "foreman", "dr. chase": "chase",
    "dr. cameron": "cameron", "dr. wilson": "wilson", "dr. cuddy": "cuddy",
    "dr. simpson": "simpson",
    # Typo
    "cudy": "cuddy",
    # Türkçe lowercase yan etkisi
    "ırene": "irene", "ıan": "ian",
    # Türkçe ↔ İngilizce rol
    "doktor": "doctor", "doktorlar": "doctor",
    "hasta": "patient", "hemşire": "nurse", "anne": "mom",
}

DROP_SPEAKERS = {
    "surgeon", "cerrah", "attending doctor",
    "paramedik", "emt",
    "hasta (jill)", "klinik hastası", "hale oliver olmayan hasta",
    "acil servis doktoru", "doktor (ob)", "anesteziyolog", "acil servis",
    "chase ve thirteen", "ekip",
}


# === [4a] Lowercase yan etkisi düzeltme ===
LOWERCASE_FIX = {
    "ıv": "iv", "ıvıg": "ivig", "ıpecac": "ipecac", "ıud": "iud",
    "ınh": "inh", "ıan": "ian", "ıge": "ige", "ıgf": "igf", "ıtp": "itp",
    "ıron": "iron", "ırene": "irene", "ıgg": "igg", "ıga": "iga",
    "ıntravenöz": "intravenöz", "ıdrarımda": "idrarımda", "ıcp": "icp",
}
LC_WORD_RE = re.compile(r"\bı[a-zçğıöşü]+")


# === [5] Label merge + top ===
INTENT_MERGE = {"teşhis": "tanı"}
STAGE_MERGE = {
    "ayırıcı tanı": "ayırıcı_tanı",
    "teşhis": "tanı",
    "kesin_tanı": "tanı",
}
EMOTION_MERGE = {
    "endişeli": "endişe", "kaygı": "endişe", "tarafsız": "nötr",
}
INTENT_TOP = {
    "açıklama", "hipotez", "soru", "tanı", "talimat", "tedavi", "değerlendirme",
}
STAGE_TOP = {
    "değerlendirme", "hipotez", "test", "tedavi", "tanı", "yok",
}
EMOTION_TOP = {
    "nötr", "ciddi", "endişe", "alaycı", "analitik",
}
OTHER = "diğer"


# === [7] Text clean ===
# Apostrofu noktalamadan KORUYORUZ — Türkçe iyelik ekleri (cameron'ın,
# mr'ını, %90'ında) tek kelime kalsın, ek parçalanmasın.
PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)
WS_RE = re.compile(r"\s+")
MIN_TEXT_LEN = 10  # bu uzunluk veya altındaki textler bilgi taşımıyor → at


# === Yol ===
DATA_DIR = Path(__file__).resolve().parents[1]
IN_FILE = DATA_DIR / "HouseMD_DataSet.csv"
OUT_FILE = DATA_DIR / "HouseMD_clean.csv"


def fix_lowercase_side_effect(s: str) -> str:
    if not s:
        return s
    return LC_WORD_RE.sub(lambda m: LOWERCASE_FIX.get(m.group(0), m.group(0)), s)


def apply_label_clean(rows, header, col, merge_map, top_set):
    i = header.index(col)
    for r in rows:
        v = r[i].strip() if i < len(r) else ""
        if not v:
            continue
        v = merge_map.get(v, v)
        if v not in top_set:
            v = OTHER
        r[i] = v


def main() -> None:
    # CSV oku
    with open(IN_FILE, encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=";")
        header = next(reader)
        rows = list(reader)
    n0 = len(rows)
    print(f"Ham veri: {n0} satır × {len(header)} kolon\n")

    # [1] Lowercase TÜM hücreler
    for r in rows:
        for i, v in enumerate(r):
            if v:
                r[i] = tr_lower(v)
    print(f"[1] Lowercase uygulandı (TÜM hücreler)")

    # [2] Sarcasm fix
    si = header.index("Sarcasm")
    before = len(rows)
    rows = [r for r in rows if si < len(r) and r[si].strip() in SARCASM_MAP]
    for r in rows:
        r[si] = SARCASM_MAP[r[si].strip()]
    print(f"[2] Sarcasm: binary → {len(rows)} satır  ({before - len(rows)} anomali atıldı)")

    # [3] Speaker clean
    spi = header.index("speaker")
    before = len(rows)
    rows = [r for r in rows if spi < len(r) and r[spi].strip() not in DROP_SPEAKERS]
    for r in rows:
        v = r[spi].strip()
        if v in SPEAKER_MAP:
            r[spi] = SPEAKER_MAP[v]
    print(f"[3] Speaker: clean → {len(rows)} satır  ({before - len(rows)} drop)")

    # [4a] Lowercase yan etkisi düzelt
    for col in ("text", "correct_prediction"):
        ci = header.index(col)
        for r in rows:
            if ci < len(r) and r[ci]:
                r[ci] = fix_lowercase_side_effect(r[ci])

    # [4b] Eksik kritik etiket at
    crit = [header.index(c) for c in ("Intent", "diagnosis_stage", "Emotion", "text")]
    before = len(rows)
    rows = [r for r in rows if all(i < len(r) and r[i].strip() for i in crit)]
    n4b = before - len(rows)

    # [4c] Tam duplike at
    seen = set()
    deduped = []
    for r in rows:
        k = tuple(r)
        if k not in seen:
            seen.add(k)
            deduped.append(r)
    n4c = len(rows) - len(deduped)
    rows = deduped

    # [4d] Çelişen text→etiket at
    ti = header.index("text")
    text_labels = defaultdict(lambda: defaultdict(set))
    for r in rows:
        t = r[ti].strip()
        if not t:
            continue
        for col in ("Intent", "diagnosis_stage", "Emotion", "Sarcasm"):
            ci = header.index(col)
            v = r[ci].strip() if ci < len(r) else ""
            if v:
                text_labels[t][col].add(v)
    conflict = {
        t for t, cols in text_labels.items()
        if any(len(vs) > 1 for vs in cols.values())
    }
    before = len(rows)
    rows = [r for r in rows if r[ti].strip() not in conflict]
    n4d = before - len(rows)
    print(f"[4] Data quality: empty={n4b}, duplicate={n4c}, conflict={n4d} → {len(rows)} satır")

    # [5] Label clean
    apply_label_clean(rows, header, "Intent", INTENT_MERGE, INTENT_TOP)
    apply_label_clean(rows, header, "diagnosis_stage", STAGE_MERGE, STAGE_TOP)
    apply_label_clean(rows, header, "Emotion", EMOTION_MERGE, EMOTION_TOP)
    print(f"[5] Labels: Intent + Stage + Emotion merge + rare → 'diğer'")

    # [6] Aux cols
    oi = header.index("Organ")
    for r in rows:
        if oi < len(r) and r[oi].strip() == "-":
            r[oi] = ""
    mpi = header.index("model_prediction")
    header = [c for i, c in enumerate(header) if i != mpi]
    rows = [[v for i, v in enumerate(r) if i != mpi] for r in rows]
    print(f"[6] Aux: Organ '-' temizlendi, model_prediction kolonu silindi")

    # [7] Text clean + feature + kısa text filtresi
    ti = header.index("text")
    header = header + ["q_count", "ex_count"]
    new_rows = []
    short_dropped = 0
    for r in rows:
        text = r[ti] if ti < len(r) else ""
        q = text.count("?")
        ex = text.count("!")
        text = PUNCT_RE.sub(" ", text)
        text = WS_RE.sub(" ", text).strip()
        if len(text) <= MIN_TEXT_LEN:
            short_dropped += 1
            continue
        r[ti] = text
        new_rows.append(r + [str(q), str(ex)])
    rows = new_rows
    print(f"[7] Text: q_count + ex_count + noktalama (apostrof korundu) + whitespace")
    print(f"    Kısa text (<={MIN_TEXT_LEN} kar) atılan: {short_dropped} satır")

    # Yaz
    with open(OUT_FILE, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        w.writerows(rows)

    print(f"\nÇıktı: {OUT_FILE.name}")
    print(f"Sonuç: {len(rows)} satır × {len(header)} kolon")
    print(f"Veri kaybı: {n0 - len(rows)} satır ({100*(n0-len(rows))/n0:.2f}%)")


if __name__ == "__main__":
    main()
