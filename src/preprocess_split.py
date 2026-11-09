# preprocess_split.py
# Faz 1 - Etiketleri ortak siniflara cevirme ve sizintisiz egitim/dogrulama/test bolmesi
# Neden grup bazli bolme? HAM10000'de ayni lezyonun birden fazla fotografi var.
# Rastgele bolersem ayni lezyon hem egitimde hem testte olur ve model "ezberleyip" yuksek skor alir.
# Bu yuzden HAM10000'de lesion_id'ye, PAD-UFES-20'de patient_id'ye gore grupluyorum.

import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold   # hem sinif oranini koruyan hem grubu bolmeyen yontem

ANA_KLASOR = Path(__file__).resolve().parents[1]
METADATA = ANA_KLASOR / "data" / "metadata"
BOLME = ANA_KLASOR / "data" / "splits"          # bolme dosyalarini buraya yazacagim
GRAFIK = ANA_KLASOR / "figures"
SONUC = ANA_KLASOR / "results"
BOLME.mkdir(parents=True, exist_ok=True)

TOHUM = 42                                  # her calistirmada ayni bolmeyi almak icin sabit sayi
SINIFLAR = ["AK", "BCC", "MEL", "NEV"]      # iki veri setinde de bulunan ortak 4 sinif

# Etiket esleme tablolari (solda orijinal ad, sagda projedeki ortak ad)
HAM_ESLEME = {"akiec": "AK", "bcc": "BCC", "mel": "MEL", "nv": "NEV"}   # bkl, df, vasc cikariliyor
PAD_ESLEME = {"ACK": "AK", "BCC": "BCC", "MEL": "MEL", "NEV": "NEV"}    # SCC, SEK cikariliyor


def grup_bazli_bol(df, etiket_sutunu, grup_sutunu):
    # Veriyi yaklasik %70 egitim, %15 dogrulama, %15 test olarak boler.
    # 1. adim: 7 parcaya bolup 1 parcayi test yapiyorum (1/7 = yaklasik %15)
    bolucu = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=TOHUM)
    egitim_dogr_idx, test_idx = next(bolucu.split(df, df[etiket_sutunu], df[grup_sutunu]))
    kalan = df.iloc[egitim_dogr_idx]
    # 2. adim: kalan veriyi 6 parcaya bolup 1 parcayi dogrulama yapiyorum (6/7 * 1/6 = yaklasik %15)
    bolucu2 = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=TOHUM)
    egitim_idx, dogr_idx = next(bolucu2.split(kalan, kalan[etiket_sutunu], kalan[grup_sutunu]))
    sonuc = df.copy()
    sonuc["split"] = "test"                                     # once herkesi test yap
    sonuc.loc[kalan.index[egitim_idx], "split"] = "train"       # egitime dusenleri isaretle
    sonuc.loc[kalan.index[dogr_idx], "split"] = "val"           # dogrulamaya dusenleri isaretle
    return sonuc


def sizinti_kontrolu(df, grup_sutunu):
    # Ayni lezyon/hasta iki farkli kumede var mi diye kontrol ediyorum
    gruplar = {k: set(df.loc[df["split"] == k, grup_sutunu]) for k in ("train", "val", "test")}
    kesisim = {
        "train∩val": len(gruplar["train"] & gruplar["val"]),    # & = iki kumenin ortak elemanlari
        "train∩test": len(gruplar["train"] & gruplar["test"]),
        "val∩test": len(gruplar["val"] & gruplar["test"]),
    }
    # herhangi bir kesisim varsa program hata verip durur, boylece fark etmeden gecmem
    assert sum(kesisim.values()) == 0, f"Veri sizintisi var: {kesisim}"
    return kesisim


def main():
    ham = pd.read_csv(METADATA / "HAM10000_metadata.csv")
    pad = pd.read_csv(METADATA / "PAD-UFES-20_metadata.csv")

    # Sadece ortak 4 sinifi tutup etiketleri yeni adlarina ceviriyorum
    ham4 = ham[ham["dx"].isin(HAM_ESLEME)].copy()
    ham4["label"] = ham4["dx"].map(HAM_ESLEME)
    pad4 = pad[pad["diagnostic"].isin(PAD_ESLEME)].copy()
    pad4["label"] = pad4["diagnostic"].map(PAD_ESLEME)
    ham4 = ham4.reset_index(drop=True)          # satir numaralarini 0'dan tekrar baslat
    pad4 = pad4.reset_index(drop=True)

    # Grup bazli bolme
    ham_bolunmus = grup_bazli_bol(ham4, "label", "lesion_id")
    pad_bolunmus = grup_bazli_bol(pad4, "label", "patient_id")

    # Sizinti kontrolu
    kontrol = {
        "HAM10000 (lesion_id)": sizinti_kontrolu(ham_bolunmus, "lesion_id"),
        "PAD-UFES-20 (patient_id)": sizinti_kontrolu(pad_bolunmus, "patient_id"),
    }

    # Bolme dosyalarini kaydet (model egitiminde bu dosyalari kullanacagim)
    ham_bolunmus[["image_id", "lesion_id", "dx", "label", "split"]].to_csv(
        BOLME / "ham10000_4class_split.csv", index=False)
    pad_bolunmus[["img_id", "lesion_id", "patient_id", "diagnostic", "label", "split"]].to_csv(
        BOLME / "pad_ufes20_4class_split.csv", index=False)

    # Her sinif ve kume icin goruntu sayisi tablosu
    tablolar = {}
    for ad, veri in (("HAM10000", ham_bolunmus), ("PAD-UFES-20", pad_bolunmus)):
        tablo = pd.crosstab(veri["label"], veri["split"])               # satir: sinif, sutun: kume
        tablo = tablo.reindex(index=SINIFLAR, columns=["train", "val", "test"])
        tablo.loc["Toplam"] = tablo.sum()                               # en alta toplam satiri ekle
        tablolar[ad] = tablo
        print(f"\n{ad}\n{tablo}")

    ozet = {
        "HAM10000_once_sonra": {"once_goruntu": len(ham), "once_sinif": 7,
                                "sonra_goruntu": len(ham4), "sonra_sinif": 4},
        "PAD_once_sonra": {"once_goruntu": len(pad), "once_sinif": 6,
                           "sonra_goruntu": len(pad4), "sonra_sinif": 4},
        "bolme_tablolari": {k: v.to_dict() for k, v in tablolar.items()},
        "sizinti_kontrolu": kontrol,
    }
    with open(SONUC / "bolme_ozeti.json", "w", encoding="utf-8") as f:
        json.dump(ozet, f, indent=2, ensure_ascii=False, default=int)
    print("\nSizinti kontrolu:", kontrol)

    # Grafik 5: bolme sonrasi her sinifin egitim/dogrulama/test dagilimi (ust uste cubuk)
    fig, eksen = plt.subplots(1, 2, figsize=(11, 4))
    renkler = {"train": "#3b6ea5", "val": "#7fb069", "test": "#c9733a"}
    for e, (ad, tablo) in zip(eksen, tablolar.items()):
        t = tablo.drop("Toplam")                    # toplam satirini grafige koymuyorum
        alt = None                                  # cubuklari ust uste koymak icin baslangic yuksekligi
        for kume in ["train", "val", "test"]:
            e.bar(t.index, t[kume], bottom=alt, label=kume, color=renkler[kume])
            alt = t[kume] if alt is None else alt + t[kume]
        e.set_title(f"{ad} - 4 sinif, grup bazli bolme")
        e.set_ylabel("Goruntu sayisi")
        e.legend()
    plt.tight_layout()
    plt.savefig(GRAFIK / "fig5_bolme_dagilimi.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
