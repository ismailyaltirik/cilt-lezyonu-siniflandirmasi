# eda.py
# Faz 1 - Kesifsel veri analizi (EDA)
# HAM10000 ve PAD-UFES-20 veri setlerinin metadata dosyalarini inceliyorum.
# Bakilanlar: sinif dagilimi, eksik veri, tekrar eden kayit, lezyon/hasta basina goruntu sayisi.

import json                      # sonuclari json dosyasina kaydetmek icin
from pathlib import Path         # dosya yollarini kolay yonetmek icin
import pandas as pd              # csv dosyalarini tablo olarak okumak icin
import matplotlib                # grafik cizmek icin
matplotlib.use("Agg")            # ekran olmadan (sunucuda) da grafik kaydedebilmek icin
import matplotlib.pyplot as plt  # grafik fonksiyonlari

# Proje klasorleri
ANA_KLASOR = Path(__file__).resolve().parents[1]        # src klasorunun bir ustu = proje klasoru
METADATA = ANA_KLASOR / "data" / "metadata"             # orijinal metadata dosyalari burada
GRAFIK = ANA_KLASOR / "figures"                         # grafikleri buraya kaydedecegim
SONUC = ANA_KLASOR / "results"                          # sayisal sonuclari buraya kaydedecegim
GRAFIK.mkdir(exist_ok=True)                             # klasor yoksa olustur
SONUC.mkdir(exist_ok=True)                              # klasor yoksa olustur

MAVI = "#3b6ea5"     # HAM10000 grafiklerinin rengi
TURUNCU = "#c9733a"  # PAD-UFES-20 grafiklerinin rengi


def verileri_oku():
    # Iki veri setinin metadata dosyalarini okuyup geri dondurur
    ham = pd.read_csv(METADATA / "HAM10000_metadata.csv")      # 10015 satir, her satir bir goruntu
    pad = pd.read_csv(METADATA / "PAD-UFES-20_metadata.csv")   # 2298 satir, her satir bir goruntu
    return ham, pad


def istatistikleri_hesapla(ham, pad):
    # Rapora yazacagim temel sayilari hesapliyorum
    sonuc = {
        "HAM10000": {
            "goruntu": len(ham),                                         # toplam goruntu sayisi
            "lezyon": ham["lesion_id"].nunique(),                        # farkli lezyon sayisi
            "sinif_sayisi": ham["dx"].nunique(),                         # kac farkli tani var
            "sinif_dagilimi": ham["dx"].value_counts().to_dict(),        # her sinifta kac goruntu var
            "yas_eksik": int(ham["age"].isna().sum()),                   # yasi bos olan kayit sayisi
            "cinsiyet_unknown": int((ham["sex"] == "unknown").sum()),    # cinsiyeti bilinmeyen kayit
            "lokasyon_unknown": int((ham["localization"] == "unknown").sum()),  # bolgesi bilinmeyen kayit
            "tekrar_eden_image_id": int(ham["image_id"].duplicated().sum()),    # ayni goruntu iki kez var mi
            # birden fazla fotografi olan lezyon sayisi (sizinti riskinin kaynagi)
            "birden_fazla_goruntulu_lezyon": int((ham["lesion_id"].value_counts() > 1).sum()),
            "tani_yontemi": ham["dx_type"].value_counts().to_dict(),     # histo, follow_up vb.
        },
        "PAD-UFES-20": {
            "goruntu": len(pad),                                         # toplam goruntu sayisi
            "lezyon": pad["lesion_id"].nunique(),                        # farkli lezyon sayisi
            "hasta": pad["patient_id"].nunique(),                        # farkli hasta sayisi
            "sinif_sayisi": pad["diagnostic"].nunique(),                 # kac farkli tani var
            "sinif_dagilimi": pad["diagnostic"].value_counts().to_dict(),
            "tekrar_eden_img_id": int(pad["img_id"].duplicated().sum()),
            "biyopsi_orani": round(float(pad["biopsed"].mean()), 3),     # biyopsi ile dogrulanan goruntu orani
            "eksik_deger_orani": pad.isna().mean().round(3).to_dict(),   # her sutundaki bos deger orani
        },
    }
    # sonuclari json olarak kaydet (default=int: numpy sayilarini normal sayiya cevirir)
    with open(SONUC / "eda_istatistikler.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, indent=2, ensure_ascii=False, default=int)
    return sonuc


def sinif_grafigi(ham, pad):
    # Grafik 1: iki veri setinin sinif dagilimi yan yana
    fig, eksen = plt.subplots(1, 2, figsize=(11, 4))
    ham_sayim = ham["dx"].value_counts()             # HAM10000 sinif sayilari (buyukten kucuge)
    pad_sayim = pad["diagnostic"].value_counts()     # PAD-UFES-20 sinif sayilari
    eksen[0].bar(ham_sayim.index, ham_sayim.values, color=MAVI)
    eksen[0].set_title("HAM10000 - sinif dagilimi (n=10.015)")
    eksen[1].bar(pad_sayim.index, pad_sayim.values, color=TURUNCU)
    eksen[1].set_title("PAD-UFES-20 - sinif dagilimi (n=2.298)")
    for e, sayim in ((eksen[0], ham_sayim), (eksen[1], pad_sayim)):
        e.set_ylabel("Goruntu sayisi")
        for i, deger in enumerate(sayim.values):            # her cubugun ustune sayisini yaz
            e.text(i, deger, str(deger), ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(GRAFIK / "fig1_sinif_dagilimi.png", dpi=150)
    plt.close()


def tekrar_grafigi(ham, pad):
    # Grafik 2: bir lezyonun (HAM) veya bir hastanin (PAD) kac goruntusu var
    fig, eksen = plt.subplots(1, 2, figsize=(11, 3.8))
    # once her lezyonun goruntu sayisini bul, sonra "kac lezyonun 1, 2, 3... goruntusu var" diye say
    ham_tekrar = ham["lesion_id"].value_counts().value_counts().sort_index()
    eksen[0].bar(ham_tekrar.index.astype(str), ham_tekrar.values, color=MAVI)
    eksen[0].set_title("HAM10000 - lezyon basina goruntu sayisi")
    eksen[0].set_xlabel("Goruntu / lezyon")
    eksen[0].set_ylabel("Lezyon sayisi")
    pad_tekrar = pad.groupby("patient_id")["img_id"].count().value_counts().sort_index()
    eksen[1].bar(pad_tekrar.index.astype(str), pad_tekrar.values, color=TURUNCU)
    eksen[1].set_title("PAD-UFES-20 - hasta basina goruntu sayisi")
    eksen[1].set_xlabel("Goruntu / hasta")
    eksen[1].set_ylabel("Hasta sayisi")
    plt.tight_layout()
    plt.savefig(GRAFIK / "fig2_lezyon_hasta_basina_goruntu.png", dpi=150)
    plt.close()


def demografi_grafigi(ham, pad):
    # Grafik 3: yas dagilimi ve lezyonun vucuttaki bolgesi
    fig, eksen = plt.subplots(1, 3, figsize=(13, 3.8))
    # density=True: iki setin boyutu farkli oldugu icin oran olarak karsilastiriyorum
    eksen[0].hist(ham["age"].dropna(), bins=18, alpha=0.7, label="HAM10000", color=MAVI, density=True)
    eksen[0].hist(pad["age"].dropna(), bins=18, alpha=0.6, label="PAD-UFES-20", color=TURUNCU, density=True)
    eksen[0].set_title("Yas dagilimi")
    eksen[0].set_xlabel("Yas")
    eksen[0].legend()
    ham_bolge = ham["localization"].value_counts().head(8)   # en sik 8 bolge
    eksen[1].barh(ham_bolge.index[::-1], ham_bolge.values[::-1], color=MAVI)
    eksen[1].set_title("HAM10000 - lezyon bolgesi (ilk 8)")
    pad_bolge = pad["region"].value_counts().head(8)
    eksen[2].barh(pad_bolge.index[::-1], pad_bolge.values[::-1], color=TURUNCU)
    eksen[2].set_title("PAD-UFES-20 - lezyon bolgesi (ilk 8)")
    plt.tight_layout()
    plt.savefig(GRAFIK / "fig3_demografi.png", dpi=150)
    plt.close()


def eksik_veri_grafigi(pad):
    # Grafik 4: PAD-UFES-20'de hangi sutunda yuzde kac bos deger var
    oran = (pad.isna().mean() * 100).sort_values(ascending=False)  # yuzdeye cevir ve sirala
    oran = oran[oran > 0]                                          # sadece eksigi olan sutunlar
    fig, eksen = plt.subplots(figsize=(8, 4))
    eksen.barh(oran.index[::-1], oran.values[::-1], color="#8a8a8a")
    eksen.set_xlabel("Eksik deger orani (%)")
    eksen.set_title("PAD-UFES-20 - metadata eksik deger oranlari")
    plt.tight_layout()
    plt.savefig(GRAFIK / "fig4_pad_eksik_veri.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    ham, pad = verileri_oku()                     # 1) verileri oku
    sonuc = istatistikleri_hesapla(ham, pad)      # 2) sayilari hesapla
    sinif_grafigi(ham, pad)                       # 3) grafikleri ciz
    tekrar_grafigi(ham, pad)
    demografi_grafigi(ham, pad)
    eksik_veri_grafigi(pad)
    print(json.dumps(sonuc, indent=2, ensure_ascii=False, default=int))   # ekrana da yazdir
