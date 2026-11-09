# ornek_goruntuler.py
# Goruntu dosyalari indirildikten sonra calistirilacak:
# - her siniftan ornek goruntu izgarasi ciziyor
# - goruntu boyutlarini ve RGB kanal ortalamalarini hesapliyor
# Kullanim: goruntuleri data/images/HAM10000/ ve data/images/PAD-UFES-20/ altina koyup
#           python src/ornek_goruntuler.py

from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image                     # goruntu acmak icin
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ANA_KLASOR = Path(__file__).resolve().parents[1]
GORUNTU = ANA_KLASOR / "data" / "images"
BOLME = ANA_KLASOR / "data" / "splits"
SINIFLAR = ["AK", "BCC", "MEL", "NEV"]


def dosya_bul(klasor, ad):
    # Kaggle/Mendeley arsivleri goruntuleri alt klasorlere dagitiyor, o yuzden her yerde ariyorum
    bulunan = list(klasor.rglob(ad))
    return bulunan[0] if bulunan else None


def izgara_ciz(df, kimlik_sutunu, klasor, uzanti, baslik, cikti, adet=4):
    # her sinif icin bir satir, her satirda 'adet' kadar ornek goruntu
    fig, eksen = plt.subplots(len(SINIFLAR), adet, figsize=(adet * 2.2, len(SINIFLAR) * 2.2))
    for i, sinif in enumerate(SINIFLAR):
        ornekler = df[df["label"] == sinif].sample(adet, random_state=0)   # rastgele ama sabit ornek
        for j, kimlik in enumerate(ornekler[kimlik_sutunu]):
            yol = dosya_bul(klasor, kimlik + uzanti)
            if yol is not None:
                eksen[i, j].imshow(Image.open(yol).convert("RGB"))
            eksen[i, j].axis("off")
        eksen[i, 0].set_title(sinif, loc="left", fontsize=10)
    fig.suptitle(baslik)
    plt.tight_layout()
    plt.savefig(ANA_KLASOR / "figures" / cikti, dpi=150)
    plt.close()


def boyut_ve_renk(df, kimlik_sutunu, klasor, uzanti, adet=300):
    # hiz icin 300 goruntuluk ornekle boyut ve RGB kanal ortalamasini hesapliyorum
    genislik, yukseklik, rgb = [], [], []
    for kimlik in df[kimlik_sutunu].sample(min(adet, len(df)), random_state=0):
        yol = dosya_bul(klasor, kimlik + uzanti)
        if yol is None:
            continue
        resim = Image.open(yol)
        genislik.append(resim.width)
        yukseklik.append(resim.height)
        rgb.append(np.asarray(resim.convert("RGB")).reshape(-1, 3).mean(axis=0))  # R, G, B ortalamasi
    print("Genislik min/maks:", min(genislik), max(genislik))
    print("Yukseklik min/maks:", min(yukseklik), max(yukseklik))
    print("Ortalama R, G, B:", np.mean(rgb, axis=0).round(1))


if __name__ == "__main__":
    ham = pd.read_csv(BOLME / "ham10000_4class_split.csv")
    pad = pd.read_csv(BOLME / "pad_ufes20_4class_split.csv")
    izgara_ciz(ham, "image_id", GORUNTU / "HAM10000", ".jpg", "HAM10000 ornek goruntuler", "fig6_ham_ornekler.png")
    izgara_ciz(pad, "img_id", GORUNTU / "PAD-UFES-20", "", "PAD-UFES-20 ornek goruntuler", "fig7_pad_ornekler.png")
    boyut_ve_renk(ham, "image_id", GORUNTU / "HAM10000", ".jpg")
    boyut_ve_renk(pad, "img_id", GORUNTU / "PAD-UFES-20", "")
