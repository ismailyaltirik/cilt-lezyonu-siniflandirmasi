# image_pipeline.py
# Faz 1 - Goruntu on isleme ve veri artirma adimlari (PyTorch / torchvision)
# Goruntuler indirildikten sonra model egitiminde bu donusumleri kullanacagim.
# Sinif agirliklarini SADECE egitim kumesinden hesapliyorum, yoksa test bilgisi egitime sizar.

import json
from pathlib import Path
import pandas as pd

ANA_KLASOR = Path(__file__).resolve().parents[1]
GORUNTU_BOYUTU = 224                        # ResNet, MobileNet, EfficientNet 224x224 giris bekliyor
IMAGENET_ORT = [0.485, 0.456, 0.406]        # ImageNet goruntulerinin R, G, B ortalamasi
IMAGENET_STD = [0.229, 0.224, 0.225]        # ImageNet goruntulerinin R, G, B standart sapmasi
SINIFLAR = ["AK", "BCC", "MEL", "NEV"]


def donusumleri_olustur():
    # torchvision'i fonksiyon icinde import ediyorum, boylece sinif agirligi hesabi torch olmadan da calisir
    from torchvision import transforms as T
    egitim_donusumu = T.Compose([
        T.Resize((GORUNTU_BOYUTU, GORUNTU_BOYUTU)),    # HAM 600x450, PAD farkli boyutlarda -> hepsi 224x224
        T.RandomHorizontalFlip(),                      # yatay cevirme (lezyonun yonu tani degistirmez)
        T.RandomVerticalFlip(),                        # dikey cevirme
        T.RandomRotation(20),                          # -20 ile +20 derece arasi dondurme
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),  # telefonlardaki isik farkina karsi
        T.ToTensor(),                                  # goruntuyu 0-1 arasi tensore cevir
        T.Normalize(IMAGENET_ORT, IMAGENET_STD),       # onceden egitilmis agin bekledigi dagilima getir
    ])
    # dogrulama ve testte veri artirma YOK, sadece boyutlandirma ve normalizasyon
    test_donusumu = T.Compose([
        T.Resize((GORUNTU_BOYUTU, GORUNTU_BOYUTU)),
        T.ToTensor(),
        T.Normalize(IMAGENET_ORT, IMAGENET_STD),
    ])
    return egitim_donusumu, test_donusumu


def sinif_agirliklari(bolme_dosyasi):
    # Az ornekli siniflara daha buyuk agirlik veriyorum (ters frekans yontemi)
    df = pd.read_csv(bolme_dosyasi)
    egitim = df[df["split"] == "train"]                          # sadece egitim kumesi
    sayilar = egitim["label"].value_counts().reindex(SINIFLAR)   # her siniftan kac goruntu var
    # formul: toplam / (sinif sayisi * o sinifin sayisi)
    agirlik = len(egitim) / (len(SINIFLAR) * sayilar)
    return sayilar.to_dict(), agirlik.round(3).to_dict()


if __name__ == "__main__":
    sonuc = {}
    for ad, dosya in (("HAM10000", "ham10000_4class_split.csv"),
                      ("PAD-UFES-20", "pad_ufes20_4class_split.csv")):
        sayilar, agirlik = sinif_agirliklari(ANA_KLASOR / "data" / "splits" / dosya)
        sonuc[ad] = {"egitim_sayilari": sayilar, "sinif_agirliklari": agirlik}
    with open(ANA_KLASOR / "results" / "sinif_agirliklari.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, indent=2, ensure_ascii=False, default=int)
    print(json.dumps(sonuc, indent=2, ensure_ascii=False, default=int))
