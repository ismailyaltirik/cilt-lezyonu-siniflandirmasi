# Cilt Lezyonu Sınıflandırması – HAM10000 & PAD-UFES-20

Uygulamalı Yapay Zeka dönem projesi · İsmail Bayhan Yaltırık (F241230022)

Dermatoskopik (HAM10000) ve akıllı telefon (PAD-UFES-20) görüntülerinde ortak 4 sınıf
(AK, BCC, MEL, NEV) için CNN mimarilerinin hasta/lezyon bazlı sızıntısız bölmeyle karşılaştırılması
ve dermatoskopikten telefon görüntüsüne genellenebilirliğin ölçülmesi.

## Veri setleri
| Veri seti | Kaynak | DOI |
|---|---|---|
| HAM10000 (V4) | Harvard Dataverse | 10.7910/DVN/DBW86T |
| PAD-UFES-20 (V1) | Mendeley Data | 10.17632/zr7vgbcyr2.1 |

`data/metadata/` altında iki veri setinin orijinal metadata dosyaları bulunur. Görüntü dosyaları
boyutları nedeniyle repoya eklenmemiştir; yukarıdaki bağlantılardan indirilip `data/images/` altına konulmalıdır.

## Klasör yapısı
```
data/metadata/   orijinal metadata (CSV)
data/splits/     4 sınıflı, grup bazlı train/val/test bölmeleri
src/eda.py               keşifsel veri analizi ve grafikler
src/preprocess_split.py  etiket eşleme, grup bazlı bölme, sızıntı kontrolü
src/image_pipeline.py    görüntü ön işleme / veri artırma, sınıf ağırlıkları
src/ornek_goruntuler.py  görüntüler indirilince örnek ızgara + boyut/RGB analizi
figures/         analiz grafikleri
results/         istatistik çıktıları (JSON)
```

## Çalıştırma
```
pip install -r requirements.txt
python src/eda.py
python src/preprocess_split.py
python src/image_pipeline.py
# görüntüler data/images/ altına indirildikten sonra:
python src/ornek_goruntuler.py
```

## Faz 1 özeti
- HAM10000: 10.015 görüntü / 7.470 lezyon; 1.956 lezyonun birden fazla görüntüsü var.
- PAD-UFES-20: 2.298 görüntü / 1.641 lezyon / 1.373 hasta; yalnızca 52 MEL.
- Bölme (%70/15/15): HAM 6.185/1.237/1.237, PAD 1.336/267/268; kümeler arası lezyon/hasta kesişimi 0.

## Planlanan modeller (Faz 2)
ResNet50, MobileNetV2, EfficientNet-B0 (ImageNet ön eğitimli), küçük CNN (sıfırdan) ve MLP (uygun olmayan model).
Metrikler: macro-F1, dengeli doğruluk, melanom duyarlılığı, ROC-AUC.
