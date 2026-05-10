import cv2
import os
import glob
import numpy as np

# Dizin Yolları
IMG_DIR = "Dataset/Image"
OUT_DIR = "Outputs/Thresholding"

# Çıktı klasörünü oluştur (yoksa)
os.makedirs(OUT_DIR, exist_ok=True)

# Dataset/Image içindeki tüm png dosyalarını oku
for img_path in glob.glob(os.path.join(IMG_DIR, "*.png")):
    filename = os.path.basename(img_path)

    # Görseli siyah-beyaz (Grayscale) formatta oku
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Adaptif Eşikleme (Çekirdekler koyu olduğu için INV kullanıyoruz)
    # Parametreler (21, 10): 21x21 piksellik komşuluk alanına bak, ortalamadan 10 çıkar
    thresh = cv2.adaptiveThreshold(img_gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 21, 10)

    # Morfolojik İşlem (Ufak gürültüleri/noktaları temizlemek için)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cleaned_mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Sonucu kaydet
    cv2.imwrite(os.path.join(OUT_DIR, filename), cleaned_mask)

print("Thresholding işlemi tamamlandı!")