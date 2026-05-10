import cv2
import numpy as np
import os
import glob

IMG_DIR = "Dataset/Image"
OUT_DIR = "Outputs/Clustering"
os.makedirs(OUT_DIR, exist_ok=True)

for img_path in glob.glob(os.path.join(IMG_DIR, "*.png")):
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)

    # Görüntüyü L*a*b* renk uzayına çevir (H&E görselleri için en iyisi)
    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Pikselleri 2 boyutlu bir diziye düzleştir
    pixel_values = lab_image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # K-Means Algoritması Kriterleri ve Uygulanması (K=3)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 3
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # En karanlık kümeyi (çekirdekleri) bul. L (Lightness) indeksi 0'dır.
    darkest_cluster = np.argmin(centers[:, 0])

    # Sadece çekirdekleri içeren bir maske oluştur
    mask = np.zeros_like(labels, dtype=np.uint8)
    mask[labels == darkest_cluster] = 255
    mask = mask.reshape((img.shape[0], img.shape[1]))

    # Morfolojik temizleme
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    cv2.imwrite(os.path.join(OUT_DIR, filename), mask_cleaned)

print("Clustering (K-Means) işlemi tamamlandı!")