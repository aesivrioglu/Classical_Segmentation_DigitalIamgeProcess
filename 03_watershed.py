import cv2
import numpy as np
import os
import glob

IMG_DIR = "Dataset/Image"
OUT_DIR = "Outputs/Watershed"
os.makedirs(OUT_DIR, exist_ok=True)

for img_path in glob.glob(os.path.join(IMG_DIR, "*.png")):
    filename = os.path.basename(img_path)
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Kaba bir arka plan/ön plan ayrımı (Otsu Thresholding)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 2. Gürültü temizleme (Açma işlemi)
    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 3. Kesin arka planı bul (Genişletme ile)
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # 4. Kesin ön planı (hücre merkezlerini) bul -> DISTANCE TRANSFORM
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)

    # 5. Bilinmeyen bölgeleri (sınırları) bul
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # 6. Marker (Etiket) oluşturma
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1 # Arka plan 0 olmasın diye 1 ekliyoruz
    markers[unknown == 255] = 0 # Bilinmeyen bölgeleri 0 yapıyoruz

    # 7. Watershed Uygula
    markers = cv2.watershed(img, markers)

    # Sonucu ikili maskeye (Binary Mask) dönüştür
    # Sınırlar -1, arka plan 1'dir. Biz sadece hücreleri (1'den büyük olanları) istiyoruz.
    output_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    output_mask[markers > 1] = 255

    cv2.imwrite(os.path.join(OUT_DIR, filename), output_mask)

print("Watershed işlemi tamamlandı!")