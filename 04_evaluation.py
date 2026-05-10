import cv2
import numpy as np
import os
import glob

# Klasör Yolları
IMG_DIR = "Dataset/Image"
GT_DIR = "Dataset/Masks"
METHODS = ["Thresholding", "Clustering", "Watershed"]
BASE_OUT_DIR = "Outputs"

def calculate_metrics(gt_mask, pred_mask):
    """Ground Truth ve Tahmin edilen maske arasındaki IoU ve Dice metriklerini hesaplar."""
    # İkili (binary) formata çevir (0 veya 1)
    gt = (gt_mask > 127).astype(np.uint8)
    pred = (pred_mask > 127).astype(np.uint8)

    # Kesişim (Intersection) ve Birleşim (Union)
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)

    tp = np.sum(intersection)
    fp = np.sum(pred) - tp
    fn = np.sum(gt) - tp

    # Sıfıra bölme hatasını engellemek için paydaya çok küçük bir sayı (1e-6) ekliyoruz
    iou = tp / (tp + fp + fn + 1e-6)
    dice = (2 * tp) / ((2 * tp) + fp + fn + 1e-6)

    return iou, dice

def create_overlay(original_img, pred_mask):
    """Orijinal görüntünün üzerine tahmin edilen sınırları (contours) çizer."""
    # Tahmin edilen maskenin sınırlarını bul
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Orijinal görüntünün kopyasını al
    overlay_img = original_img.copy()

    # Sınırları dikkat çekici yeşil renkte (0, 255, 0) ve 2 piksel kalınlığında çiz
    cv2.drawContours(overlay_img, contours, -1, (0, 255, 0), 2)
    return overlay_img

# Konsola yazdırılacak tablonun başlığı
print(f"{'Method':<15} | {'Mean IoU':<10} | {'Mean Dice':<10}")
print("-" * 40)

# Her bir algoritma ailesi için değerlendirme döngüsü
for method in METHODS:
    method_dir = os.path.join(BASE_OUT_DIR, method)
    # Overlay görselleri için yeni bir çıktı klasörü oluştur
    overlay_out_dir = os.path.join(BASE_OUT_DIR, "Evaluation", f"{method}_Overlays")
    os.makedirs(overlay_out_dir, exist_ok=True)

    iou_list = []
    dice_list = []

    # Dataset/Image içindeki tüm görselleri sırayla gez
    for img_path in glob.glob(os.path.join(IMG_DIR, "*.png")):
        filename = os.path.basename(img_path)

        # İlgili dosyaları oku
        img = cv2.imread(img_path)
        gt_mask = cv2.imread(os.path.join(GT_DIR, filename), cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(os.path.join(method_dir, filename), cv2.IMREAD_GRAYSCALE)

        # Eğer maske eksikse hata vermemesi için atla
        if gt_mask is None or pred_mask is None:
            continue

        # 1. Metrikleri hesapla ve listeye ekle
        iou, dice = calculate_metrics(gt_mask, pred_mask)
        iou_list.append(iou)
        dice_list.append(dice)

        # 2. Çakıştırma (Overlay) görselini oluştur ve kaydet
        overlay = create_overlay(img, pred_mask)
        cv2.imwrite(os.path.join(overlay_out_dir, filename), overlay)

    # İlgili yöntem için 16 görselin metrik ortalamasını al
    mean_iou = np.mean(iou_list) if iou_list else 0
    mean_dice = np.mean(dice_list) if dice_list else 0

    # Konsola yazdır
    print(f"{method:<15} | {mean_iou:.4f}     | {mean_dice:.4f}")

print("\nDeğerlendirme ve Overlay işlemi tamamlandı! Görseller Outputs/Evaluation klasöründe.")