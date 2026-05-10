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
    gt = (gt_mask > 127).astype(np.uint8)
    pred = (pred_mask > 127).astype(np.uint8)

    intersection = np.logical_and(gt, pred)
    tp = np.sum(intersection)
    fp = np.sum(pred) - tp
    fn = np.sum(gt) - tp

    iou = tp / (tp + fp + fn + 1e-6)
    dice = (2 * tp) / ((2 * tp) + fp + fn + 1e-6)

    return iou, dice

def create_overlay(original_img, pred_mask):
    """Orijinal görüntünün üzerine tahmin edilen sınırları (contours) çizer."""
    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay_img = original_img.copy()
    cv2.drawContours(overlay_img, contours, -1, (0, 255, 0), 2)
    return overlay_img

print(f"{'Method':<15} | {'Mean IoU':<10} | {'Mean Dice':<10}")
print("-" * 40)

for method in METHODS:
    method_dir = os.path.join(BASE_OUT_DIR, method)
    overlay_out_dir = os.path.join(BASE_OUT_DIR, "Evaluation", f"{method}_Overlays")
    os.makedirs(overlay_out_dir, exist_ok=True)

    iou_list = []
    dice_list = []

    for img_path in glob.glob(os.path.join(IMG_DIR, "*.png")):
        filename = os.path.basename(img_path)

        # DOSYA İSMİ DÜZELTME MANTIĞI:
        # "01_1 .png" ismindeki fazladan boşluğu ve uzantıyı atıp sadece "01_1" kısmını alıyoruz.
        base_name = os.path.splitext(filename)[0].strip()

        # Masks klasöründe bu "01_1" ile başlayan dosyayı buluyoruz (uzantısı veya takısı ne olursa olsun)
        mask_search_path = os.path.join(GT_DIR, f"{base_name}*")
        matching_masks = glob.glob(mask_search_path)

        if not matching_masks:
            print(f"UYARI: {base_name} için Masks klasöründe maske bulunamadı! Lütfen Masks klasörünü kontrol et.")
            continue

        gt_mask_path = matching_masks[0] # Bulunan ilk eşleşen maskeyi al

        # İlgili dosyaları oku
        img = cv2.imread(img_path)
        gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(os.path.join(method_dir, filename), cv2.IMREAD_GRAYSCALE)

        if gt_mask is None or pred_mask is None:
            continue

        iou, dice = calculate_metrics(gt_mask, pred_mask)
        iou_list.append(iou)
        dice_list.append(dice)

        overlay = create_overlay(img, pred_mask)
        cv2.imwrite(os.path.join(overlay_out_dir, filename), overlay)

    mean_iou = np.mean(iou_list) if iou_list else 0
    mean_dice = np.mean(dice_list) if dice_list else 0

    print(f"{method:<15} | {mean_iou:.4f}     | {mean_dice:.4f}")

print("\nDeğerlendirme ve Overlay işlemi tamamlandı! Görseller Outputs/Evaluation klasöründe.")