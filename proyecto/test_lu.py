import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset_lu import NopalDataset
from model_lu import SwinUNet
from config import VAL_IMAGES_DIR, VAL_MASKS_DIR, BATCH_SIZE, NUM_WORKERS, IMAGE_SIZE
import torchvision.transforms as transforms
from dataset_lu import NopalDataset, transform


def test(model_path):
    val_dataset = NopalDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

    model = SwinUNet(num_classes=3, img_size=100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    os.makedirs("results", exist_ok=True)

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            if idx == 9:
                break
            image_tensor = batch["image"].to(device)
            gt_mask_tensor = batch["mask"]  # [1, 3, H, W]

            output = model(image_tensor)  # [1, 3, H, W]
            pred_mask = output[0].cpu().numpy()  # [3, H, W]
            pred_mask_uint8 = (np.clip(pred_mask, 0, 1) * 255).astype(np.uint8)
            pred_mask_uint8 = pred_mask_uint8.transpose(1,2,0)  # [H, W, 3]

            gt_mask = gt_mask_tensor[0].cpu().numpy()  # [3, H, W]
            gt_mask_uint8 = (np.clip(gt_mask, 0, 1) * 255).astype(np.uint8)
            gt_mask_uint8 = gt_mask_uint8.transpose(1,2,0)

            original = image_tensor.squeeze().cpu().numpy()
            original = original.transpose(1, 2, 0)
            original_disp = (original * 255).astype(np.uint8)

            mask_file = val_dataset.mask_files[idx]
            mask_path = os.path.join(VAL_MASKS_DIR, mask_file)
            mask_original_color = cv2.imread(mask_path)
            mask_original_color = cv2.cvtColor(mask_original_color, cv2.COLOR_BGR2RGB)
            mask_original_color = cv2.resize(mask_original_color, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

            # Overlay: mezcla predicción y original de la máscara
            overlay = cv2.addWeighted(mask_original_color, 0.5, pred_mask_uint8, 0.5, 0)

            # Visualización
            fig, axs = plt.subplots(1, 5, figsize=(24, 5))
            axs[0].imshow(original_disp)
            axs[0].set_title("Imagen Original")
            axs[0].axis('off')

            axs[1].imshow(mask_original_color)
            axs[1].set_title("Máscara GT (Color)")
            axs[1].axis('off')

            axs[2].imshow(gt_mask_uint8)
            axs[2].set_title("Máscara GT (Normalizada)")
            axs[2].axis('off')

            axs[3].imshow(pred_mask_uint8)
            axs[3].set_title("Predicción RGB")
            axs[3].axis('off')

            axs[4].imshow(overlay)
            axs[4].set_title("Overlay (Pred + GT)")
            axs[4].axis('off')

            plt.tight_layout()
            plt.show()

            result_path = os.path.join("results", f"result_{idx}.png")
            cv2.imwrite(result_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            print(f"Resultado guardado en {result_path}")
def test_multiple_epochs(model_epochs=[10, 20, 30, 40], sample_idx=0):
    val_dataset = NopalDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample = next(iter(DataLoader(val_dataset, batch_size=1, shuffle=False)))

    image_tensor = sample["image"].to(device)
    gt_mask_tensor = sample["mask"]

    original = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    original_disp = (original * 255).astype(np.uint8)

    mask_file = val_dataset.mask_files[sample_idx]
    mask_path = os.path.join(VAL_MASKS_DIR, mask_file)
    mask_original_color = cv2.imread(mask_path)
    mask_original_color = cv2.cvtColor(mask_original_color, cv2.COLOR_BGR2RGB)
    mask_original_color = cv2.resize(mask_original_color, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

    gt_mask = gt_mask_tensor[0].cpu().numpy().transpose(1, 2, 0)
    gt_mask_uint8 = (np.clip(gt_mask, 0, 1) * 255).astype(np.uint8)

    fig, axs = plt.subplots(len(model_epochs), 5, figsize=(25, 5 * len(model_epochs)))

    for i, epoch in enumerate(model_epochs):
        model_path = f"modelFile/model_epoch_{epoch}.pth"
        model = SwinUNet(num_classes=3, img_size=100).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = output[0].cpu().numpy().transpose(1, 2, 0)
            pred_mask_uint8 = (np.clip(pred_mask, 0, 1) * 255).astype(np.uint8)
            overlay = cv2.addWeighted(mask_original_color, 0.5, pred_mask_uint8, 0.5, 0)

        axs[i][0].imshow(original_disp)
        axs[i][0].set_title(f"Original (Época {epoch})")
        axs[i][1].imshow(mask_original_color)
        axs[i][1].set_title("Máscara GT (Color)")
        axs[i][2].imshow(gt_mask_uint8)
        axs[i][2].set_title("Máscara GT (Normalizada)")
        axs[i][3].imshow(pred_mask_uint8)
        axs[i][3].set_title("Predicción")
        axs[i][4].imshow(overlay)
        axs[i][4].set_title("Overlay")

        for j in range(5):
            axs[i][j].axis('off')

    plt.tight_layout()
    plt.show()


def calculate_iou(gt_mask, pred_mask):
    """Calcula el IoU por clase y promedia"""


    gt_bin = gt_mask.argmax(axis=2)  # [H, W]
    pred_bin = pred_mask.argmax(axis=2)
    print("GT unique labels:", np.unique(gt_bin))
    print("Pred unique labels:", np.unique(pred_bin))
    intersection = np.logical_and(gt_bin == pred_bin, gt_bin > 0).sum()
    union = np.logical_or(gt_bin > 0, pred_bin > 0).sum()
    if union == 0: return 100.0  # Evita división por 0
    iou = (intersection / union) * 100
    return round(iou, 2)

def calculate_soft_iou(gt_mask, pred_mask, threshold=0.5):
    """
    IoU suave para regresión RGB. Binariza por canal y promedia el IoU por canal.
    """
    ious = []
    for c in range(3):
        gt_c = (gt_mask[..., c] >= threshold).astype(np.uint8)
        pred_c = (pred_mask[..., c] >= threshold).astype(np.uint8)
        intersection = np.logical_and(gt_c, pred_c).sum()
        union = np.logical_or(gt_c, pred_c).sum()
        if union == 0:
            ious.append(1.0)  # No hay píxeles de clase → perfecto
        else:
            ious.append(intersection / union)
    return round(np.mean(ious) * 100, 2)

def test_visual_summary(model_epochs=[10, 20, 30, 40, 50], sample_idx=0):
    val_dataset = NopalDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, transform=transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Obtener muestra fija
    sample = val_dataset[sample_idx]
    image_tensor = sample["image"].unsqueeze(0).to(device)
    gt_mask_tensor = sample["mask"].unsqueeze(0)  # [1, 3, H, W]

    # Procesar para visualización
    original_disp = (image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    mask_file = val_dataset.mask_files[sample_idx]
    mask_path = os.path.join(VAL_MASKS_DIR, mask_file)
    mask_original_color = cv2.imread(mask_path)
    mask_original_color = cv2.cvtColor(mask_original_color, cv2.COLOR_BGR2RGB)
    mask_original_color = cv2.resize(mask_original_color, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)

    gt_mask_norm = gt_mask_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    gt_mask_norm_uint8 = (np.clip(gt_mask_norm, 0, 1) * 255).astype(np.uint8)

    # ───────────────── VISUALIZACIÓN ────────────────
    total_rows = len(model_epochs) + 1
    fig, axs = plt.subplots(total_rows, 3, figsize=(18, 4 * total_rows))

    # Primera fila: original + máscaras
    axs[0][0].imshow(original_disp)
    axs[0][0].set_title("Imagen Original")
    axs[0][1].imshow(mask_original_color)
    axs[0][1].set_title("Máscara GT (Color)")
    axs[0][2].imshow(gt_mask_norm_uint8)
    axs[0][2].set_title("Máscara GT (Normalizada)")

    for col in range(3):
        axs[0][col].axis('off')

    # Resto de filas por época
    for i, epoch in enumerate(model_epochs):
        model_path = f"modelFile/model_epoch_{epoch}.pth"
        model = SwinUNet(num_classes=3, img_size=100).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        with torch.no_grad():
            output = model(image_tensor)
            pred_mask = output[0].cpu().numpy().transpose(1, 2, 0)  # [H, W, 3]
            pred_mask_uint8 = (np.clip(pred_mask, 0, 1) * 255).astype(np.uint8)

        # Overlay visual
        overlay = cv2.addWeighted(mask_original_color, 0.5, pred_mask_uint8, 0.5, 0)

        # IoU cálculo
        #iou_val = calculate_iou(gt_mask_norm, pred_mask)
        iou_val = calculate_soft_iou(gt_mask_norm, pred_mask)

        axs[i + 1][0].imshow(pred_mask_uint8)
        axs[i + 1][0].set_title(f"Predicción – Época {epoch}")
        axs[i + 1][1].imshow(overlay)
        axs[i + 1][1].set_title(f"Overlay – IoU: {iou_val}%")
        axs[i + 1][2].axis('off')  # Celda vacía para simetría

        axs[i + 1][0].axis('off')
        axs[i + 1][1].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    #model_checkpoint = "modelFile/model_epoch_40.pth"
    #test(model_checkpoint)
    #test_multiple_epochs(model_epochs=[10, 20, 30, 40,40], sample_idx=0)
    test_visual_summary(model_epochs=[10, 20, 30, 40, 50], sample_idx=10)

