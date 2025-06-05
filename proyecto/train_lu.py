# train_lu.py
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset_lu import NopalDataset, transform
from model_lu import SwinUNet
from config import *

# ─────────────── Métricas ───────────────
def rmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2))

def iou(pred, target, threshold=0.5):
    pred_bin = (pred >= threshold).astype(np.uint8)
    target_bin = (target >= threshold).astype(np.uint8)
    intersection = np.logical_and(pred_bin, target_bin).sum()
    union = np.logical_or(pred_bin, target_bin).sum()
    return intersection / (union + 1e-6)

def accuracy(pred, target, tolerance=0.1):
    return np.mean(np.abs(pred - target) < tolerance)

# ─────────────── Entrenamiento ───────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("📟 Dispositivo:", device)

    train_ds = NopalDataset(TRAIN_IMAGES_DIR, TRAIN_MASKS_DIR, transform=transform)
    val_ds   = NopalDataset(VAL_IMAGES_DIR, VAL_MASKS_DIR, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    model = SwinUNet(num_classes=3, img_size=IMAGE_SIZE).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler(device_type='cuda')  # ✅ ACTUALIZADO

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc",
                               "train_rmse", "val_rmse", "train_iou", "val_iou"]}

    start = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        tloss = trmse = tiou = tacc = 0.0

        for batch in train_loader:
            imgs = batch["image"].to(device, non_blocking=True)
            msk  = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):  # ✅ ACTUALIZADO
                out = model(imgs)
                loss = loss_fn(out, msk)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tloss += loss.item()
            out_np = out.detach().cpu().numpy()
            msk_np = msk.detach().cpu().numpy()
            trmse += rmse(out_np, msk_np)
            tiou  += iou(out_np, msk_np)
            tacc  += accuracy(out_np, msk_np)

        n_train = len(train_loader)
        avg_tloss = tloss / n_train
        avg_trmse = trmse / n_train
        avg_tiou  = tiou  / n_train
        avg_tacc  = tacc  / n_train

        # ─────────────── Validación ───────────────
        model.eval()
        vloss = vrmse = viou = vacc = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device, non_blocking=True)
                msk  = batch["mask"].to(device, non_blocking=True)

                out = model(imgs)
                vloss += loss_fn(out, msk).item()
                out_np = out.cpu().numpy()
                msk_np = msk.cpu().numpy()
                vrmse += rmse(out_np, msk_np)
                viou  += iou(out_np, msk_np)
                vacc  += accuracy(out_np, msk_np)

        n_val = len(val_loader)
        avg_vloss = vloss / n_val
        avg_vrmse = vrmse / n_val
        avg_viou  = viou  / n_val
        avg_vacc  = vacc  / n_val

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
              f"Train Loss: {avg_tloss:.4f} | Val Loss: {avg_vloss:.4f} | "
              f"Acc (T/V): {avg_tacc:.3f}/{avg_vacc:.3f} | "
              f"RMSE (T/V): {avg_trmse:.3f}/{avg_vrmse:.3f} | "
              f"IoU (T/V): {avg_tiou:.3f}/{avg_viou:.3f}")

        # Guardar métricas
        history["train_loss"].append(avg_tloss); history["val_loss"].append(avg_vloss)
        history["train_acc"].append(avg_tacc);   history["val_acc"].append(avg_vacc)
        history["train_rmse"].append(avg_trmse); history["val_rmse"].append(avg_vrmse)
        history["train_iou"].append(avg_tiou);   history["val_iou"].append(avg_viou)

        # Checkpoint
        if (epoch + 1) % 10 == 0:
            os.makedirs("modelFile", exist_ok=True)
            ckpt_path = os.path.join("modelFile", f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)

    print(f"\n🕒 Entrenamiento completo en {(time.time() - start)/60:.2f} min")

    # ─────────────── Gráficas ───────────────
    epochs = range(1, NUM_EPOCHS + 1)
    plt.figure(figsize=(16, 8))
    for i, key in enumerate(["loss", "acc", "rmse", "iou"], 1):
        plt.subplot(2, 2, i)
        plt.plot(epochs, history[f"train_{key}"], label="Train")
        plt.plot(epochs, history[f"val_{key}"], label="Val")
        plt.title(key.upper()); plt.xlabel("Época"); plt.legend(); plt.grid(True, ls='--', alpha=.4)
    plt.suptitle("Desempeño SwinUNet - Regresión RGB")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    train()
