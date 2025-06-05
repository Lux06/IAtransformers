# config.py
import os

# Directorios del dataset (ajusta las rutas según tu proyecto)
"""DATA_DIR = os.path.join(os.getcwd(), "dataset")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "images", "train")
TRAIN_MASKS_DIR = os.path.join(DATA_DIR, "annotations", "train")
VAL_IMAGES_DIR = os.path.join(DATA_DIR, "images", "val")
VAL_MASKS_DIR = os.path.join(DATA_DIR, "annotations", "val")
"""

DATA_DIR = os.path.join(os.getcwd(), "data")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train", "imagenes")
TRAIN_MASKS_DIR = os.path.join(DATA_DIR, "train", "mascaras")
VAL_IMAGES_DIR = os.path.join(DATA_DIR, "val", "imagenes")
VAL_MASKS_DIR = os.path.join(DATA_DIR, "val", "mascaras")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test", "imagenes")
TEST_MASKS_DIR = os.path.join(DATA_DIR, "test", "mascaras")

# Parámetros de entrenamiento
"""BATCH_SIZE = 4
NUM_WORKERS = 2
NUM_CLASSES = 2          # nopal y fondo 256 o salida
IMAGE_SIZE = (224, 224)  # ancho x alto
LEARNING_RATE = 1e-4
NUM_EPOCHS = 120"""

BATCH_SIZE    = 32    # Aumenta si tienes más VRAM
NUM_WORKERS   = 4
NUM_CLASSES   = 3
IMAGE_SIZE    = (100, 100)
LEARNING_RATE = 1e-4
NUM_EPOCHS    = 10

