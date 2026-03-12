import os
import time
from chexnet.ChexnetTrainer import ChexnetTrainer

# ================= CONFIGURATION =================
# Paths
BASE_DIR = r"c:\Users\91939\Desktop\chexnet-proto\chexnet"
IMAGES_DIR = os.path.join(BASE_DIR, "dataset", "train_sample")
TRAIN_LIST = os.path.join(BASE_DIR, "dataset", "train_1.csv")
VAL_LIST = os.path.join(BASE_DIR, "dataset", "val_1.csv")

# Training Parameters
ARCHITECTURE = 'DENSE-NET-121'
USE_PRETRAINED = True
NUM_CLASSES = 14
BATCH_SIZE = 16          # Lower this if you run out of GPU memory
BATCH_SIZE = 8           # Reduced to 8 for RTX 3050 (4GB VRAM) stability
MAX_EPOCHS = 10          # How long to train
TRANS_RESIZE = 256
TRANS_CROP = 224
TIMESTAMP = time.strftime("%d%m%Y-%H%M%S")
# =================================================

def run():
    print("Starting training process...")
    print(f"Images: {IMAGES_DIR}")
    print(f"Train List: {TRAIN_LIST}")
    
    # Check if images exist
    if not os.path.exists(IMAGES_DIR) or not os.listdir(IMAGES_DIR):
        print("❌ Error: Image directory is empty or missing.")
        print("   Please run 'organize_images.py' first.")
        return

    # Start Training
    # This will print loss and validation metrics to the console
    ChexnetTrainer.train(
        IMAGES_DIR, TRAIN_LIST, VAL_LIST, ARCHITECTURE, 
        USE_PRETRAINED, NUM_CLASSES, BATCH_SIZE, MAX_EPOCHS, 
        TRANS_RESIZE, TRANS_CROP, TIMESTAMP, None
    )

if __name__ == "__main__":
    run()