import os
import shutil
import pandas as pd
from pathlib import Path

# ================= CONFIGURATION =================
# Path to the dataset folder containing images and CSV files
DATASET_DIR = r"c:\Users\91939\Desktop\chexnet-proto\chexnet\dataset"

# CSV filenames and their corresponding target folders
# We look for these CSVs in the DATASET_DIR
MAPPINGS = [
    {"csv": "train_1.csv", "folder": "train_sample", "alt": ["train_csv", "train.csv"]},
    {"csv": "val_1.csv",   "folder": "val_sample",   "alt": ["val_csv", "val.csv"]},
    {"csv": "test_1.csv",  "folder": "test_sample",  "alt": ["test_csv", "test.csv"]}
]
# =================================================

def organize_images():
    base_path = Path(DATASET_DIR)
    
    if not base_path.exists():
        print(f"❌ Error: Dataset directory not found: {base_path}")
        return

    print(f"📂 Working in: {base_path}")
    print("🔄 Starting organization process...")

    for item in MAPPINGS:
        target_csv = item["csv"]
        target_folder = item["folder"]
        alternatives = item["alt"]

        # 1. Locate the CSV file
        csv_path = base_path / target_csv
        
        # If primary name doesn't exist, check alternatives
        if not csv_path.exists():
            for alt in alternatives:
                if (base_path / alt).exists():
                    csv_path = base_path / alt
                    print(f"   ℹ️  Using alternative file: {alt}")
                    break
        
        if not csv_path.exists():
            print(f"   ⚠️  Skipping {target_folder}: CSV file not found (looked for {target_csv})")
            continue

        # 2. Create target directory
        dest_dir = base_path / target_folder
        dest_dir.mkdir(exist_ok=True)

        print(f"\n📋 Processing {csv_path.name} → {target_folder}/")

        # 3. Read CSV and extract filenames
        try:
            # Try reading with pandas
            # First try with header inference
            df = pd.read_csv(csv_path)
            
            # Identify the column containing filenames (looks for .png, .jpg, etc)
            filename_col = None
            
            # Check columns
            for col in df.columns:
                if df[col].astype(str).str.contains(r'\.(png|jpg|jpeg|tiff)$', case=False, na=False).any():
                    filename_col = col
                    break
            
            # If not found in columns, maybe it's a headerless CSV?
            if filename_col is None:
                df = pd.read_csv(csv_path, header=None)
                for col in df.columns:
                    if df[col].astype(str).str.contains(r'\.(png|jpg|jpeg|tiff)$', case=False, na=False).any():
                        filename_col = col
                        break
            
            if filename_col is None:
                # Fallback: assume first column
                print(f"   ⚠️  Could not auto-detect image column. Assuming first column.")
                filename_col = df.columns[0]

            image_files = df[filename_col].dropna().astype(str).tolist()
            print(f"   Found {len(image_files)} entries in CSV.")

        except Exception as e:
            print(f"   ❌ Error reading CSV: {e}")
            continue

        # 4. Move images
        moved = 0
        missing = 0
        already_there = 0

        for img_file in image_files:
            img_file = img_file.strip()
            # Handle potential paths in CSV (e.g. images/001.png -> 001.png)
            img_name = os.path.basename(img_file)
            
            src = base_path / img_name
            dst = dest_dir / img_name

            if src.exists():
                try:
                    shutil.move(str(src), str(dst))
                    moved += 1
                except Exception as e:
                    print(f"      Error moving {img_name}: {e}")
            elif dst.exists():
                already_there += 1
            else:
                missing += 1

        print(f"   ✅ Moved: {moved}")
        if already_there > 0:
            print(f"   ℹ️  Already in target: {already_there}")
        if missing > 0:
            print(f"   ⚠️  Missing/Not found: {missing}")

    print("\n✨ Organization complete!")

if __name__ == "__main__":
    organize_images()