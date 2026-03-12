import os
import shutil
import random

# Paths
base_path = 'archive-full dataset/CXR8'
images_path = os.path.join(base_path, 'images')
dataset_path = os.path.join(base_path, 'dataset')
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')

# Create directories
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Function to find image path
def find_image_path(image_name):
    for subdir in os.listdir(images_path):
        subdir_path = os.path.join(images_path, subdir)
        if os.path.isdir(subdir_path):
            image_path = os.path.join(subdir_path, image_name)
            if os.path.exists(image_path):
                return image_path
    return None

# Read test_list.txt
with open(os.path.join(base_path, 'test_list.txt'), 'r') as f:
    test_images = [line.strip() for line in f]

# Move test images
test_count = 0
for img in test_images:
    src = find_image_path(img)
    if src:
        shutil.copy(src, os.path.join(test_path, img))
        test_count += 1
print(f"Copied {test_count} test images.")

# Read train_val_list.txt
with open(os.path.join(base_path, 'train_val_list.txt'), 'r') as f:
    train_val_images = [line.strip() for line in f]

# Split train_val into train and val, 80-20
random.seed(42)
random.shuffle(train_val_images)
split_idx = int(0.8 * len(train_val_images))
train_images = train_val_images[:split_idx]
val_images = train_val_images[split_idx:]
    
# Move train images
for img in train_images:
    src = find_image_path(img)
    if src:
        shutil.copy(src, os.path.join(train_path, img))

# Move val images
for img in val_images:
    src = find_image_path(img)
    if src:
        shutil.copy(src, os.path.join(val_path, img))

print("Dataset organization complete.")
