"""
Data Organizer for Binary Classification Models
==============================================

Organizes training data for the 3 binary classifiers.
Run this before training to ensure proper data structure.

Expected final structure:
data/
├── model1_garbage_vs_xray/
│   ├── xray/          # X-ray images (chest, hand, etc.)
│   └── garbage/       # Non-X-ray images (photos, CT, MRI, etc.)
├── model2_chest_vs_other/
│   ├── chest/         # Chest X-ray images only
│   └── other/         # Other body part X-rays (hand, skull, etc.)
└── model3_normal_vs_abnormal/
    ├── normal/        # Normal chest X-rays
    └── abnormal/      # Abnormal chest X-rays
"""

import os
import shutil
import argparse
from pathlib import Path

def create_data_structure(base_dir='data'):
    """Create the required directory structure"""

    dirs = [
        f'{base_dir}/model1_garbage_vs_xray/xray',
        f'{base_dir}/model1_garbage_vs_xray/garbage',
        f'{base_dir}/model2_chest_vs_other/chest',
        f'{base_dir}/model2_chest_vs_other/other',
        f'{base_dir}/model3_normal_vs_abnormal/normal',
        f'{base_dir}/model3_normal_vs_abnormal/abnormal'
    ]

    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created: {dir_path}")

def organize_sample_data(source_dir, target_dir='data'):
    """
    Organize sample data from a source directory
    This is a helper function - you'll need to adapt it to your data sources
    """

    print("This is a template function. Adapt it to your data sources.")
    print("You'll need to:")
    print("1. Download datasets from the sources mentioned in CLEAR_CUT_ANSWER.txt")
    print("2. Extract them to appropriate directories")
    print("3. Run this script to organize them")

    # Example organization (adapt to your needs):
    # - Copy chest X-rays to model1_garbage_vs_xray/xray/
    # - Copy hand X-rays to model1_garbage_vs_xray/xray/ and model2_chest_vs_other/other/
    # - Copy photos to model1_garbage_vs_xray/garbage/
    # - Copy normal chest X-rays to model3_normal_vs_abnormal/normal/
    # - Copy abnormal chest X-rays to model3_normal_vs_abnormal/abnormal/

def count_images(data_dir='data'):
    """Count images in each category"""

    categories = [
        'model1_garbage_vs_xray/xray',
        'model1_garbage_vs_xray/garbage',
        'model2_chest_vs_other/chest',
        'model2_chest_vs_other/other',
        'model3_normal_vs_abnormal/normal',
        'model3_normal_vs_abnormal/abnormal'
    ]

    print("\n📊 Data Summary:")
    print("=" * 50)

    total_images = 0
    for category in categories:
        path = os.path.join(data_dir, category)
        if os.path.exists(path):
            count = len([f for f in os.listdir(path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
            print("25")
            total_images += count
        else:
            print("25")

    print("=" * 50)
    print(f"Total images: {total_images}")

    return total_images

def validate_data_structure(data_dir='data'):
    """Validate that data structure is correct"""

    required_dirs = [
        'model1_garbage_vs_xray/xray',
        'model1_garbage_vs_xray/garbage',
        'model2_chest_vs_other/chest',
        'model2_chest_vs_other/other',
        'model3_normal_vs_abnormal/normal',
        'model3_normal_vs_abnormal/abnormal'
    ]

    print("\n🔍 Data Structure Validation:")
    print("=" * 50)

    all_good = True
    for dir_path in required_dirs:
        full_path = os.path.join(data_dir, dir_path)
        if os.path.exists(full_path):
            count = len([f for f in os.listdir(full_path)
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))])
            status = "✅" if count > 0 else "⚠️ (empty)"
            print("30")
        else:
            print("30")
            all_good = False

    print("=" * 50)
    if all_good:
        print("✅ Data structure is valid!")
    else:
        print("❌ Some directories are missing. Run --create-dirs first.")

    return all_good

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Organizer for Binary Classification Models')
    parser.add_argument('--create-dirs', action='store_true', help='Create directory structure')
    parser.add_argument('--count', action='store_true', help='Count images in each category')
    parser.add_argument('--validate', action='store_true', help='Validate data structure')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory path')

    args = parser.parse_args()

    if args.create_dirs:
        create_data_structure(args.data_dir)
        print("\n✅ Directory structure created!")
        print("\nNext steps:")
        print("1. Download datasets from sources in CLEAR_CUT_ANSWER.txt")
        print("2. Extract images to appropriate directories")
        print("3. Run: python data_organizer.py --count --validate")

    elif args.count:
        count_images(args.data_dir)

    elif args.validate:
        validate_data_structure(args.data_dir)

    else:
        print("Binary Classification Data Organizer")
        print("===================================")
        print()
        print("Usage:")
        print("  python data_organizer.py --create-dirs    # Create directory structure")
        print("  python data_organizer.py --count          # Count images")
        print("  python data_organizer.py --validate       # Validate structure")
        print()
        print("Expected data structure:")
        print("data/")
        print("├── model1_garbage_vs_xray/")
        print("│   ├── xray/      # X-ray images")
        print("│   └── garbage/   # Non-X-ray images")
        print("├── model2_chest_vs_other/")
        print("│   ├── chest/     # Chest X-rays")
        print("│   └── other/     # Other X-rays")
        print("└── model3_normal_vs_abnormal/")
        print("    ├── normal/    # Normal chest X-rays")
        print("    └── abnormal/  # Abnormal chest X-rays")