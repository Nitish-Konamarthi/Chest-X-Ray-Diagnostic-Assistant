"""
Train All Binary Classification Models
=====================================

Trains all 3 binary classifiers sequentially.
Make sure you have organized your data first using data_organizer.py

Usage:
    python train_all.py --data data --epochs 20
"""

import os
import argparse
import subprocess
import sys

def run_training_script(script_name, data_dir, model_path, epochs=20, batch_size=None, sample_size=None):
    """Run a training script with given parameters"""

    cmd = [sys.executable, script_name, '--train', '--data', data_dir, '--model', model_path, '--epochs', str(epochs)]

    if batch_size:
        cmd.extend(['--batch_size', str(batch_size)])

    if sample_size:
        cmd.extend(['--sample-size', str(sample_size)])

    print(f"\n🚀 Training {script_name}...")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 50)

    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)

    if result.returncode != 0:
        print(f"❌ Training failed for {script_name}")
        return False
    else:
        print(f"✅ Training completed for {script_name}")
        return True

def main():
    parser = argparse.ArgumentParser(description='Train All Binary Classification Models')
    parser.add_argument('--data', type=str, default='data', help='Data directory')
    parser.add_argument('--models', type=str, default='models', help='Models directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for Model 1 & 2')
    parser.add_argument('--epochs-model3', type=int, default=10, help='Number of epochs for Model 3 (transfer learning, optimized for RTX 3050)')
    parser.add_argument('--batch-size-model1', type=int, default=64, help='Batch size for Model 1')
    parser.add_argument('--batch-size-model2', type=int, default=64, help='Batch size for Model 2')
    parser.add_argument('--batch-size-model3', type=int, default=16, help='Batch size for Model 3 (optimized for RTX 3050 4GB)')
    parser.add_argument('--sample-size', type=int, default=None, help='Sample size per class (None = use all data)')
    parser.add_argument('--skip-validation', action='store_true', help='Skip data validation')

    args = parser.parse_args()

    print("🔬 Binary Classification Model Training")
    print("=" * 50)
    print(f"Data directory: {args.data}")
    print(f"Models directory: {args.models}")
    print(f"Epochs per model: {args.epochs}")
    print()

    # Validate data structure
    if not args.skip_validation:
        print("🔍 Validating data structure...")
        result = subprocess.run([sys.executable, 'data_organizer.py', '--validate', '--data-dir', args.data],
                              capture_output=True, text=True)

        if result.returncode != 0:
            print("❌ Data validation failed!")
            print(result.stdout)
            print("Please run: python data_organizer.py --create-dirs")
            print("Then organize your data according to the structure shown.")
            return

        print("✅ Data structure validated!")

    # Create models directory
    os.makedirs(args.models, exist_ok=True)

    # Training configurations for each model
    trainings = [
        {
            'script': 'binary_model1.py',
            'data_dir': os.path.join(args.data, 'model1_garbage_vs_xray'),
            'model_path': os.path.join(args.models, 'garbage_vs_xray.pth'),
            'batch_size': args.batch_size_model1,
            'sample_size': args.sample_size,
            'epochs': args.epochs,
            'description': 'Model 1: Garbage vs X-ray (MobileNetV2)'
        },
        {
            'script': 'binary_model2.py',
            'data_dir': os.path.join(args.data, 'model2_chest_vs_other'),
            'model_path': os.path.join(args.models, 'chest_vs_other.pth'),
            'batch_size': args.batch_size_model2,
            'sample_size': args.sample_size,
            'epochs': args.epochs,
            'description': 'Model 2: Chest vs Other X-rays (ResNet18)'
        },
        {
            'script': 'binary_model3.py',
            'data_dir': os.path.join(args.data, 'model3_normal_vs_abnormal'),
            'model_path': os.path.join(args.models, 'normal_vs_abnormal.pth'),
            'batch_size': args.batch_size_model3,
            'sample_size': args.sample_size,
            'epochs': args.epochs_model3,
            'description': 'Model 3: Normal vs Abnormal (ResNet50 - Transfer Learning with Fine-tuning ⭐ IMPROVED!)'
        }
    ]

    # Train each model
    success_count = 0
    for training in trainings:
        print(f"\n🎯 {training['description']}")
        print("=" * 50)

        success = run_training_script(
            training['script'],
            training['data_dir'],
            training['model_path'],
            training['epochs'],
            training['batch_size'],
            training['sample_size']
        )

        if success:
            success_count += 1
        else:
            print(f"⚠️  Continuing with next model...")

    print(f"\n🎉 Training Summary")
    print("=" * 50)
    print(f"Models trained successfully: {success_count}/3")

    if success_count == 3:
        print("✅ All models trained successfully!")
        print("\n🚀 Next steps:")
        print("1. Test the models: python test_all.py")
        print("2. Integrate with Streamlit: The binary_pipeline.py is ready!")
        print("3. Add 10 lines to streamlit_app.py as shown in CLEAR_CUT_ANSWER.txt")
    else:
        print("❌ Some models failed to train. Check the output above.")

    print(f"\n📁 Model files saved to: {args.models}/")
    print("- garbage_vs_xray.pth")
    print("- chest_vs_other.pth")
    print("- normal_vs_abnormal.pth")

if __name__ == '__main__':
    main()