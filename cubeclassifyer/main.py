import torch
import os
from cube_classifier import LightweightCubeClassifier, train_model, CubeDataset, get_transforms, convert_model_for_rpi
from torch.utils.data import DataLoader

def prepare_data():
    """Prepare data directories structure"""
    data_dirs = [
        "cube_dataset/train/good",
        "cube_dataset/train/defective",
        "cube_dataset/val/good",
        "cube_dataset/val/defective"
    ]
    
    for dir_path in data_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("Data directory structure created:")
    print("- cube_dataset/")
    print("  - train/")
    print("    - good/")
    print("    - defective/")
    print("  - val/")
    print("    - good/")
    print("    - defective/")
    print("\nPlease place your 320x240 grayscale cube images in the appropriate folders.")

def train_cube_classifier():
    """Train the cube classifier model"""
    # Check if data directories exist
    if not os.path.exists("cube_dataset"):
        print("Data directories not found. Creating structure...")
        prepare_data()
    
    # Create datasets
    train_dataset = CubeDataset(
        root_dir="cube_dataset/train",
        transform=get_transforms(train=True)
    )
    
    val_dataset = CubeDataset(
        root_dir="cube_dataset/val",
        transform=get_transforms(train=False)
    )
    
    # Check if we have data
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("No data found in dataset directories.")
        print("Please place your 320x240 grayscale cube images in the appropriate folders:")
        print("- cube_dataset/train/good/")
        print("- cube_dataset/train/defective/")
        print("- cube_dataset/val/good/")
        print("- cube_dataset/val/defective/")
        return
    
    # Create data loaders
    # Use num_workers=0 for compatibility across different systems
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    # Create model
    model = LightweightCubeClassifier(num_classes=2)
    
    print(f"Training model with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    print("Starting training...")
    
    # Train model
    trained_model = train_model(model, train_loader, val_loader, num_epochs=20)

    print("Training completed! Model saved as 'best_cube_classifier.pth'")

    # Convert model for Raspberry Pi deployment
    if os.path.exists('best_cube_classifier.pth'):
        print("\nConverting model for Raspberry Pi deployment...")
        convert_model_for_rpi('best_cube_classifier.pth')
        print("Model converted! Transfer 'cube_classifier_rpi.pt' to your Raspberry Pi.")

def main():
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    
    if torch.cuda.is_available():
        print("CUDA version:", torch.version.cuda)
        print("GPU count:", torch.cuda.device_count())
        print("Current GPU:", torch.cuda.get_device_name())
    
    print("\nOptions:")
    print("1. Prepare data directory structure")
    print("2. Train cube classifier model")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        prepare_data()
    elif choice == "2":
        train_cube_classifier()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()