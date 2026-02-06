# Cube Defect Detection System

This project implements a lightweight object detection system to classify cubes as either "good" or "defective" using PyTorch. The model is designed to run efficiently on both a desktop GPU (for training) and a Raspberry Pi 5 with 2GB RAM (for inference).

## System Architecture

1. **Training Environment**: Desktop PC with RTX 3060 Ti
2. **Model**: Lightweight CNN for grayscale images
3. **Deployment Target**: Raspberry Pi 5 with 2GB RAM
4. **Input Format**: 224x224 grayscale images

## Features

### Training
- **Automated Checkpointing**: Save progress every N epochs
- **Resume Capability**: Continue training from any checkpoint
- **Early Stopping**: Prevent overfitting with configurable patience
- **Class Imbalance Handling**: Weighted loss function
- **Gradient Clipping**: Prevent exploding gradients
- **Model Versioning**: Timestamped model saves
- **Comprehensive Logging**: Timestamped logs to file
- **CLI Interface**: Non-interactive command-line interface

### Data Collection (Raspberry Pi)
- **Live Camera Preview**: Real-time view with 224x224 capture region
- **Simple Controls**: Press `G` for good, `D` for defective, `Q` to quit
- **Auto-Naming**: Automatic incrementing filenames
- **Live Counter**: Shows collected good/defective counts
- **Auto-Processing**: Automatic grayscale conversion and resize

### Inference (Raspberry Pi)
- **Confidence Thresholding**: Reject low-confidence predictions
- **Color-Coded Display**: Green (good), Red (defective), Yellow (uncertain)
- **FPS Display**: Real-time performance monitoring
- **Camera Auto-Reconnect**: Automatic recovery from camera failures
- **Frame Saving**: Option to save detection frames
- **Graceful Shutdown**: Clean resource cleanup on interruption

## Setup Instructions

### Desktop Setup (Training)

1. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

2. Organize your data in the following structure:
   ```
   cube_dataset/
   ├── train/
   │   ├── good/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   └── defective/
   │       ├── image1.jpg
   │       ├── image2.jpg
   │       └── ...
   └── val/
       ├── good/
       │   ├── image1.jpg
       │   ├── image2.jpg
       │   └── ...
       └── defective/
           ├── image1.jpg
           ├── image2.jpg
           └── ...
   ```

 3. Ensure all images are 224x224 grayscale images (images will be automatically resized)

4. Run the training script:
    ```bash
    # Prepare data directories
    python -m cubeclassifyer.main prepare

    # Train model
    python -m cubeclassifyer.main train

    # Resume from checkpoint
    python -m cubeclassifyer.main train --resume checkpoints/checkpoint_epoch_10.pth

    # Export deployment models (TorchScript + ONNX + quantized TorchScript)
    python -m cubeclassifyer.main export --model-path best_cube_classifier.pth
    ```

5. Monitor training logs:
    ```bash
    tail -f logs/training.log
    ```

### Configuration

Edit `config.py` to customize training parameters:
- `NUM_EPOCHS`: Training duration (default: 20)
- `LEARNING_RATE`: Optimization rate (default: 0.001)
- `BATCH_SIZE`: Batch size (default: 16)
- `PATIENCE`: Early stopping patience (default: 5)
- `SAVE_CHECKPOINT_EVERY`: Save checkpoint interval (default: 5)
- `LOG_LEVEL`: Logging verbosity (default: "INFO")

### Raspberry Pi Setup (Data Collection & Deployment)

1. Install required packages on Raspberry Pi:
   ```bash
   pip install -r rpi_requirements.txt
   ```

2. **Collect Training Data** using the data collector:
    ```bash
    # Transfer rpi_data_collector.py to Raspberry Pi, then run:
    python rpi_data_collector.py
    
    # Controls:
    #   [G] - Save as GOOD cube
    #   [D] - Save as DEFECTIVE cube  
    #   [Q] - Quit
    
    # Custom output directory
    python rpi_data_collector.py --output-dir my_cubes
    ```
    
    Position the cube inside the yellow box and press G or D to capture.
    Images are automatically saved as 224x224 grayscale.

3. Transfer collected images to your training PC and organize into train/val folders.

4. After training, transfer the following files to your Raspberry Pi:
    - `cube_classifier_rpi.pt` (generated after training)
    - `rpi_cube_detector.py`

5. Run the detection script:
    ```bash
    # Basic detection
    python rpi_cube_detector.py

    # Save frames with detections
    python rpi_cube_detector.py --save-frames

    # Custom confidence threshold
    python rpi_cube_detector.py --threshold 0.8

    # Different camera index
    python rpi_cube_detector.py --camera 1

    # Combine options
    python rpi_cube_detector.py --save-frames --save-dir my_frames --threshold 0.9
    ```

   4. Monitor performance:
     - Press 'q' to quit
     - View FPS and inference time in real-time
     - Frames saved to `saved_frames/` if enabled

### Exported Deployment Artifacts

Training or explicit export creates these files:
- `cube_classifier_rpi.pt` (TorchScript)
- `cube_classifier_rpi_int8.pt` (quantized TorchScript)
- `cube_classifier_rpi.onnx` (ONNX)

You can skip optional exports with:
```bash
python -m cubeclassifyer.main train --no-onnx --no-quantized
```

For inference backend selection on Raspberry Pi:
```bash
# TorchScript backend (default)
python rpi_cube_detector.py --backend torchscript --model cube_classifier_rpi.pt

# ONNX Runtime backend
python rpi_cube_detector.py --backend onnx --model cube_classifier_rpi.onnx
```

## Model Details
- **Architecture**: Custom lightweight CNN for grayscale images
- **Input Size**: 224x224 grayscale images
- **Output**: Classification (good/defective) with confidence score
- **Model Size**: Optimized for low memory footprint

Core modules:
- `main.py`: CLI entrypoint (`prepare`, `train`, `export`)
- `dataset.py`: dataset loading and validation
- `modeling.py`: model definitions
- `training.py`: transforms and training loop
- `exporting.py`: TorchScript/ONNX/quantized export helpers
- `rpi_cube_detector.py`: Raspberry Pi inference CLI

## Performance Considerations
- The model is designed to be lightweight for Raspberry Pi deployment
- Uses TorchScript for optimized inference on the Pi
- Image preprocessing is optimized for speed
- Real-time inference capability on Raspberry Pi 5
- Camera capture defaults to 640x480 and is resized to 224x224 during preprocessing

## Customization

You can modify the following parameters in `config.py`:
- Number of training epochs
- Learning rate
- Batch size
- Data augmentation techniques

For deployment on Raspberry Pi, you can adjust:
- Confidence threshold
- Display settings in `rpi_cube_detector.py`
