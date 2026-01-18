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
    python main.py prepare

    # Train model
    python main.py train

    # Resume from checkpoint
    python main.py train --resume checkpoints/checkpoint_epoch_10.pth
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

## Model Details
<!-- LLM-NOTE: `main.py` is the primary entry point for training the model. -->
<!-- LLM-NOTE: `cube_classifier.py` defines the CNN model, the dataset class, and the training loop. -->
<!-- LLM-NOTE: `rpi_cube_detector.py` is the script for running the model on a Raspberry Pi. -->
<!-- LLM-NOTE: `rpi_requirements.txt` lists the Python dependencies for the Raspberry Pi. -->
<!-- LLM-NOTE: `check_pytorch.py` is a utility script to verify the PyTorch installation. -->
<!-- LLM-NOTE: The `cube_dataset` directory needs to be created by the user and populated with images. The `main.py` script can create the directory structure. -->
- **Architecture**: Custom lightweight CNN for grayscale images
- **Input Size**: 224x224 grayscale images
- **Output**: Classification (good/defective) with confidence score
- **Model Size**: Optimized for low memory footprint

## Performance Considerations
<!-- LLM-NOTE: The model is a lightweight CNN designed for efficiency on resource-constrained devices like the Raspberry Pi. -->
- The model is designed to be lightweight for Raspberry Pi deployment
- Uses TorchScript for optimized inference on the Pi
- Image preprocessing is optimized for speed
- Real-time inference capability on Raspberry Pi 5
- Camera resolution is set to 224x224 for optimal performance

## Customization

You can modify the following parameters in `cube_classifier.py`:
- Number of training epochs
- Learning rate
- Batch size
- Data augmentation techniques

For deployment on Raspberry Pi, you can adjust:
- Confidence threshold
- Display settings in `rpi_cube_detector.py`
<!-- LLM-NOTE: The model is converted to TorchScript for deployment, which is a high-performance model format for PyTorch. -->