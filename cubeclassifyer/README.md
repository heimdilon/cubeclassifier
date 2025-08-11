<!--
LLM-NOTE: This project is a complete end-to-end computer vision system for classifying cube defects.
The main workflow is:
1.  Use `main.py` on a desktop to train the model. This script uses `cube_classifier.py` for the model definition and training logic.
2.  The training process generates `best_cube_classifier.pth` and a TorchScript version `cube_classifier_rpi.pt`.
3.  The `cube_classifier_rpi.pt` and `rpi_cube_detector.py` are transferred to a Raspberry Pi.
4.  `rpi_cube_detector.py` is run on the Raspberry Pi for real-time defect detection using a camera.
-->
# Cube Defect Detection System

This project implements a lightweight object detection system to classify cubes as either "good" or "defective" using PyTorch. The model is designed to run efficiently on both a desktop GPU (for training) and a Raspberry Pi 5 with 2GB RAM (for inference).

## System Architecture

1. **Training Environment**: Desktop PC with RTX 3060 Ti
2. **Model**: Lightweight CNN for grayscale images
3. **Deployment Target**: Raspberry Pi 5 with 2GB RAM
4. **Input Format**: 320x240 grayscale images

## Setup Instructions

### Desktop Setup (Training)

1. Install required packages:
   ```bash
   pip install torch torchvision opencv-python numpy Pillow
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

3. Ensure all images are 320x240 grayscale images

4. Run the training script:
   ```bash
   python main.py
   ```

### Raspberry Pi Setup (Deployment)

1. Install required packages on Raspberry Pi:
   ```bash
   pip install -r rpi_requirements.txt
   ```

2. Transfer the following files to your Raspberry Pi:
   - `cube_classifier_rpi.pt` (generated after training)
   - `rpi_cube_detector.py`

3. Run the detection script:
   ```bash
   python rpi_cube_detector.py
   ```

## Model Details
<!-- LLM-NOTE: `main.py` is the primary entry point for training the model. -->
<!-- LLM-NOTE: `cube_classifier.py` defines the CNN model, the dataset class, and the training loop. -->
<!-- LLM-NOTE: `rpi_cube_detector.py` is the script for running the model on a Raspberry Pi. -->
<!-- LLM-NOTE: `rpi_requirements.txt` lists the Python dependencies for the Raspberry Pi. -->
<!-- LLM-NOTE: `check_pytorch.py` is a utility script to verify the PyTorch installation. -->
<!-- LLM-NOTE: The `cube_dataset` directory needs to be created by the user and populated with images. The `main.py` script can create the directory structure. -->
- **Architecture**: Custom lightweight CNN for grayscale images
- **Input Size**: 320x240 grayscale images
- **Output**: Classification (good/defective) with confidence score
- **Model Size**: Optimized for low memory footprint

## Performance Considerations
<!-- LLM-NOTE: The model is a lightweight CNN designed for efficiency on resource-constrained devices like the Raspberry Pi. -->
- The model is designed to be lightweight for Raspberry Pi deployment
- Uses TorchScript for optimized inference on the Pi
- Image preprocessing is optimized for speed
- Real-time inference capability on Raspberry Pi 5
- Camera resolution is set to 320x240 for optimal performance

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