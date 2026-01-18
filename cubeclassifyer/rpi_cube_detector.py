import torch
import cv2
import numpy as np
import time
import os
import sys
import argparse
from datetime import datetime
from utils import logger


# Check if model file exists
MODEL_PATH = "cube_classifier_rpi.pt"
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file '{MODEL_PATH}' not found!")
    print("Please ensure you have transferred the model file to this directory.")
    sys.exit(1)

# Load the TorchScript model
try:
    model = torch.jit.load(MODEL_PATH)
    model.eval()
    print(f"Model loaded successfully from '{MODEL_PATH}'")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Class names
class_names = ["good", "defective", "uncertain"]

# Confidence threshold for accepting predictions
CONFIDENCE_THRESHOLD = 0.7

# Camera configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# FPS measurement
FPS_WINDOW_SIZE = 30  # Number of frames to average over

# Frame saving
SAVE_FRAMES = False
SAVE_DIR = "saved_frames"


def init_camera(camera_index=0):
    """Initialize camera with error handling and retry logic"""
    for attempt in range(MAX_RETRIES):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            # Set camera resolution to 224x224
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
            return cap

        if attempt < MAX_RETRIES - 1:
            logger.warning(
                f"Camera initialization failed (attempt {attempt + 1}/{MAX_RETRIES}), retrying..."
            )
            time.sleep(RETRY_DELAY)

    return None


def preprocess_image(image):
    """Optimized preprocessing using OpenCV directly (faster than PIL)

    Args:
        image: Raw BGR image from camera

    Returns:
        Preprocessed tensor ready for model
    """
    # Convert to grayscale (OpenCV is faster than PIL)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize using OpenCV (much faster than PIL)
    resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Normalize to [-1, 1] (equivalent to Normalize(mean=[0.5], std=[0.5]))
    # PIL: (x / 255 - 0.5) / 0.5 = (x - 128) / 128
    normalized = (resized.astype(np.float32) - 128.0) / 128.0

    # Convert to tensor directly from NumPy (skip PIL step)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

    return tensor


def predict_cube(image):
    """Predict if cube is good or defective"""
    # Preprocess image
    input_tensor = preprocess_image(image)

    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class].item()

    # Check if confidence is below threshold
    if confidence < CONFIDENCE_THRESHOLD:
        return class_names[2], confidence  # Return "uncertain"

    return class_names[predicted_class], confidence


def main():
    parser = argparse.ArgumentParser(
        description="Cube Detector - Real-time defect detection"
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    parser.add_argument(
        "--save-frames", action="store_true", help="Save frames with detections"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="saved_frames",
        help="Directory to save frames (default: saved_frames)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold (default: 0.7)",
    )

    args = parser.parse_args()

    # Update global config from arguments
    global SAVE_FRAMES, SAVE_DIR, CONFIDENCE_THRESHOLD
    SAVE_FRAMES = args.save_frames
    SAVE_DIR = args.save_dir
    CONFIDENCE_THRESHOLD = args.threshold

    # Create save directory if needed
    if SAVE_FRAMES:
        os.makedirs(SAVE_DIR, exist_ok=True)
        logger.info(f"Saving frames to: {SAVE_DIR}")

    # Initialize camera
    cap = init_camera(camera_index=args.camera)

    if cap is None:
        logger.error("Could not initialize camera after multiple attempts")
        return

    logger.info(f"Starting cube detection. Press 'q' to quit.")
    logger.info(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    # FPS tracking variables
    frame_times = []

    try:
        while True:
            # Capture frame
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read frame. Attempting to reconnect...")
                cap.release()
                cap = init_camera(camera_index=0)

                if cap is None:
                    logger.error("Failed to reconnect. Exiting.")
                    break

                continue  # Skip this iteration and try again with new cap

            # Make prediction
            start_time = time.time()
            prediction, confidence = predict_cube(frame)
            inference_time = time.time() - start_time

            # Calculate FPS
            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            if len(frame_times) > FPS_WINDOW_SIZE:
                frame_times.pop(0)
            fps = 1.0 / (sum(frame_times) / len(frame_times))

            # Determine color based on prediction
            if prediction == "good":
                color = (0, 255, 0)  # Green
            elif prediction == "defective":
                color = (0, 0, 255)  # Red
            else:  # uncertain
                color = (0, 255, 255)  # Yellow

            # Display result on frame
            label = f"{prediction}: {confidence:.2f}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(
                frame,
                f"Inf: {inference_time * 1000:.1f}ms FPS: {fps:.1f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            # Show frame
            cv2.imshow("Cube Detection", frame)

            # Save frame if enabled
            if SAVE_FRAMES:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                safe_prediction = prediction.replace(" ", "_")
                filename = os.path.join(SAVE_DIR, f"{safe_prediction}_{timestamp}.jpg")
                cv2.imwrite(filename, frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Cleaning up...")
    finally:
        # Release resources (ensure cleanup always happens)
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Cube detection stopped")


if __name__ == "__main__":
    main()
