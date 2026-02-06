import argparse
import hashlib
import os
import re
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch

if __package__:
    from .utils import logger
else:
    from utils import logger


CLASS_NAMES = ["good", "defective"]
UNCERTAIN_LABEL = "uncertain"

DEFAULT_MODEL_PATH = "cube_classifier_rpi.pt"
DEFAULT_BACKEND = "torchscript"

MAX_RETRIES = 3
RETRY_DELAY = 1
FPS_WINDOW_SIZE = 30


def compute_sha256(file_path):
    hasher = hashlib.sha256()
    with open(file_path, "rb") as handle:
        while True:
            chunk = handle.read(8192)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_model_checksum(model_path, expected_sha256=None):
    if expected_sha256 is None:
        return

    if re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256) is None:
        raise ValueError("Invalid SHA256 format. Expected 64 hexadecimal characters.")

    actual_sha256 = compute_sha256(model_path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"Model checksum mismatch. Expected {expected_sha256}, got {actual_sha256}."
        )


def init_camera(camera_index=0, width=640, height=480):
    for attempt in range(MAX_RETRIES):
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap

        if attempt < MAX_RETRIES - 1:
            logger.warning(
                "Camera initialization failed (attempt %s/%s), retrying...",
                attempt + 1,
                MAX_RETRIES,
            )
            time.sleep(RETRY_DELAY)

    return None


def load_torchscript_model(model_path=DEFAULT_MODEL_PATH, model_sha256=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' not found. Please copy it to this directory."
        )

    verify_model_checksum(model_path, model_sha256)
    model = torch.jit.load(model_path)
    model.eval()
    return model


def load_onnx_model(model_path, model_sha256=None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ONNX model file '{model_path}' not found. Please copy it to this directory."
        )

    verify_model_checksum(model_path, model_sha256)

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is required for --backend onnx. Install it first."
        ) from exc

    return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_LINEAR)
    normalized = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    return normalized[np.newaxis, np.newaxis, :, :]


def predict_cube(image, model, backend, confidence_threshold):
    input_array = preprocess_image(image)

    if backend == "onnx":
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: input_array})
        logits = torch.from_numpy(outputs[0][0])
    else:
        input_tensor = torch.from_numpy(input_array)
        with torch.no_grad():
            logits = model(input_tensor)[0].cpu()

    probabilities = torch.nn.functional.softmax(logits, dim=0)
    predicted_class = int(torch.argmax(probabilities).item())
    confidence = float(probabilities[predicted_class].item())

    if confidence < confidence_threshold:
        return UNCERTAIN_LABEL, confidence

    return CLASS_NAMES[predicted_class], confidence


def main():
    parser = argparse.ArgumentParser(
        description="Cube Detector - Real-time defect detection"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index (default: 0)",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save frames with detections",
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
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help=f"Model path (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["torchscript", "onnx"],
        default=DEFAULT_BACKEND,
        help="Inference backend to use (default: torchscript)",
    )
    parser.add_argument(
        "--model-sha256",
        type=str,
        default=None,
        help="Optional expected SHA256 checksum for model provenance validation",
    )

    args = parser.parse_args()
    exit_code = 0

    try:
        if args.backend == "onnx":
            model = load_onnx_model(args.model, model_sha256=args.model_sha256)
        else:
            model = load_torchscript_model(args.model, model_sha256=args.model_sha256)
        logger.info(
            "Model loaded successfully from '%s' using backend '%s'",
            args.model,
            args.backend,
        )
    except Exception as exc:
        logger.error(f"Error loading model: {exc}")
        return 1

    if args.save_frames:
        os.makedirs(args.save_dir, exist_ok=True)
        logger.info(f"Saving frames to: {args.save_dir}")

    cap = init_camera(camera_index=args.camera)
    if cap is None:
        logger.error("Could not initialize camera after multiple attempts")
        return 1

    logger.info("Starting cube detection. Press 'q' to quit.")
    logger.info(f"Confidence threshold: {args.threshold}")

    frame_times = deque(maxlen=FPS_WINDOW_SIZE)

    try:
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                logger.error("Could not read frame. Attempting to reconnect...")
                cap.release()
                cap = init_camera(camera_index=args.camera)

                if cap is None:
                    logger.error("Failed to reconnect. Exiting.")
                    exit_code = 1
                    break

                continue

            inference_start = time.time()
            prediction, confidence = predict_cube(
                frame,
                model=model,
                backend=args.backend,
                confidence_threshold=args.threshold,
            )
            inference_time = time.time() - inference_start

            frame_time = time.time() - frame_start
            frame_times.append(frame_time)
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

            if prediction == "good":
                color = (0, 255, 0)
            elif prediction == "defective":
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)

            cv2.putText(
                frame,
                f"{prediction}: {confidence:.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"Inf: {inference_time * 1000:.1f}ms FPS: {fps:.1f}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

            cv2.imshow("Cube Detection", frame)

            if args.save_frames:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                safe_prediction = prediction.replace(" ", "_")
                filename = os.path.join(
                    args.save_dir,
                    f"{safe_prediction}_{timestamp}.jpg",
                )
                cv2.imwrite(filename, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user. Cleaning up...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Cube detection stopped")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
