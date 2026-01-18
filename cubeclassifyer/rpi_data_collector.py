"""
Raspberry Pi Data Collection Script for Cube Classifier
Captures 224x224 grayscale images using Pi Camera

Controls:
  'g' - Save as GOOD cube
  'd' - Save as DEFECTIVE cube
  'q' - Quit

Usage:
  python rpi_data_collector.py
  python rpi_data_collector.py --output-dir my_dataset
"""

import cv2
import os
import argparse
from datetime import datetime


def init_camera(camera_index=0, width=640, height=480):
    """Initialize camera with specified resolution"""
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap
    return None


def create_directories(base_dir):
    """Create output directories for good and defective images"""
    dirs = {
        "good": os.path.join(base_dir, "good"),
        "defective": os.path.join(base_dir, "defective"),
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs


def get_next_filename(directory, prefix):
    """Get next available filename with incrementing number"""
    existing = [f for f in os.listdir(directory) if f.startswith(prefix)]
    if not existing:
        return f"{prefix}_001.jpg"

    numbers = []
    for f in existing:
        try:
            num = int(f.split("_")[1].split(".")[0])
            numbers.append(num)
        except (IndexError, ValueError):
            continue

    next_num = max(numbers) + 1 if numbers else 1
    return f"{prefix}_{next_num:03d}.jpg"


def draw_ui(
    frame, good_count, defective_count, status_msg="", status_color=(255, 255, 255)
):
    """Draw UI overlay on frame"""
    h, w = frame.shape[:2]

    # Semi-transparent header background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(
        frame,
        "CUBE DATA COLLECTOR",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    # Counts
    cv2.putText(
        frame,
        f"Good: {good_count}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Defective: {defective_count}",
        (150, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 255),
        2,
    )

    # Controls
    cv2.putText(
        frame,
        "[G] Good  [D] Defective  [Q] Quit",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )

    # Status message at bottom
    if status_msg:
        cv2.rectangle(frame, (0, h - 40), (w, h), status_color, -1)
        cv2.putText(
            frame,
            status_msg,
            (10, h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    # Center crosshair for alignment
    center_x, center_y = w // 2, h // 2
    size = 20
    cv2.line(
        frame,
        (center_x - size, center_y),
        (center_x + size, center_y),
        (0, 255, 255),
        1,
    )
    cv2.line(
        frame,
        (center_x, center_y - size),
        (center_x, center_y + size),
        (0, 255, 255),
        1,
    )

    # 224x224 capture region box (centered)
    box_size = 224
    x1 = center_x - box_size // 2
    y1 = center_y - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

    return frame, (x1, y1, x2, y2)


def process_and_save(frame, roi, save_path):
    """Process frame and save as 224x224 grayscale"""
    x1, y1, x2, y2 = roi

    # Crop to ROI
    cropped = frame[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Resize to 224x224 (should already be, but ensure)
    resized = cv2.resize(gray, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Save
    cv2.imwrite(save_path, resized)
    return True


def count_images(directory):
    """Count images in directory"""
    if not os.path.exists(directory):
        return 0
    return len(
        [f for f in os.listdir(directory) if f.endswith((".jpg", ".jpeg", ".png"))]
    )


def main():
    parser = argparse.ArgumentParser(description="Cube Data Collector for Raspberry Pi")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="collected_data",
        help="Output directory for collected images (default: collected_data)",
    )
    parser.add_argument(
        "--camera", type=int, default=0, help="Camera index (default: 0)"
    )
    args = parser.parse_args()

    # Create directories
    dirs = create_directories(args.output_dir)
    print(f"Saving images to: {args.output_dir}/")
    print(f"  - good/")
    print(f"  - defective/")

    # Initialize camera
    cap = init_camera(args.camera)
    if cap is None:
        print("ERROR: Could not initialize camera!")
        return

    print("\nCamera initialized successfully!")
    print("\nControls:")
    print("  [G] - Save as GOOD cube")
    print("  [D] - Save as DEFECTIVE cube")
    print("  [Q] - Quit")
    print("\nPosition the cube inside the yellow box and press G or D to save.")

    status_msg = ""
    status_color = (100, 100, 100)
    status_timeout = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Could not read frame!")
                break

            # Count existing images
            good_count = count_images(dirs["good"])
            defective_count = count_images(dirs["defective"])

            # Clear status after timeout
            if status_timeout > 0:
                status_timeout -= 1
            else:
                status_msg = ""

            # Draw UI and get ROI
            display_frame, roi = draw_ui(
                frame.copy(), good_count, defective_count, status_msg, status_color
            )

            # Show frame
            cv2.imshow("Cube Data Collector", display_frame)

            # Handle key press
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                print("\nQuitting...")
                break

            elif key == ord("g") or key == ord("G"):
                filename = get_next_filename(dirs["good"], "good")
                save_path = os.path.join(dirs["good"], filename)
                if process_and_save(frame, roi, save_path):
                    status_msg = f"SAVED: {filename}"
                    status_color = (0, 150, 0)  # Green
                    status_timeout = 30  # ~1 second at 30fps
                    print(f"Saved GOOD: {save_path}")

            elif key == ord("d") or key == ord("D"):
                filename = get_next_filename(dirs["defective"], "defective")
                save_path = os.path.join(dirs["defective"], filename)
                if process_and_save(frame, roi, save_path):
                    status_msg = f"SAVED: {filename}"
                    status_color = (0, 0, 150)  # Red
                    status_timeout = 30
                    print(f"Saved DEFECTIVE: {save_path}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        cap.release()
        cv2.destroyAllWindows()

        # Print summary
        good_count = count_images(dirs["good"])
        defective_count = count_images(dirs["defective"])
        print(f"\n{'=' * 40}")
        print("COLLECTION SUMMARY")
        print(f"{'=' * 40}")
        print(f"Good images:      {good_count}")
        print(f"Defective images: {defective_count}")
        print(f"Total:            {good_count + defective_count}")
        print(f"{'=' * 40}")
        print(f"\nImages saved to: {os.path.abspath(args.output_dir)}")
        print("\nNext steps:")
        print("  1. Transfer images to your training PC")
        print("  2. Split into train/val folders (80/20)")
        print("  3. Run: python main.py train")


if __name__ == "__main__":
    main()
