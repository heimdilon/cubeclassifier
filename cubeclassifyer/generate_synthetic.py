"""
Synthetic defective cube generator
Generates synthetic defects on undamaged cube images to augment your dataset
"""

import cv2
import numpy as np
import random
import os
import argparse
from pathlib import Path


def add_scratch(image, intensity_range=(30, 70), length_range=(10, 50)):
    """Add scratch defects (dark lines) to image"""
    defective = image.copy()
    h, w = defective.shape

    # Number of scratches
    num_scratches = random.randint(1, 4)

    for _ in range(num_scratches):
        # Random scratch parameters
        x1, y1 = random.randint(0, w - 1), random.randint(0, h - 1)
        angle = random.randint(0, 360)
        length = random.randint(*length_range)

        # Calculate end point
        x2 = int(x1 + length * np.cos(np.radians(angle)))
        y2 = int(y1 + length * np.sin(np.radians(angle)))

        # Draw scratch (dark line)
        intensity = random.randint(*intensity_range)
        thickness = random.randint(1, 2)
        cv2.line(defective, (x1, y1), (x2, y2), intensity, -1, thickness)

        # Add slight thickness variation
        if random.random() > 0.5:
            cv2.line(defective, (x1 + 1, y1), (x2 + 1, y2), intensity // 2, -1, 1)

    return defective


def add_dent(image, radius_range=(8, 25)):
    """Add dent defects (dark circles/ellipses) to image"""
    defective = image.copy()
    h, w = defective.shape

    # Number of dents
    num_dents = random.randint(1, 3)

    for _ in range(num_dents):
        x = random.randint(20, w - 20)
        y = random.randint(20, h - 20)
        radius = random.randint(*radius_range)

        # Shape: circle or ellipse
        if random.random() > 0.5:
            cv2.circle(defective, (x, y), radius, random.randint(20, 50), -1, -1)
        else:
            radius_y = random.randint(radius // 2, radius * 2)
            cv2.ellipse(
                defective,
                (x, y),
                (radius, radius_y),
                0,
                360,
                random.randint(20, 50),
                -1,
                -1,
            )

    return defective


def add_crack(image, length_range=(15, 40)):
    """Add crack defects (jagged dark lines) to image"""
    defective = image.copy()
    h, w = defective.shape

    # Number of cracks
    num_cracks = random.randint(1, 3)

    for _ in range(num_cracks):
        # Starting point
        x, y = random.randint(10, w - 10), random.randint(10, h - 10)

        # Generate jagged crack path
        num_points = random.randint(4, 8)
        points = [(x, y)]

        for _ in range(num_points):
            # Random offset to create jaggedness
            offset_x = random.randint(-10, 10)
            offset_y = random.randint(-10, 10)
            x = max(0, min(w - 1, x + offset_x))
            y = max(0, min(h - 1, y + offset_y))
            points.append((x, y))

        # Draw crack
        points = np.array(points, dtype=np.int32)
        length = random.randint(*length_range)
        thickness = random.randint(1, 3)
        intensity = random.randint(30, 60)

        cv2.polylines(defective, [points], False, intensity, -1, thickness)

    return defective


def add_stain(image, radius_range=(10, 35)):
    """Add stain defects (dark blobs) to image"""
    defective = image.copy()
    h, w = defective.shape

    # Number of stains
    num_stains = random.randint(1, 3)

    for _ in range(num_stains):
        x = random.randint(10, w - 10)
        y = random.randint(10, h - 10)
        radius_x = random.randint(*radius_range)
        radius_y = random.randint(*radius_range)

        # Draw stain (dark ellipse)
        intensity = random.randint(30, 60)
        cv2.ellipse(defective, (x, y), (radius_x, radius_y), 0, 360, intensity, -1, -1)

        # Add some blur for realism
        if random.random() > 0.5:
            blurred = cv2.GaussianBlur(defective, (3, 3), 0)
            # Blend original and blurred
            defective = cv2.addWeighted(defective, 0.7, blurred, 0.3)

    return defective


def add_combined_defects(image, max_defects=5):
    """Add combination of multiple defect types"""
    defective = image.copy()

    defect_types = [add_scratch, add_dent, add_crack, add_stain]
    num_defects = random.randint(1, max_defects)

    for _ in range(num_defects):
        defect_func = random.choice(defect_types)
        defective = defect_func(defective)

    return defective


def generate_synthetic_dataset(
    input_dir, output_dir, num_per_image=5, defect_type="combined"
):
    """
    Generate synthetic defective images from undamaged cubes

    Args:
        input_dir: Directory with undamaged cube images
        output_dir: Directory to save synthetic defective images
        num_per_image: Number of synthetic variants per input image
        defect_type: Type of defects ('scratch', 'dent', 'crack', 'stain', 'combined')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_files = [
        f
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if len(input_files) == 0:
        print(f"Error: No images found in {input_dir}")
        return

    print(f"Found {len(input_files)} images in {input_dir}")
    print(f"Generating {num_per_image} synthetic variants each...")
    print(f"Defect type: {defect_type}")

    generated_count = 0

    for filename in input_files:
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Warning: Could not load {image_path}")
            continue

        base_name = os.path.splitext(filename)[0]
        extension = os.path.splitext(filename)[1]

        # Generate synthetic variants
        for i in range(num_per_image):
            if defect_type == "scratch":
                defective = add_scratch(image)
            elif defect_type == "dent":
                defective = add_dent(image)
            elif defect_type == "crack":
                defective = add_crack(image)
            elif defect_type == "stain":
                defective = add_stain(image)
            elif defect_type == "combined":
                defective = add_combined_defects(image)
            else:
                defective = add_combined_defects(image)

            # Save synthetic image
            output_filename = f"{base_name}_synthetic_{i}{extension}"
            output_file = output_path / output_filename
            cv2.imwrite(str(output_file), defective)
            generated_count += 1

        if (generated_count % 10) == 0:
            print(f"Generated {generated_count} synthetic images...")

    print(f"\nComplete! Generated {generated_count} synthetic defective images")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic defective cube images"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="cube_dataset/train/good",
        help="Directory with undamaged cube images",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="cube_dataset/train/defective_synth",
        help="Output directory for synthetic defective images",
    )
    parser.add_argument(
        "--num-per-image",
        type=int,
        default=5,
        help="Number of synthetic variants per input image (default: 5)",
    )
    parser.add_argument(
        "--defect-type",
        type=str,
        default="combined",
        choices=["scratch", "dent", "crack", "stain", "combined"],
        help="Type of defects to generate (default: combined)",
    )

    args = parser.parse_args()

    # Verify input directory exists
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        print("Please create it and add your undamaged cube images")
        return

    generate_synthetic_dataset(
        input_dir=args.input,
        output_dir=args.output,
        num_per_image=args.num_per_image,
        defect_type=args.defect_type,
    )


if __name__ == "__main__":
    main()
