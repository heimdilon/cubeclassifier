"""
Benchmark script for cube classifier
Run this before and after optimizations to measure improvements
"""

import torch
import torch.nn as nn
import time
import numpy as np
from cube_classifier import LightweightCubeClassifier
import os


def benchmark_model_inference(model, input_shape=(1, 1, 240, 320), iterations=1000):
    """
    Benchmark model inference speed

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, channels, height, width)
        iterations: Number of iterations to run

    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    device = next(model.parameters()).device

    # Warmup
    dummy_input = torch.randn(input_shape).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    # Actual benchmark
    times = []
    with torch.no_grad():
        for _ in range(iterations):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            times.append(end - start)

    # Calculate statistics
    times = np.array(times) * 1000  # Convert to ms

    results = {
        "avg": np.mean(times),
        "std": np.std(times),
        "median": np.median(times),
        "min": np.min(times),
        "max": np.max(times),
        "p95": np.percentile(times, 95),
        "p99": np.percentile(times, 99),
        "fps": 1000 / np.mean(times),  # Frames per second
    }

    return results, model


def benchmark_model_size(model_path):
    """
    Get model file size

    Args:
        model_path: Path to model file

    Returns:
        Size in KB and MB
    """
    size_bytes = os.path.getsize(model_path)
    size_kb = size_bytes / 1024
    size_mb = size_bytes / (1024 * 1024)

    return {"bytes": size_bytes, "kb": size_kb, "mb": size_mb}


def benchmark_parameters(model):
    """
    Count model parameters

    Args:
        model: PyTorch model

    Returns:
        Parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params,
    }


def print_benchmark_results(results, model_name="Model"):
    """
    Print benchmark results in a nice format
    """
    print(f"\n{'=' * 70}")
    print(f"  {model_name} Benchmark Results")
    print(f"{'=' * 70}\n")

    # Inference timing
    print("Inference Speed:")
    print(f"  Average:      {results['avg']:.2f} ms")
    print(f"  Median:       {results['median']:.2f} ms")
    print(f"  Std Dev:      {results['std']:.2f} ms")
    print(f"  Min:          {results['min']:.2f} ms")
    print(f"  Max:          {results['max']:.2f} ms")
    print(f"  P95:          {results['p95']:.2f} ms")
    print(f"  P99:          {results['p99']:.2f} ms")
    print(f"  FPS:          {results['fps']:.2f} frames/sec")


def compare_models(
    base_results, optimized_results, base_name="Base", optimized_name="Optimized"
):
    """
    Compare two benchmark results
    """
    print(f"\n{'=' * 70}")
    print(f"  Comparison: {base_name} vs {optimized_name}")
    print(f"{'=' * 70}\n")

    metrics = [
        ("Average Time", base_results["avg"], optimized_results["avg"], "ms"),
        ("FPS", base_results["fps"], optimized_results["fps"], "fps"),
    ]

    for name, base_val, opt_val, unit in metrics:
        if opt_val < base_val:
            improvement = ((base_val - opt_val) / base_val) * 100
            print(f"{name}:")
            print(f"  {base_name}:    {base_val:.2f} {unit}")
            print(f"  {optimized_name}: {opt_val:.2f} {unit}")
            print(f"  Improvement: {improvement:+.1f}%")
        else:
            regression = ((opt_val - base_val) / base_val) * 100
            print(f"{name}:")
            print(f"  {base_name}:    {base_val:.2f} {unit}")
            print(f"  {optimized_name}: {opt_val:.2f} {unit}")
            print(f"  Regression:  {regression:+.1f}% ⚠️")


def main():
    print("\nCube Classifier Benchmark")
    print("=" * 70)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    # Create and benchmark model
    print("\nInitializing model...")
    model = LightweightCubeClassifier(num_classes=2)
    model.to(device)

    # Benchmark inference speed
    print("\nRunning inference benchmark...")
    results, model = benchmark_model_inference(model, iterations=1000)
    print_benchmark_results(results, "LightweightCubeClassifier")

    # Benchmark model size (if model file exists)
    model_path = "best_cube_classifier.pth"
    if os.path.exists(model_path):
        print(f"\nModel File Size ({model_path}):")
        size = benchmark_model_size(model_path)
        print(f"  {size['kb']:.2f} KB")
        print(f"  {size['mb']:.4f} MB")

    # Benchmark parameters
    params = benchmark_parameters(model)
    print("\nModel Parameters:")
    print(f"  Total:        {params['total']:,}")
    print(f"  Trainable:   {params['trainable']:,}")
    print(f"  Non-trainable: {params['non_trainable']:,}")

    # Memory usage (if CUDA available)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        dummy_input = torch.randn(1, 1, 240, 320).to(device)

        with torch.no_grad():
            _ = model(dummy_input)

        max_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        print(f"\nPeak GPU Memory:")
        print(f"  {max_memory:.3f} GB")

    print(f"\n{'=' * 70}")
    print("  Benchmark Complete!")
    print(f"{'=' * 70}\n")

    print("\n💡 Tips for optimization:")
    print("  - Run this script BEFORE applying optimizations")
    print("  - Apply optimizations from OPTIMIZATION_SUGGESTIONS.md")
    print("  - Run this script AGAIN to measure improvements")
    print("  - Compare results to see actual gains")
    print("\n  Expected improvements:")
    print("    - Mixed Precision: 50-100% faster training")
    print("    - Quantization: 200-300% faster inference")
    print("    - MobileNetV3: 30-50% faster inference, +10-20% accuracy")


if __name__ == "__main__":
    main()
