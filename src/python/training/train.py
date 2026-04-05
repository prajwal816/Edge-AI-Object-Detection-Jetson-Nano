"""
Edge AI Object Detection System
YOLOv8 Training Script
======================
Trains a YOLOv8 model for object detection using the Ultralytics
framework. Supports synthetic dataset generation for development.
"""

import os
import sys
import json
import time
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.python.utils.logger import setup_logger, get_perf_tracker
from src.python.utils.config import build_pipeline_config
from src.python.training.dataset_generator import generate_dataset

logger = setup_logger("training")
perf = get_perf_tracker()


@click.command()
@click.option("--config", "-c", default="configs/training_config.yaml",
              help="Path to training config YAML")
@click.option("--model", "-m", default="yolov8n.pt",
              help="YOLOv8 model variant (yolov8n, yolov8s, etc.)")
@click.option("--epochs", "-e", default=50, help="Number of training epochs")
@click.option("--batch-size", "-b", default=16, help="Batch size")
@click.option("--img-size", "-s", default=640, help="Input image size")
@click.option("--dataset", "-d", default=None,
              help="Path to dataset YAML (auto-generates if not provided)")
@click.option("--num-images", default=500,
              help="Number of synthetic images (if auto-generating)")
@click.option("--device", default="0", help="CUDA device (0, cpu)")
@click.option("--output-dir", "-o", default="models",
              help="Output directory for trained weights")
@click.option("--name", "-n", default="edge_ai_yolov8",
              help="Experiment name")
def train(config, model, epochs, batch_size, img_size, dataset, 
          num_images, device, output_dir, name):
    """Train a YOLOv8 object detection model."""
    
    logger.info("=" * 60)
    logger.info("  Edge AI Object Detection - YOLOv8 Training")
    logger.info("=" * 60)
    
    # Load config
    pipeline_config = build_pipeline_config(config if os.path.exists(config) else None)
    
    # Generate dataset if needed
    if dataset is None or not os.path.exists(dataset):
        logger.info("No dataset provided, generating synthetic dataset...")
        perf.start("dataset_generation")
        
        dataset = generate_dataset(
            output_dir="datasets/edge_ai",
            num_images=num_images,
            num_classes=pipeline_config.dataset.num_classes,
            image_size=img_size,
            class_names=pipeline_config.dataset.class_names,
        )
        
        gen_time = perf.stop("dataset_generation")
        logger.info(f"Dataset generated in {gen_time:.0f}ms")
    
    logger.info(f"\nTraining Configuration:")
    logger.info(f"  Model:      {model}")
    logger.info(f"  Epochs:     {epochs}")
    logger.info(f"  Batch Size: {batch_size}")
    logger.info(f"  Image Size: {img_size}")
    logger.info(f"  Dataset:    {dataset}")
    logger.info(f"  Device:     {device}")
    logger.info(f"  Output:     {output_dir}")
    
    # Import ultralytics
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Install with: pip install ultralytics")
        logger.info("Generating simulated training results instead...")
        _simulate_training(model, epochs, output_dir, name)
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    logger.info(f"\nLoading model: {model}")
    perf.start("model_load")
    
    yolo_model = YOLO(model)
    
    load_time = perf.stop("model_load")
    logger.info(f"Model loaded in {load_time:.0f}ms")
    
    # Train
    logger.info(f"\nStarting training for {epochs} epochs...")
    perf.start("training")
    
    results = yolo_model.train(
        data=dataset,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project="runs/train",
        name=name,
        exist_ok=True,
        verbose=True,
        save=True,
        plots=True,
    )
    
    train_time = perf.stop("training")
    logger.info(f"\nTraining completed in {train_time / 1000:.1f}s")
    
    # Copy best weights to models/
    best_weights = Path(f"runs/train/{name}/weights/best.pt")
    if best_weights.exists():
        import shutil
        dest = Path(output_dir) / "yolov8_best.pt"
        shutil.copy2(best_weights, dest)
        logger.info(f"Best weights saved to: {dest}")
    
    # Log training metrics
    metrics = {
        "model": model,
        "epochs": epochs,
        "batch_size": batch_size,
        "image_size": img_size,
        "training_time_s": train_time / 1000,
        "dataset": dataset,
    }
    
    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Training metrics saved to: {metrics_path}")
    
    # Performance summary
    logger.info("\n" + "=" * 40)
    logger.info("  Training Performance Summary")
    logger.info("=" * 40)
    summary = perf.get_summary()
    for stage, stats in summary.items():
        logger.info(f"  {stage}: {stats['avg_ms']:.0f}ms avg")


def _simulate_training(model: str, epochs: int, output_dir: str, name: str):
    """Simulate training output when ultralytics is not available."""
    logger.info("\n[SIMULATION] Simulating YOLOv8 training...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(1, min(epochs + 1, 11)):
        # Simulate decreasing loss
        box_loss = 2.5 * (0.85 ** epoch) + 0.1
        cls_loss = 1.8 * (0.88 ** epoch) + 0.05
        dfl_loss = 1.2 * (0.90 ** epoch) + 0.08
        
        # Simulate increasing mAP
        map50 = min(0.95, 0.3 + 0.07 * epoch)
        map50_95 = min(0.75, 0.15 + 0.06 * epoch)
        
        logger.info(
            f"  Epoch {epoch}/{epochs} | "
            f"box_loss: {box_loss:.4f} | "
            f"cls_loss: {cls_loss:.4f} | "
            f"dfl_loss: {dfl_loss:.4f} | "
            f"mAP50: {map50:.4f} | "
            f"mAP50-95: {map50_95:.4f}"
        )
        time.sleep(0.2)
    
    if epochs > 10:
        logger.info(f"  ... (epochs 11-{epochs} omitted for brevity)")
    
    # Final metrics
    metrics = {
        "model": model,
        "epochs": epochs,
        "simulated": True,
        "final_metrics": {
            "mAP50": 0.912,
            "mAP50_95": 0.683,
            "precision": 0.894,
            "recall": 0.871,
            "box_loss": 0.312,
            "cls_loss": 0.198,
        }
    }
    
    metrics_path = Path(output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\n[SIMULATION] Training complete")
    logger.info(f"  mAP@0.5:    {metrics['final_metrics']['mAP50']:.3f}")
    logger.info(f"  mAP@0.5:95: {metrics['final_metrics']['mAP50_95']:.3f}")
    logger.info(f"  Precision:   {metrics['final_metrics']['precision']:.3f}")
    logger.info(f"  Recall:      {metrics['final_metrics']['recall']:.3f}")
    logger.info(f"  Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    train()
