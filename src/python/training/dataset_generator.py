"""
Edge AI Object Detection System
Synthetic Dataset Generator
============================
Generates a YOLO-format training dataset with synthetic images
containing geometric shapes as object proxies. Used for training
and validating the pipeline without requiring real data.
"""

import os
import random
import math
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.python.utils.logger import setup_logger

logger = setup_logger("dataset_gen")


# Default class definitions (proxied by shapes)
DEFAULT_CLASSES = {
    0: {"name": "person",     "shape": "rectangle", "color": (60, 60, 220)},
    1: {"name": "car",        "shape": "rectangle", "color": (220, 140, 60)},
    2: {"name": "bicycle",    "shape": "triangle",  "color": (60, 200, 60)},
    3: {"name": "motorcycle", "shape": "ellipse",   "color": (220, 200, 60)},
    4: {"name": "bus",        "shape": "rectangle", "color": (180, 60, 220)},
}


def generate_background(width: int, height: int) -> np.ndarray:
    """Generate a varied background image."""
    bg_type = random.choice(["gradient", "noise", "solid", "pattern"])
    
    if bg_type == "gradient":
        c1 = np.array([random.randint(10, 60)] * 3, dtype=np.float32)
        c2 = np.array([random.randint(40, 100)] * 3, dtype=np.float32)
        ys = np.linspace(0, 1, height).reshape(-1, 1, 1)
        grad = (c1 * (1 - ys) + c2 * ys).astype(np.uint8)
        img = np.broadcast_to(grad, (height, width, 3)).copy()
    elif bg_type == "noise":
        base = random.randint(20, 50)
        img = np.random.randint(base - 15, base + 15, (height, width, 3), dtype=np.uint8)
    elif bg_type == "pattern":
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = (random.randint(20, 50), random.randint(20, 50), random.randint(20, 50))
        # Add grid lines
        spacing = random.choice([20, 30, 40])
        line_color = tuple([random.randint(40, 80)] * 3)
        for x in range(0, width, spacing):
            cv2.line(img, (x, 0), (x, height), line_color, 1)
        for y in range(0, height, spacing):
            cv2.line(img, (0, y), (width, y), line_color, 1)
    else:
        c = (random.randint(20, 60), random.randint(20, 60), random.randint(20, 60))
        img = np.full((height, width, 3), c, dtype=np.uint8)
    
    return img


def draw_object(
    img: np.ndarray,
    class_id: int,
    x: int, y: int, w: int, h: int,
    classes: dict = DEFAULT_CLASSES,
) -> None:
    """Draw a synthetic object on the image."""
    cls_info = classes[class_id]
    shape = cls_info["shape"]
    base_color = cls_info["color"]
    
    # Vary color slightly
    color = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in base_color)
    
    if shape == "rectangle":
        cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
        # Internal detail
        margin = max(3, min(w, h) // 6)
        darker = tuple(max(0, c - 40) for c in color)
        cv2.rectangle(img, (x + margin, y + margin), 
                     (x + w - margin, y + h - margin), darker, -1)
        # Highlight edge
        lighter = tuple(min(255, c + 30) for c in color)
        cv2.rectangle(img, (x, y), (x + w, y + h), lighter, 2)
        
    elif shape == "ellipse":
        center = (x + w // 2, y + h // 2)
        axes = (w // 2, h // 2)
        cv2.ellipse(img, center, axes, 0, 0, 360, color, -1)
        # Inner ellipse
        inner_axes = (max(1, w // 4), max(1, h // 4))
        darker = tuple(max(0, c - 50) for c in color)
        cv2.ellipse(img, center, inner_axes, 0, 0, 360, darker, -1)
        
    elif shape == "triangle":
        pts = np.array([
            [x + w // 2, y],
            [x, y + h],
            [x + w, y + h],
        ], dtype=np.int32)
        cv2.fillPoly(img, [pts], color)
        # Inner triangle
        margin = max(3, min(w, h) // 5)
        inner_pts = np.array([
            [x + w // 2, y + margin],
            [x + margin, y + h - margin],
            [x + w - margin, y + h - margin],
        ], dtype=np.int32)
        darker = tuple(max(0, c - 50) for c in color)
        cv2.fillPoly(img, [inner_pts], darker)


def generate_dataset(
    output_dir: str,
    num_images: int = 500,
    num_classes: int = 5,
    image_size: int = 640,
    min_objects: int = 1,
    max_objects: int = 8,
    train_split: float = 0.8,
    class_names: List[str] = None,
) -> str:
    """
    Generate a complete YOLO-format synthetic dataset.
    
    Args:
        output_dir: Root directory for the dataset
        num_images: Total number of images to generate
        num_classes: Number of object classes
        image_size: Image width and height
        min_objects: Minimum objects per image
        max_objects: Maximum objects per image
        train_split: Fraction of images for training
        class_names: Optional custom class names
    
    Returns:
        Path to the generated dataset YAML file
    """
    output_dir = Path(output_dir)
    
    # Create directory structure
    for split in ["train", "val"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Use class names
    if class_names is None:
        class_names = [DEFAULT_CLASSES.get(i, {"name": f"class_{i}"})["name"] 
                       for i in range(num_classes)]
    
    # Generate images
    num_train = int(num_images * train_split)
    indices = list(range(num_images))
    random.shuffle(indices)
    
    logger.info(f"Generating {num_images} synthetic images ({num_train} train, "
                f"{num_images - num_train} val)")
    logger.info(f"  Classes: {class_names}")
    logger.info(f"  Image size: {image_size}x{image_size}")
    logger.info(f"  Objects per image: {min_objects}-{max_objects}")
    
    for idx in range(num_images):
        split = "train" if idx < num_train else "val"
        img_name = f"img_{idx:06d}.jpg"
        lbl_name = f"img_{idx:06d}.txt"
        
        # Generate image
        img = generate_background(image_size, image_size)
        
        # Add objects
        num_objects = random.randint(min_objects, max_objects)
        labels = []
        
        for _ in range(num_objects):
            class_id = random.randint(0, num_classes - 1)
            
            # Random size (5-30% of image)
            obj_w = random.randint(int(image_size * 0.05), int(image_size * 0.3))
            obj_h = random.randint(int(image_size * 0.05), int(image_size * 0.3))
            
            # Random position
            obj_x = random.randint(0, image_size - obj_w)
            obj_y = random.randint(0, image_size - obj_h)
            
            # Draw object
            draw_object(img, class_id, obj_x, obj_y, obj_w, obj_h)
            
            # YOLO format: class_id, center_x, center_y, width, height (normalized)
            cx = (obj_x + obj_w / 2) / image_size
            cy = (obj_y + obj_h / 2) / image_size
            nw = obj_w / image_size
            nh = obj_h / image_size
            
            labels.append(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        
        # Apply random augmentations
        if random.random() > 0.5:
            # Brightness variation
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
        
        if random.random() > 0.7:
            # Gaussian blur
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)
        
        # Save image and label
        cv2.imwrite(str(output_dir / split / "images" / img_name), img)
        
        with open(output_dir / split / "labels" / lbl_name, "w") as f:
            f.write("\n".join(labels))
        
        if (idx + 1) % 100 == 0:
            logger.info(f"  Generated {idx + 1}/{num_images} images")
    
    # Create dataset YAML
    yaml_path = output_dir / "dataset.yaml"
    yaml_content = (
        f"# Edge AI Synthetic Dataset\n"
        f"# Generated: {num_images} images, {num_classes} classes\n\n"
        f"path: {output_dir.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n\n"
        f"nc: {num_classes}\n"
        f"names: {class_names}\n"
    )
    
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    
    logger.info(f"Dataset generated: {yaml_path}")
    logger.info(f"  Train: {num_train} images")
    logger.info(f"  Val: {num_images - num_train} images")
    
    return str(yaml_path)


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--output", "-o", default="datasets/edge_ai", help="Output directory")
    @click.option("--num-images", "-n", default=500, help="Number of images")
    @click.option("--num-classes", "-c", default=5, help="Number of classes")
    @click.option("--image-size", "-s", default=640, help="Image size")
    @click.option("--train-split", default=0.8, help="Train/val split ratio")
    def main(output, num_images, num_classes, image_size, train_split):
        """Generate a synthetic YOLO-format dataset."""
        generate_dataset(
            output_dir=output,
            num_images=num_images,
            num_classes=num_classes,
            image_size=image_size,
            train_split=train_split,
        )

    main()
