"""
Edge AI Object Detection System
Configuration Loader
====================
YAML-based configuration with dataclass validation.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Model training and export configuration."""
    variant: str = "yolov8n"
    num_classes: int = 80
    input_size: int = 640
    pretrained: bool = True


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 50
    batch_size: int = 16
    learning_rate: float = 0.01
    optimizer: str = "SGD"
    augmentation: bool = True
    workers: int = 4
    device: str = "0"
    project: str = "runs/train"
    name: str = "edge_ai_yolov8"


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    root: str = "datasets/edge_ai"
    train_split: float = 0.8
    num_images: int = 500
    num_classes: int = 5
    image_size: int = 640
    class_names: List[str] = field(default_factory=lambda: [
        "person", "car", "bicycle", "motorcycle", "bus"
    ])


@dataclass
class ExportConfig:
    """Model export configuration."""
    onnx_opset: int = 12
    dynamic_batch: bool = True
    simplify: bool = True
    output_dir: str = "models"
    
    # TensorRT settings
    trt_precision: str = "fp16"  # fp32, fp16, int8
    trt_workspace_mb: int = 1024
    trt_max_batch: int = 1
    int8_calibration_images: int = 100


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    
    return config or {}


def build_pipeline_config(config_path: Optional[str] = None) -> PipelineConfig:
    """Build a PipelineConfig from YAML file or defaults."""
    config = PipelineConfig()
    
    if config_path and os.path.exists(config_path):
        raw = load_config(config_path)
        
        if "model" in raw:
            for k, v in raw["model"].items():
                if hasattr(config.model, k):
                    setattr(config.model, k, v)
        
        if "training" in raw:
            for k, v in raw["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)
        
        if "dataset" in raw:
            for k, v in raw["dataset"].items():
                if hasattr(config.dataset, k):
                    setattr(config.dataset, k, v)
        
        if "export" in raw:
            for k, v in raw["export"].items():
                if hasattr(config.export, k):
                    setattr(config.export, k, v)
    
    return config
