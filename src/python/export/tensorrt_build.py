"""
Edge AI Object Detection System
TensorRT Engine Builder
=======================
Converts ONNX model to TensorRT engine with configurable
precision (FP32/FP16/INT8) for deployment on NVIDIA Jetson.
"""

import os
import sys
import json
import time
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.python.utils.logger import setup_logger

logger = setup_logger("trt_build")


@click.command()
@click.option("--onnx", "-i", default="models/yolov8_det.onnx",
              help="Input ONNX model path")
@click.option("--output", "-o", default="models/yolov8_det.engine",
              help="Output TensorRT engine path")
@click.option("--precision", "-p", default="fp16",
              type=click.Choice(["fp32", "fp16", "int8"]),
              help="Inference precision")
@click.option("--workspace", "-w", default=1024, help="Workspace size in MB")
@click.option("--max-batch", default=1, help="Maximum batch size")
@click.option("--calib-images", default=100,
              help="Number of calibration images for INT8")
@click.option("--verbose/--quiet", default=False, help="Verbose TRT logging")
def build_engine(onnx, output, precision, workspace, max_batch, 
                 calib_images, verbose):
    """Build TensorRT engine from ONNX model."""
    
    logger.info("=" * 60)
    logger.info("  Edge AI - TensorRT Engine Builder")
    logger.info("=" * 60)
    logger.info(f"  ONNX Input:  {onnx}")
    logger.info(f"  Engine Out:  {output}")
    logger.info(f"  Precision:   {precision.upper()}")
    logger.info(f"  Workspace:   {workspace} MB")
    logger.info(f"  Max Batch:   {max_batch}")
    
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    
    # Try real TensorRT build
    try:
        import tensorrt as trt
        
        if not os.path.exists(onnx):
            logger.error(f"ONNX file not found: {onnx}")
            return
        
        _build_real_engine(onnx, output, precision, workspace, 
                          max_batch, calib_images, verbose)
        
    except ImportError:
        logger.warning("TensorRT not available, generating simulated engine")
        _simulate_engine_build(onnx, output, precision, workspace, max_batch)
    
    # Save build metadata
    meta = {
        "onnx_path": onnx,
        "engine_path": output,
        "precision": precision,
        "workspace_mb": workspace,
        "max_batch_size": max_batch,
        "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    if os.path.exists(output):
        meta["engine_size_mb"] = os.path.getsize(output) / (1024 * 1024)
    
    meta_path = Path(output).parent / "engine_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    logger.info(f"\nBuild complete: {output}")


def _build_real_engine(onnx_path, engine_path, precision, workspace_mb,
                       max_batch, calib_images, verbose):
    """Build engine using real TensorRT API."""
    import tensorrt as trt
    
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    
    logger.info("\nBuilding TensorRT engine...")
    start = time.perf_counter()
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 
                                 workspace_mb * (1 << 20))
    
    # Set precision
    if precision == "fp16" and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        logger.info("  FP16 mode enabled")
    elif precision == "int8" and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        logger.info("  INT8 mode enabled")
    
    # Parse ONNX
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                logger.error(f"  Parser error: {parser.get_error(i)}")
            return
    
    logger.info(f"  Network parsed: {network.num_inputs} inputs, "
                f"{network.num_outputs} outputs")
    
    # Build engine
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        logger.error("  Engine build failed")
        return
    
    with open(engine_path, "wb") as f:
        f.write(serialized)
    
    elapsed = time.perf_counter() - start
    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    logger.info(f"  Engine built in {elapsed:.1f}s ({size_mb:.1f} MB)")


def _simulate_engine_build(onnx_path, engine_path, precision, workspace_mb,
                           max_batch):
    """Simulate TensorRT engine building."""
    logger.info("\n[SIMULATION] Building TensorRT engine...")
    
    # Simulated build stages
    stages = [
        ("Parsing ONNX model", 0.5),
        ("Analyzing network topology", 0.3),
        ("Optimizing layer fusion", 0.8),
        (f"Applying {precision.upper()} quantization", 0.4),
        ("Building CUDA kernels", 1.0),
        ("Profiling kernel performance", 0.6),
        ("Selecting optimal tactics", 0.5),
        ("Serializing engine", 0.3),
    ]
    
    for stage_name, delay in stages:
        logger.info(f"  {stage_name}...")
        time.sleep(delay * 0.3)  # Reduced delay for simulation
    
    # Engine size based on precision
    size_map = {"fp32": 12, "fp16": 8, "int8": 5}
    engine_size_mb = size_map.get(precision, 8)
    
    # Create simulated engine file
    header = b"TENSORRT_ENGINE_SIMULATED_v8"
    meta = json.dumps({
        "format": "trt_engine_simulated",
        "precision": precision,
        "max_batch_size": max_batch,
        "workspace_mb": workspace_mb,
        "input_shape": [max_batch, 3, 640, 640],
        "output_shape": [max_batch, 84, 8400],
        "num_layers": 225,
        "trt_version": "8.5.2",
        "cuda_version": "11.4",
        "device": "Jetson Nano (simulated)",
    }).encode()
    
    with open(engine_path, "wb") as f:
        f.write(header)
        f.write(len(meta).to_bytes(4, "little"))
        f.write(meta)
        # Pad to realistic size
        remaining = engine_size_mb * 1024 * 1024 - len(header) - 4 - len(meta)
        if remaining > 0:
            f.write(b"\x00" * remaining)
    
    actual_size = os.path.getsize(engine_path) / (1024 * 1024)
    logger.info(f"\n[SIMULATION] Engine built successfully")
    logger.info(f"  Engine size: {actual_size:.1f} MB")
    logger.info(f"  Precision:   {precision.upper()}")
    logger.info(f"  Layers:      225 (optimized)")
    logger.info(f"  Device:      Jetson Nano (simulated)")


if __name__ == "__main__":
    build_engine()
