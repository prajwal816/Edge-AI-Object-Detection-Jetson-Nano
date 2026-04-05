"""
Edge AI Object Detection System
Structured Logging Utility
==========================
Provides colored console logging with file output and
performance metric tracking for the training/export pipeline.
"""

import logging
import sys
import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


class PerformanceTracker:
    """Tracks timing metrics for pipeline stages."""

    def __init__(self):
        self.timers: Dict[str, float] = {}
        self.metrics: Dict[str, list] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, name: str):
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        if name not in self._start_times:
            return 0.0
        elapsed = (time.perf_counter() - self._start_times[name]) * 1000  # ms
        self.timers[name] = elapsed
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(elapsed)
        del self._start_times[name]
        return elapsed

    def get_avg(self, name: str) -> float:
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])

    def get_summary(self) -> Dict[str, Any]:
        summary = {}
        for name, values in self.metrics.items():
            summary[name] = {
                "count": len(values),
                "avg_ms": sum(values) / len(values),
                "min_ms": min(values),
                "max_ms": max(values),
                "total_ms": sum(values),
            }
        return summary

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.get_summary(), f, indent=2)


def setup_logger(
    name: str = "edge_ai",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup a structured logger with colored console output and optional file logging.
    
    Args:
        name: Logger name
        level: Logging level
        log_dir: Directory for log files
        log_file: Specific log file name
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    logger.propagate = False

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if HAS_COLORLOG:
        console_format = colorlog.ColoredFormatter(
            "%(log_color)s[%(asctime)s] %(levelname)-8s%(reset)s "
            "%(blue)s%(name)s%(reset)s: %(message)s",
            datefmt="%H:%M:%S",
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )
    else:
        console_format = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler
    if log_dir or log_file:
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            if not log_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
            else:
                log_file = os.path.join(log_dir, log_file)

        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s [%(filename)s:%(lineno)d]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


# Global instances
_perf_tracker = PerformanceTracker()


def get_perf_tracker() -> PerformanceTracker:
    return _perf_tracker
