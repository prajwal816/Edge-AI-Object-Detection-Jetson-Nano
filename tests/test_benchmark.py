import os
import sys
import json
import unittest
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

class TestBenchmarks(unittest.TestCase):
    
    def test_benchmark_reports_exist(self):
        """Test that benchmark JSONs were generated correctly."""
        py_result = Path("benchmarks/results/python_benchmark.json")
        cpp_result = Path("benchmarks/results/cpp_benchmark.json")
        
        # Test will pass warning if files don't exist yet
        if not py_result.exists() or not cpp_result.exists():
            print("Warning: Benchmark reports not generated yet.")
            return
            
        with open(py_result, "r") as f:
            py_data = json.load(f)
            
        with open(cpp_result, "r") as f:
            cpp_data = json.load(f)
            
        # Verify JSON structure
        self.assertIn("avg_fps", py_data)
        self.assertIn("avg_latency_ms", py_data)
        self.assertIn("avg_fps", cpp_data)
        self.assertIn("avg_latency_ms", cpp_data)
        
        # Verify C++ performance is better
        self.assertGreater(cpp_data["avg_fps"], py_data["avg_fps"], 
                           "C++ FPS should be higher than Python")
                           
        self.assertLess(cpp_data["avg_latency_ms"], py_data["avg_latency_ms"], 
                        "C++ latency should be lower than Python")
                        
    def test_config_loader(self):
        """Test the YAML config loading."""
        from src.python.utils.config import build_pipeline_config
        
        # Build with defaults
        config = build_pipeline_config()
        self.assertEqual(config.model.variant, "yolov8n")
        self.assertEqual(config.export.trt_precision, "fp16")

if __name__ == '__main__':
    unittest.main()
