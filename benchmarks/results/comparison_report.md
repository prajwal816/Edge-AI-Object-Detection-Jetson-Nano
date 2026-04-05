# Edge AI Pipeline Comparison

Performance comparison between the baseline Python pipeline and the optimized C++ TensorRT pipeline.

## Results Table

| Metric | Python Baseline | C++ TensorRT | Improvement |
|--------|----------------------|-----------------------|-------------|
| **FPS** | 10.5 | 28.5 | **2.7x faster** |
| **Latency** | 95.2 ms | 35.1 ms | **63.1% reduction** |
| **GPU Memory** | 1250 MB | 256 MB | **79.5% reduction** |

## Stage Breakdown (C++ Pipeline)

| Stage | Latency (ms) | % of Total |
|-------|--------------|------------|
| capture | 2.1 | 6.0% |
| preprocess | 3.4 | 9.7% |
| inference | 28.1 | 80.1% |
| postprocess | 1.5 | 4.3% |
