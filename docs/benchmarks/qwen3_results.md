---
title: Qwen 3
parent: Benchmarks
nav_order: 2
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance of Qwen 3 on NPU with FastFlowLM (FLM).

> **Note:** 
- Results are based on FastFlowLM v0.9.19.  
- Under FLM's default NPU power mode (Performance)    
- Test system spec: AMD Ryzen™ AI 7 350 (Krakan Point) with 32 GB DRAM.   
- Newer versions may deliver improved performance.   

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **Hardware**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |**Model**  |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|---------|
| **Qwen 3 0.6B**  | NPU (FLM)    | 53.7 |	47.3 |	38.7 |	28.7 |	18.9 |	11.2|**Qwen 3 0.6B**|
| **Qwen 3 1.7B**  | NPU (FLM)    | 28.4 |	26.8 |	23.9 |	19.6 |	14.5 |	9.5|**Qwen 3 1.7B**  |
| **Qwen 3 4B**    | NPU (FLM)    | 14.3 |	13.7 |	12.7 |	11.1 |	8.8 |	6.3|**Qwen 3 4B**    | 
| **Qwen 3 8B**    | NPU (FLM)    | 8.3 |	8.2 |	7.7 |	7.2 |	6.1 |	4.8|**Qwen 3 8B**    |

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **Hardware**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |**Model**  |
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|---------|
| **Qwen 3 0.6B**  | NPU (FLM)    | 1280 |	1356 |	1128 |	779 |	444 |	236 |**Qwen 3 0.6B**|
| **Qwen 3 1.7B**  | NPU (FLM)    | 939 |	1029 |	916 |	674 |	408 |	225|**Qwen 3 1.7B**  |
| **Qwen 3 4B**    | NPU (FLM)    | 509 |	574 |	543 |	435 |	282 |	164|**Qwen 3 4B**    | 
| **Qwen 3 8B**    | NPU (FLM)    | 357 |	420 |	408 |	345 |	243 |	150|**Qwen 3 8B**    |