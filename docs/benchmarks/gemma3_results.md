---
title: Gemma 3
parent: Benchmarks
nav_order: 3
---

## ⚡ Performance and Efficiency Benchmarks

This section reports the performance on NPU with FastFlowLM (FLM).

> **Note:** 
- Results are based on FastFlowLM v0.9.19.
- Under FLM's default NPU power mode (Performance)  
- Test system spec: AMD Ryzen™ AI 7 350 (Krakan Point) with 32 GB DRAM. 
- Newer versions may deliver improved performance. 

---

### 🚀 Decoding Speed (TPS, or Tokens per Second, starting @ different context lengths)

| **Model**        | **Hardware**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |**64k** | **128k** |**Model**|
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|---------:|---------:|---------|
| **Gemma 3 1B**  | NPU (FLM)    | 39.3	| 38.6 |	37.6 |	35.5 |	32.1 |	29.1|	OOC|	OOC| **Gemma 3 1B**  | 
| **Gemma 3 4B**  | NPU (FLM)    | 15.2	| 15.1 |	14.9 | 	14.5 |	13.8 |	12.8 |	11.8 |	10.2 |**Gemma 3 4B**  |

> OOC: Out Of Context Length  
> Each LLM has a maximum supported context window. For example, the gemma3:1b model supports up to 32k tokens.

---

### 🚀 Prefill Speed (TPS, or Tokens per Second, with different prompt lengths)

| **Model**        | **Hardware**       | **1k** | **2k** | **4k** | **8k** | **16k** | **32k** |**Model**|
|------------------|--------------------|--------:|--------:|--------:|--------:|---------:|---------:|---------|
| **Gemma 3 1B**   | NPU (FLM)    | 1004 |	1321|	1528 |	1645 |	1657 |	1596|**Gemma 3 1B**  |
| **Gemma 3 4B**   | NPU (FLM)    | 528 |	654 |	738 |	754 |	739 |	673|**Gemma 3 4B**  |


### 🚀 Prefill TTFT with image

| **Model**        | **Hardware**       | **Image** |
|------------------|--------------------|--------:|
| **Gemma 3 4B**   | NPU (FLM)    | 4.3|

> This test uses a short prompt: “Describe this image.”
