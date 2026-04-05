# Implementation Notes: Environment & Llama 3.1 8B Integration

This document outlines the changes made to the Semantic Energy project to successfully enable GPU (CUDA) support and integrate the **Llama 3.1 8B Instruct** model using 8-bit quantization on an NVIDIA RTX 3060 (12GB VRAM).

## 1. Environment and CUDA Fixes

Initially, the project fell back to running on the CPU because it was using **Python 3.14**, for which pre-compiled CUDA binaries of PyTorch are not yet available.

**Changes Made:**
- **Explicit Python Versioning:** Updated `setup.bat` to forcefully use `py -3.12`. Python 3.12 has excellent, stable support for PyTorch and CUDA 12.4.
- **Rebuilding Virtual Environment:** Deleted the corrupted `.venv` folder and rebuilt it cleanly using the new `setup.bat`, ensuring all dependencies and the correct GPU-accelerated PyTorch wheels were fetched.

## 2. Model Upgrades (Llama 3.1 8B)

The original Qwen 1.5B model lacked the reasoning depth required for high-quality semantic evaluation. We upgraded to **Meta's Llama-3.1-8B-Instruct**. 

Because a standard 8B model requires ~16GB of VRAM (exceeding the 12GB available on the RTX 3060), we implemented **8-bit Quantization**.

### Why 8-bit Quantization?
We chose 8-bit (via `bitsandbytes`) instead of 4-bit quantization because 8-bit preserves the original floating-point weights much more accurately. This is critical for the Semantic Energy algorithm, which relies on the exact mathematical values of the logprobs (`outputs.scores`) to compute uncertainty.

**Changes Made in `backend/engine.py`:**
- **`BitsAndBytesConfig` Integration:** Configured the `SemanticEngine` to dynamically load the model using `load_in_8bit=True`.
- **Memory Management:** Added `device_map="auto"` to allow the `transformers` library to seamlessly handle moving the 8-bit quantized layers onto the GPU VRAM. In our tests, this stabilized at ~9GB VRAM, safely within the 12GB limit.

## 3. Prompt Engineering Adaptations

Llama 3 has a very strict chat templating structure (expecting specific system/user roles and `<|eot_id|>` markers). 

**Changes Made in `backend/engine.py` (`semantic_analyse`):**
- Migrated from raw string concatenation to a structured `messages` list.
- Passed the structured list through `self.tokenizer.apply_chat_template()` to perfectly format the zero-shot verification prompt according to Llama 3's native training data format.
- Instructed the model to explicitly output `"Final Decision: Yes"` or `"Final Decision: No"` to align with the framework's clustering logic.

## 4. New Dependencies

To support the above changes, the following requirements were added to `requirements.txt`:
- `bitsandbytes>=0.43.0`: The core quantization library underlying `load_in_8bit`.
- `scipy>=1.14.0`: Required by `bitsandbytes` for certain mathematical operations.

---
*The project is fully ready for high-quality semantic hallucination detection on the GPU.*
