# Stable Diffusion WebUI Forge - Neo

## Project Overview
**Stable Diffusion WebUI Forge - Neo** is an optimized and performance-focused fork of the [Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) (A1111). It serves as a continuation of the "latest" version of Forge, integrating specialized backends, memory management optimizations, and support for modern model architectures (e.g., Flux, Anima, Wan 2.2, Qwen-Image).

The project aims to provide a faster, more resource-efficient platform for image generation while maintaining compatibility with most base features of the original WebUI.

### Main Technologies
- **Python:** 3.13.12 (recommended)
- **UI Framework:** Gradio 4.40.0
- **Deep Learning:** PyTorch (2.5+), Transformers, Diffusers
- **Backend:** ComfyUI-inspired memory management and model patching
- **Optimization:** SageAttention, FlashAttention, xformers, Triton, bitsandbytes (optional)
- **Package Management:** Supports `uv` for accelerated installations

## Core Architecture
Forge Neo utilizes a hybrid architecture:
- **Frontend/UI:** Based on A1111's Gradio implementation.
- **Backend:** Rewritten logic in `backend/` and `comfy/` directories, leveraging `ModelPatcher` and optimized attention mechanisms.
- **Forge Enhancements:** Located in `modules_forge/`, handling the Forge Canvas, preset system, and specialized sampler/diffuser patches.
- **Extensions:** Built-in extensions are located in `extensions-builtin/`. Note that many legacy A1111 features/extensions have been removed to reduce bloat.

## Building and Running
The project is primarily launched via Windows batch scripts which initialize the environment and call the Python entry points.

### Key Commands
- **Launch:** Run `webui-user.bat` or `webui.bat`.
- **Installation:** Handled automatically on first launch via `launch.py`.
- **Environment Setup:** 
  ```bash
  uv venv venv --python 3.13 --seed
  ```
- **Help:** `python launch.py --help`

### Common Commandline Arguments
- `--uv`: Use the `uv` package manager for faster dependency installation.
- `--api`: Enable the FastAPI-based web API.
- `--xformers`, `--flash`, `--sage`: Install/enable specific attention optimizations.
- `--cuda-malloc`, `--cuda-stream`: Advanced memory allocation optimizations.
- `--forge-ref-comfy-home <path>`: Link models from an existing ComfyUI installation.

## Development Conventions
- **Backend Logic:** Core model handling and memory management should be investigated in `backend/` and `comfy/`.
- **UI Modifications:** Most UI logic resides in `modules/ui.py` and `javascript/`.
- **State Management:** `modules/shared.py` and `backend/shared.py` hold global application states (`cmd_opts`, `sd_model`, etc.).
- **Extensions:** New features are often implemented as built-in extensions in `extensions-builtin/`.

## Key Directories
- `backend/`: Optimized memory management and attention kernels.
- `comfy/`: Backend components derived from ComfyUI (model detection, patching).
- `modules/`: Core WebUI logic (A1111 base).
- `modules_forge/`: Unique Forge features (Presets, Canvas, Sampler patches).
- `extensions-builtin/`: Performance-critical or core-integrated extensions.
- `models/`: Default directory for checkpoints, LoRAs, VAEs, etc.
