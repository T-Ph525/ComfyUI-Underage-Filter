# ComfyUI Underage Filter

![ComfyUI](https://img.shields.io/badge/ComfyUI-Node-blueviolet)
[![Comfy Registry](https://comfyregistry.org/api/badge/comfyui-underage-filter)](https://comfyregistry.org/node/comfyui-underage-filter)

This extension adds moderation capabilities to ComfyUI, enabling automatic detection and optional blocking of underage content in image-based workflows. It uses the `nateraw/vit-age-classifier` model from Hugging Face to estimate the subject's age group with configurable confidence.

---

## ✨ Features

- 🔍 **UnderageFilterNode**  
  - Detects if an image contains a subject estimated to be underage (`0-2`, `3-9`, `10-19`)  
  - Outputs a `BOOLEAN` result (True if underage)

- 🔒 **BooleanGateNode**  
  - Takes a `BOOLEAN` input  
  - If `True`, raises a `403 PermissionError`, halting workflow execution  
  - Useful for API-based image safety pipelines

---

## 🧩 Use Case

```mermaid
flowchart TD
    A[Image Input] --> B[Underage Filter Node]
    B --> C[Boolean Gate Node]
    C -- Not Underage --> D[Continue to Generation]
    C -- Underage --> E[403 Blocked]
