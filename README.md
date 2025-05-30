---

# ComfyUI Underage Filter

A custom ComfyUI extension for filtering images and videos that may depict **underage content**, using state-of-the-art image classification models. This tool is designed to support responsible deployment of generative AI by enforcing ethical and legal content moderation.

## Features

* ✅ Classifies whether subjects in images/videos appear to be underage
* 📼 Supports both **image** and **video** inputs
* 🧠 Uses ViT-based models for age classification
* 🚫 Blocks unsafe content and optionally halts the workflow
* 🔁 Seamless integration with ComfyUI's workflow system
* 🧩 Built as a custom ComfyUI node, plug-and-play compatible

## Installation

Clone the repository or copy the node files into your ComfyUI custom node directory:

```bash
git clone https://github.com/your-username/ComfyUI-Underage-Filter.git
cd ComfyUI-Underage-Filter
```

Then, move the files to your ComfyUI custom nodes directory (if necessary):

```bash
cp -r ComfyUI-Underage-Filter /path/to/ComfyUI/custom_nodes/
```

Restart ComfyUI.

> Requires ComfyUI >= `2024.03` and Python >= `3.10`

## Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

Or if you're using a Docker build, make sure these dependencies are in your `Dockerfile`.

## How It Works

The node uses a ViT-based classifier (e.g., `nateraw/vit-age-classifier`) from Hugging Face to estimate the subject’s age from an image. If the subject appears underage (typically <18), the workflow can halt or flag the content.

### Node Types

#### `UnderageCheck (Image)`

* **Input:** `IMAGE`
* **Output:** `BOOLEAN` (`is_underage`), `STRING` (`predicted_age`)
* Optional: Can halt workflow execution or return error

#### `UnderageCheck (Video)`

* **Input:** `VIDEO_PATH`
* **Output:** `BOOLEAN` (`is_underage`), `STRING` (`predicted_age`)
* Samples frames from the video and applies the same classification model.

## Configuration

You can configure:

* Age threshold (default is 18)
* Confidence threshold
* Whether to block generation or just log

Options can be set directly in the node interface or by editing default values in the script.

## Models

By default, this node uses:

```
Model: nateraw/vit-age-classifier
Source: https://huggingface.co/nateraw/vit-age-classifier
```

You may replace it with any `AutoModelForImageClassification`-compatible model trained for age detection.

## License

MIT License © 2025

---

## Disclaimer

This tool **does not guarantee 100% accuracy**. It is intended as a **preventative safety measure**. Always perform additional reviews when necessary. The developers are not liable for misuse or misclassification.

---
