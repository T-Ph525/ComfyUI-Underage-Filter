import os
import torch
from PIL import Image
from transformers import ViTFeatureExtractor, ViTForImageClassification, pipeline
import torchvision.transforms as T


class AgeCheckerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "gate_enabled": ("BOOLEAN", {"default": True}),
                "use_local_model": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN", "INT", "FLOAT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("is_underage", "predicted_age", "confidence", "status", "gate_output")

    FUNCTION = "check_age"
    CATEGORY = "Moderation/Age"

    def __init__(self):
        model_path = os.getenv("LOCAL_AGE_MODEL_PATH", "")
        if model_path and os.path.exists(model_path):
            self.model = ViTForImageClassification.from_pretrained(model_path).eval().to("cuda")
            self.processor = ViTFeatureExtractor.from_pretrained(model_path)
        else:
            self.model = ViTForImageClassification.from_pretrained("nateraw/vit-age-classifier").eval().to("cuda")
            self.processor = ViTFeatureExtractor.from_pretrained("nateraw/vit-age-classifier")

    def check_age(self, image, gate_enabled, use_local_model):
        if use_local_model:
            model_path = os.getenv("LOCAL_AGE_MODEL_PATH", "")
            if model_path and os.path.exists(model_path):
                self.model = ViTForImageClassification.from_pretrained(model_path).eval().to("cuda")
                self.processor = ViTFeatureExtractor.from_pretrained(model_path)

        pil_image = Image.fromarray(image[0])
        inputs = self.processor(images=pil_image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_age = int(logits.argmax(-1).item())
            confidence = float(logits.softmax(-1).max().item())

        is_underage = predicted_age < 18
        gate_output = not is_underage if gate_enabled else True

        status = "Underage" if is_underage else "OK"

        if gate_enabled and is_underage:
            raise PermissionError(403, f"Blocked: Detected age {predicted_age} with confidence {confidence:.2f}.")

        return (is_underage, predicted_age, confidence, status, gate_output)


class UnderageFilterNode:
    def __init__(self):
        self.classifier = pipeline("image-classification", model="nateraw/vit-age-classifier")
        self.to_pil = T.ToPILImage()
        self.underage_labels = {"0-2", "3-9", "10-19"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "score": ("FLOAT", {
                    "default": 0.85,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "age_confidence_threshold"
                }),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    FUNCTION = "check_underage"
    CATEGORY = "Moderation/Detection"

    def check_underage(self, image, score):
        img_tensor = image[0]
        pil_img = self.to_pil(img_tensor.permute(2, 0, 1))
        results = self.classifier(pil_img)
        top = max(results, key=lambda r: r["score"])
        is_underage = top["label"] in self.underage_labels and top["score"] >= score
        return (is_underage,)


class MultiTypeGateNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (["BOOLEAN", "INT", "FLOAT", "STRING"],),
                "block_on": ("STRING", {
                    "default": "falsy",  # options: 'falsy', 'truthy', 'equal'
                }),
                "match_value": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "message": ("STRING", {
                    "default": "Blocked by moderation gate."
                })
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "evaluate"
    CATEGORY = "Moderation/Blockers"

    def evaluate(self, value, block_on, match_value, message):
        should_block = False

        if isinstance(value, str):
            value = value.strip()

        if block_on == "falsy":
            should_block = not bool(value)
        elif block_on == "truthy":
            should_block = bool(value)
        elif block_on == "equal":
            try:
                match_casted = type(value)(match_value)
            except Exception:
                match_casted = match_value
            should_block = value == match_casted

        if should_block:
            raise PermissionError(403, message)

        return ()


NODE_CLASS_MAPPINGS = {
    "AgeCheckerNode": AgeCheckerNode,
    "UnderageFilterNode": UnderageFilterNode,
    "MultiTypeGateNode": MultiTypeGateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AgeCheckerNode": "Age Checker",
    "UnderageFilterNode": "Underage Filter",
    "MultiTypeGateNode": "Flexible Gate",
}
