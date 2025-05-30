import os                         # For environment variable and path check
import torch                      # For tensor operations and model inference
from PIL import Image             # For converting tensors to images
from transformers import ViTFeatureExtractor, ViTForImageClassification


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
        from transformers import ViTFeatureExtractor, ViTForImageClassification
        import os

        model_path = os.getenv("LOCAL_AGE_MODEL_PATH", "")
        if model_path and os.path.exists(model_path):
            self.model = ViTForImageClassification.from_pretrained(model_path).eval().to("cuda")
            self.processor = ViTFeatureExtractor.from_pretrained(model_path)
        else:
            self.model = ViTForImageClassification.from_pretrained("nateraw/vit-age-classifier").eval().to("cuda")
            self.processor = ViTFeatureExtractor.from_pretrained("nateraw/vit-age-classifier")

    def check_age(self, image, gate_enabled, use_local_model):
        import torch
        from PIL import Image
        from transformers import ViTFeatureExtractor, ViTForImageClassification
        import os

        # Reload local model if toggled at runtime
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
        return (is_underage, predicted_age, confidence, status, gate_output)
        
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

        # Normalize string values
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
    "UnderageFilterNode": UnderageFilterNode,
    "MultitypeGateNode": MultiTypeGateNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UnderageFilterNode": "Underage Filter",
    "MultitypeGateNode": "Flexible Gate",
}
