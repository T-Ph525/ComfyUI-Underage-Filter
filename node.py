class AgeCheckerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold_age": ("INT", {"default": 18, "min": 0, "max": 100}),
                "show_on_node": ("BOOLEAN", {"default": False}),
                "use_local_model": ("BOOLEAN", {"default": False}),
                "local_model_path": ("STRING", {"default": "/comfy/models/age-classifier"}),
            },
        }

    OUTPUT_NODE = True

    RETURN_TYPES = ("STRING", "FLOAT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("Predicted Age", "Confidence", "Is Underage", "Message")
    FUNCTION = "check_age"
    CATEGORY = "ImageProcessing/Classification"

    def __init__(self):
        self.model = None
        self.extractor = None

    def load_model(self, use_local, local_path):
        if self.model is not None and self.extractor is not None:
            return

        model_path = local_path if use_local else "nateraw/vit-age-classifier"

        self.extractor = ViTFeatureExtractor.from_pretrained(model_path)
        self.model = ViTForImageClassification.from_pretrained(model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()

    def predict_age(self, img_tensor):
        img = Image.fromarray(np.clip(255. * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)).convert("RGB")
        inputs = self.extractor(images=img, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]

        top_idx = torch.argmax(probs).item()
        confidence = float(probs[top_idx].item())
        label = self.model.config.id2label[top_idx]

        try:
            min_age = int(label.split('-')[0].replace('+', ''))
        except:
            min_age = 0

        return label, confidence, min_age

    def check_age(self, image, threshold_age, show_on_node, use_local_model, local_model_path):
        try:
            self.load_model(use_local_model, local_model_path)

            # Support batch (video frames) or single image
            frames = image if isinstance(image, list) else [image]

            underage_detected = False
            best_confidence = 0.0
            best_label = ""
            messages = []

            for idx, frame in enumerate(frames):
                label, confidence, min_age = self.predict_age(frame)
                is_underage = min_age < threshold_age
                underage_detected |= is_underage

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_label = label

                messages.append(f"Frame {idx}: {label} ({confidence:.2%}) - Underage: {'Yes' if is_underage else 'No'}")

            summary = "\n".join(messages)
            output_ui = {"text": [summary]} if show_on_node else {}

            return {
                "result": (best_label, best_confidence, underage_detected, summary),
                "ui": output_ui
            }

        except Exception as e:
            error_msg = f"Error in AgeCheckerNode: {str(e)}"
            return {
                "result": ("Error", 0.0, False, error_msg),
                "ui": {"text": [error_msg]} if show_on_node else {}
            }
NODE_CLASS_MAPPINGS = {
    "AgeCheckerNode": AgeCheckerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AgeCheckerNode": "Age Checker",
}
