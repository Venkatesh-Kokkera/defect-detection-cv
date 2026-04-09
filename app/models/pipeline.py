import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Image

class DefectDetectionPipeline:
    """
    Two-stage defect detection pipeline:
    Stage 1: YOLOv8 for defect region detection
    Stage 2: EfficientNet-B4 for defect classification
    """

    DEFECT_CLASSES = [
        "scratch",
        "crack",
        "dent",
        "contamination",
        "corrosion",
        "pass"
    ]

    def __init__(
        self,
        yolo_weights: str = "weights/yolov8_defect.pt",
        efficientnet_weights: str = "weights/efficientnet_defect.pt"
    ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Running on: {self.device}")

        # Load YOLOv8
        self.yolo = YOLO(yolo_weights)

        # Load EfficientNet-B4
        self.classifier = models.efficientnet_b4(pretrained=False)
        self.classifier.classifier[1] = torch.nn.Linear(
            self.classifier.classifier[1].in_features,
            len(self.DEFECT_CLASSES)
        )
        self.classifier = self.classifier.to(self.device)
        self.classifier.eval()

        # Image transforms for classifier
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def detect_regions(self, image: np.ndarray) -> list:
        """Stage 1: YOLOv8 detection."""
        results = self.yolo(image)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                boxes.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })
        return boxes

    def classify_region(self, roi: np.ndarray) -> dict:
        """Stage 2: EfficientNet classification."""
        pil_img = Image.fromarray(
            cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        )
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.classifier(tensor)
            probs = torch.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, 1)

        return {
            "defect_type": self.DEFECT_CLASSES[pred.item()],
            "confidence": round(conf.item(), 4)
        }

    def run(self, image: np.ndarray) -> dict:
        """Run full two-stage pipeline."""
        # Stage 1: Detect regions
        boxes = self.detect_regions(image)

        if not boxes:
            return {
                "defect_found": False,
                "defect_type": "pass",
                "confidence": 0.99,
                "bbox": []
            }

        # Stage 2: Classify best region
        best = max(boxes, key=lambda x: x["confidence"])
        x1, y1, x2, y2 = best["bbox"]
        roi = image[y1:y2, x1:x2]

        classification = self.classify_region(roi)

        return {
            "defect_found": classification["defect_type"] != "pass",
            "defect_type": classification["defect_type"],
            "confidence": classification["confidence"],
            "bbox": best["bbox"]
        }
