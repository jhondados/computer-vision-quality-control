"""AI visual quality control inspector."""
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Tuple
from pathlib import Path

class VisualQCInspector:
    DEFECT_CLASSES = ["scratch", "dent", "discoloration", "contamination",
                      "missing_component", "wrong_orientation", "dimension_error"]

    def __init__(self, model_path: str = "qc_yolov9.pt", conf_threshold: float = 0.85):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.inspection_count = 0
        self.defect_count = 0

    def inspect(self, image: np.ndarray) -> Dict:
        """Inspect a single part image for defects."""
        results = self.model.predict(image, conf=self.conf_threshold, verbose=False)[0]
        defects = []
        for box in results.boxes:
            defect = {"class": self.DEFECT_CLASSES[int(box.cls[0])] if int(box.cls[0]) < len(self.DEFECT_CLASSES) else "unknown",
                      "confidence": float(box.conf[0]),
                      "bbox": box.xyxy[0].tolist(),
                      "severity": "critical" if float(box.conf[0]) > 0.95 else "warning"}
            defects.append(defect)
        self.inspection_count += 1
        if defects: self.defect_count += 1
        decision = "REJECT" if any(d["severity"] == "critical" for d in defects) else \
                   "REVIEW" if defects else "PASS"
        return {"decision": decision, "defects": defects, "defect_count": len(defects),
                "inspection_id": self.inspection_count}

    def draw_annotations(self, image: np.ndarray, inspection_result: Dict) -> np.ndarray:
        colors = {"PASS": (0, 255, 0), "REVIEW": (0, 165, 255), "REJECT": (0, 0, 255)}
        annotated = image.copy()
        for defect in inspection_result["defects"]:
            x1, y1, x2, y2 = [int(v) for v in defect["bbox"]]
            color = (0, 0, 255) if defect["severity"] == "critical" else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{defect['class']} {defect['confidence']:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        decision = inspection_result["decision"]
        cv2.putText(annotated, decision, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, colors[decision], 3)
        return annotated
