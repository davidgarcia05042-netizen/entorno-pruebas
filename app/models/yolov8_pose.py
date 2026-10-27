"""
Wrapper de YOLO-Pose (librería Ultralytics).

Sirve para cualquier familia de pesos pose de Ultralytics (YOLOv8, YOLO11,
YOLO26): todas se cargan con la misma clase `YOLO` y producen el esquema
COCO-17. El modelo activo del proyecto es YOLO26-nano (default); YOLOv8 y
YOLO11 se descartaron tras el barrido inicial (ver historial de git para
reproducirlo). El nombre de la clase se conserva por el valor
`ModelName.YOLOV8` ya guardado en los resultados.

Inferencia con los umbrales por defecto de Ultralytics (conf=0.25,
iou=0.7): subirlos (conf=0.80, iou=0.45) se probó sin efecto alguno, porque
con una sola persona en cuadro no hay detecciones débiles que filtrar.
"""

import numpy as np

from app.core.keypoint_schema import COCO_ORDER, Keypoint
from app.models.base import PoseEstimator


class YoloV8PoseEstimator(PoseEstimator):
    name = "yolov8_pose"

    def __init__(self, weights: str = "yolo26n-pose.pt", confidence_threshold: float = 0.5):
        """`confidence_threshold` solo marca `visible` en un keypoint ya calculado; no filtra detecciones."""
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._confidence_threshold = confidence_threshold

    def predict(self, frame: np.ndarray) -> list[Keypoint]:
        results = self._model(frame, verbose=False)[0]

        keypoints: list[Keypoint] = []

        if results.keypoints is None or len(results.keypoints.data) == 0:
            for name in COCO_ORDER:
                keypoints.append(Keypoint(name=name, x=0.0, y=0.0, confidence=0.0, visible=False))
            return keypoints

        # Si hay varias personas detectadas, se toma la de mayor confianza
        # de bounding box (asumiendo un solo paciente en cuadro, como es
        # el caso esperado en telefisioterapia).
        person_idx = 0
        if results.boxes is not None and len(results.boxes.conf) > 1:
            person_idx = int(results.boxes.conf.argmax())

        kpts_xy = results.keypoints.xy[person_idx].cpu().numpy()  # (17, 2)
        kpts_conf = results.keypoints.conf[person_idx].cpu().numpy()  # (17,)

        for i, name in enumerate(COCO_ORDER):
            x, y = kpts_xy[i]
            conf = float(kpts_conf[i])
            keypoints.append(
                Keypoint(
                    name=name,
                    x=float(x),
                    y=float(y),
                    confidence=conf,
                    visible=conf >= self._confidence_threshold,
                )
            )

        return keypoints

    def close(self) -> None:
        # Ultralytics no requiere liberar recursos explícitamente
        pass
