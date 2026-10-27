"""Modelos de datos (Pydantic) compartidos por toda la aplicación."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class ModelName(str, Enum):
    MEDIAPIPE = "mediapipe_pose"
    YOLOV8 = "yolov8_pose"


class KeypointResult(BaseModel):
    """Un punto anatómico detectado, ya en el esquema unificado (ver keypoint_schema.py)."""

    name: str
    x: float = Field(..., description="Coordenada horizontal en píxeles")
    y: float = Field(..., description="Coordenada vertical en píxeles")
    confidence: float = Field(..., ge=0.0, le=1.0)
    visible: bool = True


class FrameResult(BaseModel):
    """Resultado de un modelo de pose sobre un único frame de video."""

    frame_index: int
    timestamp_ms: float
    keypoints: list[KeypointResult]
    inference_latency_ms: float
    illumination_lux_estimate: Optional[float] = None
    occlusion_applied: Optional[str] = None  # e.g. "none" | "natural" | "synthetic:left_knee"


class VideoAnalysisResult(BaseModel):
    """Resultado consolidado del procesamiento completo de un video con un modelo."""

    model_config = ConfigDict(protected_namespaces=())  # permite el campo `model_name` sin advertencia

    video_id: str
    model_name: ModelName
    fps: float
    total_frames: int
    frames: list[FrameResult]
    mean_latency_ms: float
