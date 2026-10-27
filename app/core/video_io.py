"""
Apertura de videos con la orientación correcta.

Los celulares graban en vertical guardando los píxeles "acostados" más un
metadato de rotación (ej. rotation=90) que los reproductores aplican al
mostrar. OpenCV no lo aplica por defecto: sin esto, los modelos reciben a
la persona girada 90°. MediaPipe (BlazePose) compensa internamente porque
alinea la rotación del cuerpo antes de ubicar los puntos; YOLO-Pose no, y
localiza peor (medido en TOMA FRONTAL/5EJ6: MAE 51.6° -> 44.1° al corregir,
MediaPipe Heavy 8.03° -> 8.04°). Todo video del pipeline debe abrirse aquí.
"""

import cv2


def open_video(video_path: str) -> cv2.VideoCapture:
    """Abre `video_path` aplicando la rotación declarada en sus metadatos."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"No se pudo abrir el video: {video_path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap
