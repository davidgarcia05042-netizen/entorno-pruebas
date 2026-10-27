"""
Genera un video comparativo: esqueleto del modelo a la izquierda, curva de
ángulo (gold standard vs. predicción) revelándose en sincronía a la derecha
arriba, y métricas corridas (error instantáneo, MAE/RMSE acumulados) abajo
a la derecha.

Diseño de dos pasadas sobre el video (ver `render_comparison_video`):
la primera corre el modelo UNA vez (reusa `process_video`) y calcula la
serie completa de ángulo predicho; la segunda vuelve a leer los frames
originales para componer el panel ya con el rango de la gráfica fijo de
antemano (min/max de ambas curvas) -- así el eje no se reescala sobre la
marcha y el video queda visualmente estable. La curva del gold standard
también se conoce de antemano (viene del mocap, ya resampleada sobre la
grilla temporal del video por el caller) -- lo único que se calcula en
tiempo real, por así decirlo, es la del modelo.
"""

import math

import cv2
import numpy as np

from app.config import classify_angular_error
from app.core.ground_truth import compare_angle_series, resample_series
from app.core.keypoint_schema import Keypoint, UnifiedKeypoint
from app.core.metrics import compute_joint_angles
from app.core.video_io import open_video
from app.core.illumination import IlluminationLevel
from app.core.occlusion import OcclusionRegion
from app.core.video_processor import apply_conditions, process_video
from app.core.visualization import draw_pose_on_frame
from app.models.base import PoseEstimator
from app.schemas.pose_result import ModelName

COLOR_GT = (90, 90, 90)  # BGR -- gris oscuro, neutral (no pertenece a ningún modelo)
COLOR_YOLO = (235, 99, 37)  # BGR de #2563eb -- misma convención que evaluate_model_variants.py
COLOR_MEDIAPIPE = (38, 38, 220)  # BGR de #dc2626
COLOR_CURSOR = (0, 0, 0)
COLOR_BG = (255, 255, 255)
COLOR_GRID = (225, 225, 225)
COLOR_TEXT = (30, 30, 30)

COLOR_ACCEPTABLE = (0, 150, 0)
COLOR_MODERATE = (0, 165, 255)
COLOR_NOT_ACCEPTABLE = (0, 0, 220)


def model_curve_color(model_label: str) -> tuple[int, int, int]:
    return COLOR_YOLO if model_label.startswith("yolo") else COLOR_MEDIAPIPE


def classification_color(error_deg: float) -> tuple[int, int, int]:
    if math.isnan(error_deg):
        return COLOR_TEXT
    label = classify_angular_error(error_deg)
    return {
        "aceptable": COLOR_ACCEPTABLE,
        "moderado": COLOR_MODERATE,
        "no_aceptable": COLOR_NOT_ACCEPTABLE,
    }[label]


def _series_to_points(series: list[float], up_to_idx: int, width: int, height: int, y_min: float, y_max: float) -> list[tuple[int, int, int]] | None:
    """
    Convierte los valores [0, up_to_idx] de una serie a coordenadas de
    píxel dentro del panel. Devuelve (idx, x, y) -- el índice original se
    conserva para que `_draw_series` pueda cortar la línea cuando dos
    puntos válidos consecutivos no corresponden a frames consecutivos
    (tramo NaN de más de 1 frame, ej. el modelo no detectó a la persona).
    Devuelve None si no hay ningún punto válido todavía.
    """
    n = len(series)
    if n <= 1:
        return None
    x_step = width / (n - 1)
    y_span = (y_max - y_min) or 1.0

    points: list[tuple[int, int, int]] = []
    for i in range(up_to_idx + 1):
        value = series[i]
        if math.isnan(value):
            continue
        x = int(i * x_step)
        y = int(height - ((value - y_min) / y_span) * height)
        points.append((i, x, y))
    return points or None


def draw_chart_panel(
    gt_series: list[float],
    pred_series: list[float],
    current_idx: int,
    panel_size: tuple[int, int],
    y_range: tuple[float, float],
    model_label: str,
    margin_left: int = 55,
    margin_bottom: int = 22,
    margin_top: int = 24,
) -> np.ndarray:
    """
    Dibuja el panel de la gráfica: dos curvas (gold standard y predicción)
    reveladas hasta `current_idx`, con un cursor vertical marcando el
    frame actual. `y_range` se calcula UNA vez para todo el video (ver
    docstring del módulo) para que el eje no salte de frame a frame.
    """
    width, height = panel_size
    canvas = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)

    plot_w = width - margin_left - 15
    plot_h = height - margin_top - margin_bottom
    y_min, y_max = y_range

    # Gridlines horizontales + etiquetas del eje Y (min, medio, max)
    for frac in (0.0, 0.5, 1.0):
        y_px = margin_top + int(plot_h * (1 - frac))
        cv2.line(canvas, (margin_left, y_px), (margin_left + plot_w, y_px), COLOR_GRID, 1, cv2.LINE_AA)
        value = y_min + frac * (y_max - y_min)
        cv2.putText(canvas, f"{value:.0f}", (5, y_px + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    def _draw_series(series: list[float], color: tuple[int, int, int]) -> None:
        points = _series_to_points(series, current_idx, plot_w, plot_h, y_min, y_max)
        if points is None:
            return
        shifted = [(idx, x + margin_left, y + margin_top) for idx, x, y in points]
        for (idx_a, xa, ya), (idx_b, xb, yb) in zip(shifted, shifted[1:]):
            if idx_b - idx_a > 1:
                continue  # hueco NaN de más de 1 frame -- no conectar la línea a través de él
            cv2.line(canvas, (xa, ya), (xb, yb), color, 2, cv2.LINE_AA)
        _, x_last, y_last = shifted[-1]
        cv2.circle(canvas, (x_last, y_last), 4, color, thickness=-1)

    _draw_series(gt_series, COLOR_GT)
    _draw_series(pred_series, model_curve_color(model_label))

    # Cursor vertical del frame actual
    n = len(gt_series)
    if n > 1:
        x_cursor = margin_left + int((current_idx / (n - 1)) * plot_w)
        cv2.line(canvas, (x_cursor, margin_top), (x_cursor, margin_top + plot_h), COLOR_CURSOR, 1, cv2.LINE_AA)

    # Leyenda
    cv2.line(canvas, (margin_left, height - 8), (margin_left + 18, height - 8), COLOR_GT, 2)
    cv2.putText(canvas, "Gold standard", (margin_left + 24, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
    legend_x2 = margin_left + 175
    cv2.line(canvas, (legend_x2, height - 8), (legend_x2 + 18, height - 8), model_curve_color(model_label), 2)
    cv2.putText(canvas, model_label, (legend_x2 + 24, height - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), COLOR_GRID, 1)
    return canvas


def draw_metrics_panel(
    gt_current: float,
    pred_current: float,
    running_mae: float,
    running_rmse: float,
    joint_label: str,
    panel_size: tuple[int, int],
) -> np.ndarray:
    """
    Panel de texto con las métricas del Paso 2 acordadas: ángulo actual
    (gold standard vs. predicción), error instantáneo + clasificación, y
    MAE/RMSE acumulados desde el frame 0 hasta el actual. `running_mae`/
    `running_rmse` ya vienen calculados por el caller (ver
    `render_comparison_video`) -- este panel solo dibuja.
    """
    width, height = panel_size
    canvas = np.full((height, width, 3), COLOR_BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), COLOR_GRID, 1)

    error_current = abs(pred_current - gt_current) if not (math.isnan(pred_current) or math.isnan(gt_current)) else float("nan")
    classification = classify_angular_error(error_current) if not math.isnan(error_current) else "sin datos"

    # Espaciado proporcional al alto del panel -- 7 líneas de contenido más
    # los huecos entre secciones, para que siga cabiendo sin importar la
    # resolución final del video (el panel de métricas puede terminar
    # siendo bastante más bajo que en estas pruebas sintéticas).
    pad_x = 18
    top_pad = height * 0.08
    line_h = (height - 2 * top_pad) / 7.6
    scale = max(0.35, min(0.62, line_h * 0.021))
    y = int(top_pad + line_h * 0.6)

    def _line(label: str, value: str, color: tuple[int, int, int] = COLOR_TEXT) -> None:
        nonlocal y
        cv2.putText(canvas, label, (pad_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, COLOR_TEXT, 1, cv2.LINE_AA)
        (label_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(canvas, value, (pad_x + label_w + 10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += int(line_h)

    cv2.putText(canvas, f"Articulacion: {joint_label}", (pad_x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, COLOR_TEXT, 1, cv2.LINE_AA)
    y += int(line_h * 1.15)

    gt_txt = f"{gt_current:.1f} deg" if not math.isnan(gt_current) else "sin dato"
    pred_txt = f"{pred_current:.1f} deg" if not math.isnan(pred_current) else "sin dato"
    _line("Gold standard:", gt_txt)
    _line("Prediccion:", pred_txt, color=(0, 0, 0))

    error_txt = f"{error_current:.1f} deg" if not math.isnan(error_current) else "sin dato"
    _line("Error instantaneo:", error_txt, color=classification_color(error_current))
    _line("Clasificacion:", classification, color=classification_color(error_current))

    y += int(line_h * 0.2)
    mae_txt = f"{running_mae:.2f} deg" if not math.isnan(running_mae) else "sin dato"
    rmse_txt = f"{running_rmse:.2f} deg" if not math.isnan(running_rmse) else "sin dato"
    _line("MAE acumulado:", mae_txt)
    _line("RMSE acumulado:", rmse_txt)

    return canvas


def compose_frame(
    video_frame: np.ndarray,
    chart_panel: np.ndarray,
    metrics_panel: np.ndarray,
    title: str,
) -> np.ndarray:
    """
    Arma el canvas final: video (con esqueleto ya dibujado por el caller)
    a la izquierda, gráfica arriba a la derecha, métricas abajo a la
    derecha -- mismo layout del boceto de referencia. `chart_panel` y
    `metrics_panel` deben tener el mismo ancho entre sí; su alto puede
    diferir, el resultado se ajusta a `video_frame.shape[0]`.
    """
    video_h, video_w = video_frame.shape[:2]
    right_w = chart_panel.shape[1]

    right_column = np.full((video_h, right_w, 3), COLOR_BG, dtype=np.uint8)
    chart_h = chart_panel.shape[0]
    right_column[0:chart_h, :, :] = chart_panel
    metrics_resized_h = video_h - chart_h
    if metrics_resized_h > 0:
        metrics_resized = cv2.resize(metrics_panel, (right_w, metrics_resized_h))
        right_column[chart_h:video_h, :, :] = metrics_resized

    # Barra de título semitransparente sobre la esquina superior izquierda del video
    annotated_video = video_frame.copy()
    overlay = annotated_video.copy()
    bar_h = 34
    cv2.rectangle(overlay, (0, 0), (video_w, bar_h), (0, 0, 0), thickness=-1)
    cv2.addWeighted(overlay, 0.55, annotated_video, 0.45, 0, dst=annotated_video)
    cv2.putText(annotated_video, title, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    canvas = np.zeros((video_h, video_w + right_w, 3), dtype=np.uint8)
    canvas[:, :video_w, :] = annotated_video
    canvas[:, video_w:, :] = right_column
    cv2.line(canvas, (video_w, 0), (video_w, video_h), (0, 0, 0), 2)
    return canvas


def render_comparison_video(
    video_path: str,
    estimator: PoseEstimator,
    model_name: ModelName,
    model_label: str,
    joint: str,
    gt_times_s: list[float],
    gt_angles_deg: list[float],
    offset_s: float,
    out_path: str,
    title: str,
    max_frames: int | None = None,
    illumination_level: IlluminationLevel | None = None,
    occlusion_region: OcclusionRegion | None = None,
    right_panel_width_fraction: float = 0.78,
    chart_height_fraction: float = 0.55,
) -> dict:
    """
    Genera el video comparativo completo. Dos pasadas sobre `video_path`
    (ver docstring del módulo):

    1. Corre el modelo UNA vez (`process_video`, sin escribir video) para
       obtener el ángulo predicho por frame, y resamplea el gold standard
       sobre la grilla temporal del video (usando `offset_s`, ya
       confirmado por el pipeline de sincronización existente) -- así el
       rango del eje Y de la gráfica se puede fijar de antemano.
    2. Vuelve a leer los frames, les aplica las mismas condiciones
       (iluminación/oclusión) para mostrar exactamente lo que vio el modelo,
       dibuja el esqueleto (mismos keypoints de la pasada 1, no se vuelve a
       inferir) y compone cada frame final con `compose_frame`.

    Devuelve el mismo dict que `compare_angle_series` (para loguear el
    resultado final igual que el resto del pipeline).
    """
    with estimator:
        result = process_video(
            video_path, "comparison_video", estimator, model_name,
            illumination_level=illumination_level, occlusion_region=occlusion_region, max_frames=max_frames,
        )

    video_times_s = [f.timestamp_ms / 1000 for f in result.frames]
    keypoints_per_frame: list[list[Keypoint]] = []
    pred_angles: list[float] = []
    for frame in result.frames:
        kpts = [
            Keypoint(name=UnifiedKeypoint(kp.name), x=kp.x, y=kp.y, confidence=kp.confidence, visible=kp.visible)
            for kp in frame.keypoints
        ]
        keypoints_per_frame.append(kpts)
        pred_angles.append(compute_joint_angles(kpts)[joint])

    shifted_video_times = [t - offset_s for t in video_times_s]
    gt_on_video_grid = resample_series(gt_times_s, gt_angles_deg, shifted_video_times)

    valid_values = [v for v in gt_on_video_grid + pred_angles if not math.isnan(v)]
    if valid_values:
        v_min, v_max = min(valid_values), max(valid_values)
        pad = max(10.0, (v_max - v_min) * 0.15)
        y_range = (v_min - pad, v_max + pad)
    else:
        y_range = (0.0, 180.0)

    cap = open_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    right_w = int(frame_w * right_panel_width_fraction)
    chart_h = int(frame_h * chart_height_fraction)
    metrics_h = frame_h - chart_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_w + right_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"No se pudo crear el video de salida en: {out_path}")

    sum_abs = 0.0
    sum_sq = 0.0
    count = 0
    n_frames = len(video_times_s)

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        frame = apply_conditions(frame, illumination_level, occlusion_region)
        annotated = draw_pose_on_frame(frame, keypoints_per_frame[i])

        gt_v = gt_on_video_grid[i]
        pred_v = pred_angles[i]
        if not (math.isnan(gt_v) or math.isnan(pred_v)):
            err = abs(pred_v - gt_v)
            sum_abs += err
            sum_sq += err**2
            count += 1
        running_mae = sum_abs / count if count else float("nan")
        running_rmse = math.sqrt(sum_sq / count) if count else float("nan")

        chart_panel = draw_chart_panel(gt_on_video_grid, pred_angles, i, (right_w, chart_h), y_range, model_label)
        metrics_panel = draw_metrics_panel(gt_v, pred_v, running_mae, running_rmse, joint, (right_w, metrics_h))
        final_frame = compose_frame(annotated, chart_panel, metrics_panel, title)
        writer.write(final_frame)

    cap.release()
    writer.release()

    return compare_angle_series(pred_angles, gt_on_video_grid)
