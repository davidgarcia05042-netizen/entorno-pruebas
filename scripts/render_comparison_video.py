"""
Genera un video comparativo (esqueleto + gráfica de ángulo en sincronía +
métricas corridas) para UN modelo sobre UN video, contra su gold standard
Maxtraq ya sincronizado (offset/marcadores ya confirmados por el pipeline
de sincronización existente, ver scripts/sync_video_to_ground_truth.py).

No forma parte del comparativo de precisión (evaluate_model_variants.py)
-- es una herramienta de inspección visual, pensada para corroborar de
primera mano el comportamiento frame a frame de un modelo (ej. el hallazgo
de que YOLO26-nano no trackea flexión profunda en TOMA FRONTAL).

Uso (Marcha P01, MediaPipe Heavy):
    python -m scripts.render_comparison_video \
        --video data/gold_standard/marcha_p01_2026-06-19/video.mp4 \
        --maxtraq data/gold_standard/marcha_p01_2026-06-19/maxtraq.TXT \
        --offset -0.02 --joint left_knee --model mediapipe_heavy \
        --out data/gold_standard/marcha_p01_2026-06-19/comparison_videos/mediapipe_heavy.mp4

Uso (TOMA FRONTAL/5EJ6, YOLO26-nano, rodilla derecha):
    python -m scripts.render_comparison_video \
        --video "data/gold_standard/TOMA FRONTAL/5EJ6/5EJ6.mp4" \
        --maxtraq "data/gold_standard/TOMA FRONTAL/5EJ6/5EJ6.TXT" \
        --offset -0.40 --joint right_knee --gt-marker-ids 1 2 3 --model yolo26n \
        --out "data/gold_standard/TOMA FRONTAL/5EJ6/comparison_videos/yolo26n.mp4"

Condiciones experimentales (mismas que evaluate_model_variants.py; el video
muestra el frame tal como lo vio el modelo): agregar
`--occlude-joint right_knee` y/o `--illumination dark`.
"""

import argparse
from pathlib import Path

from app.core.comparison_video import render_comparison_video
from app.core.ground_truth import marker_angle_series_deg, parse_maxtraq_txt
from app.core.illumination import IlluminationLevel
from app.core.keypoint_schema import JOINT_ANGLE_TRIPLETS, UnifiedKeypoint
from app.core.occlusion import detect_reference_joint_position, get_first_frame_shape, region_around_point
from app.models.mediapipe_pose import MediaPipePoseEstimator
from app.schemas.pose_result import ModelName
from scripts.evaluate_model_variants import MEDIAPIPE_VARIANTS, YOLO_VARIANTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True)
    parser.add_argument("--maxtraq", required=True)
    parser.add_argument("--offset", type=float, required=True, help="Offset de sincronización ya confirmado (segundos)")
    parser.add_argument("--joint", required=True, choices=list(JOINT_ANGLE_TRIPLETS))
    parser.add_argument(
        "--gt-marker-ids", type=int, nargs=3, default=[1, 3, 5], metavar=("PROXIMAL", "VERTEX", "DISTAL"),
        help="Default (1,3,5) = convención Marcha P01. TOMA FRONTAL usa (1,2,3).",
    )
    parser.add_argument("--model", required=True, choices=list(YOLO_VARIANTS) + list(MEDIAPIPE_VARIANTS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--title", default=None, help="Título mostrado en el video (default: '<video> -- <modelo> -- <condición>')")
    parser.add_argument(
        "--occlude-joint", choices=[k.value for k in UnifiedKeypoint], default=None,
        help="Oclusión sintética auto-centrada en esta articulación, igual que evaluate_model_variants.py",
    )
    parser.add_argument("--occlude-radius-px", type=int, default=60)
    parser.add_argument("--illumination", choices=[level.value for level in IlluminationLevel], default=None)
    args = parser.parse_args()

    occlusion_region = None
    if args.occlude_joint:
        center_x, center_y = detect_reference_joint_position(args.video, UnifiedKeypoint(args.occlude_joint))
        occlusion_region = region_around_point(
            get_first_frame_shape(args.video), center_x=center_x, center_y=center_y, radius_px=args.occlude_radius_px
        )
    illumination_level = IlluminationLevel(args.illumination) if args.illumination else None

    condition_parts = []
    if args.occlude_joint:
        condition_parts.append(f"oclusion {args.occlude_joint}")
    if args.illumination:
        condition_parts.append(f"iluminacion {args.illumination}")
    condition = " + ".join(condition_parts) or "sin oclusion"

    if args.model in YOLO_VARIANTS:
        estimator = YOLO_VARIANTS[args.model]()
        model_name = ModelName.YOLOV8
    else:
        estimator = MediaPipePoseEstimator(model_complexity=MEDIAPIPE_VARIANTS[args.model])
        model_name = ModelName.MEDIAPIPE

    recording = parse_maxtraq_txt(args.maxtraq)
    gt_angles = marker_angle_series_deg(recording, tuple(args.gt_marker_ids))

    video_label = Path(args.video).stem
    title = args.title or f"{video_label} -- {args.model} -- {condition}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Renderizando {args.model} sobre {args.video} (articulación={args.joint}, condición={condition})...")
    report = render_comparison_video(
        video_path=args.video,
        estimator=estimator,
        model_name=model_name,
        model_label=args.model,
        joint=args.joint,
        gt_times_s=recording.times_s,
        gt_angles_deg=gt_angles,
        offset_s=args.offset,
        out_path=str(out_path),
        title=title,
        max_frames=args.max_frames,
        illumination_level=illumination_level,
        occlusion_region=occlusion_region,
    )
    print(f"Listo: {out_path}")
    print(f"  MAE={report['mean_error_deg']}°  RMSE={report['rmse_deg']}°  clasificación={report.get('classification')}")


if __name__ == "__main__":
    main()
