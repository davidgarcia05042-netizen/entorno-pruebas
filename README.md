# Entorno Experimental — Evaluación de Modelos de Estimación de Pose Humana

Trabajo de grado (Universidad Antonio Nariño, Ing. de Sistemas) que compara
**MediaPipe Pose** y **YOLO-Pose** para estimación de pose humana aplicada a
telerehabilitación, evaluando su precisión angular bajo oclusión parcial e
iluminación variable, contra datos de referencia de laboratorio (captura de
movimiento óptica, exportación Maxtraq).

Este repositorio es **exclusivamente el entorno de pruebas/comparación de
modelos** — no expone ningún servicio ni API. La integración con la
plataforma real de telerehabilitación (contrato REST, webhook de resultados,
autenticación entre servicios, etc.) está en
[`servicio-procesamiento-pose`](https://github.com/davidgarcia05042-netizen/servicio-procesamiento-pose).
Las decisiones de diseño y su justificación están en
[`docs/DECISIONES_TECNICAS.md`](docs/DECISIONES_TECNICAS.md).

**Repositorios relacionados:**

| Repositorio | Qué es |
|---|---|
| [`servicio-procesamiento-pose`](https://github.com/davidgarcia05042-netizen/servicio-procesamiento-pose) | Servicio que sirve el modelo elegido aquí a la plataforma |
| [`pose-benchmark`](https://github.com/davidgarcia05042-netizen/pose-benchmark) | Tiempo, CPU y memoria de cada modelo |
| [`telefisioterapia-backend`](https://github.com/davidgarcia05042-netizen/telefisioterapia-backend) | Backend de la plataforma de telefisioterapia |
| [`telefisioterapia-panel`](https://github.com/davidgarcia05042-netizen/telefisioterapia-panel) | Panel web del terapeuta |
| [`telefisioterapia-app`](https://github.com/davidgarcia05042-netizen/telefisioterapia-app) | App Android del paciente |

## Qué hay acá

- `app/models/`: wrappers de MediaPipe Pose y YOLO-Pose sobre un esquema
  unificado de keypoints (COCO-17).
- `app/core/`: parser del ground truth de laboratorio (Maxtraq),
  sincronización temporal por correlación, simulación de oclusión e
  iluminación, cálculo de métricas (MAE, RMSE, MPJPE, PCK) y persistencia de
  resultados por corrida.
- `scripts/`: los puntos de entrada reales del proyecto (ver abajo).
- `data/gold_standard/`: videos de laboratorio + su ground truth y los
  resultados de cada evaluación. **No se sube a git** (contiene video de
  pacientes/participantes y resultados derivados) — ver `.gitignore`.
- `resultados/`: keypoints crudos de la demo rápida sobre 5 videos de stock,
  con MediaPipe y YOLOv8 (10 JSON). No contienen video ni datos de
  participantes.

### Datos y seudónimos

**La ruta de los datos es configurable.** Los scripts principales
(`evaluate_model_variants`, `sync_video_to_ground_truth`,
`render_comparison_video`, `compare_runs`, `run_single_video`) reciben las
rutas por argumento (`--video`, `--maxtraq`, `--store-dir`, `--out`, …), así que
los datos pueden estar en cualquier carpeta; `data/gold_standard/` es solo la
convención usada en los ejemplos. Los `scripts/_diag_*.py` generan figuras
puntuales del documento y tienen la ruta fija en una constante al inicio de
cada archivo: se ajusta ahí.

Los conjuntos del laboratorio se identifican con seudónimos: el conjunto de
marcha es **P01** (carpeta `marcha_p01_2026-06-19`, archivo Maxtraq
`MARCHA_P01_19_06_2026_maxtraq_1.TXT`) y las tomas frontales usan sus códigos
de toma (`2EJ1`, `5EJ6`, …). Para correr los ejemplos con los datos originales,
la carpeta local debe llamarse con el seudónimo.
- Las mediciones de tiempo/CPU/RAM de cada variante de modelo están en
  [`pose-benchmark`](https://github.com/davidgarcia05042-netizen/pose-benchmark),
  un entorno aparte y deliberadamente mínimo (sin el overhead de este
  paquete).

## Cómo correr esto

```bash
python -m venv venv
source venv/bin/activate  # en Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Correr los tests

```bash
pytest tests/test_core.py -v
```

### Demo rápida sobre un video local (sin gold standard)

Corre ambos modelos sobre un video y guarda los keypoints crudos en JSON,
sin comparar contra ningún ground truth:

```bash
python -m scripts.run_single_video --video ruta/al/video.mp4 --out resultados/
```

Con oclusión programada (auto-centrada en la rodilla real detectada) e
iluminación oscura simulada:

```bash
python -m scripts.run_single_video --video ruta/al/video.mp4 --out resultados/ \
    --occlude-knee auto --illumination dark
```

### Evaluación completa contra el gold standard de laboratorio

`scripts/evaluate_model_variants.py` es el script principal: corre los
modelos elegidos sobre un video, compara el ángulo de la articulación
indicada contra el ground truth (Maxtraq) ya sincronizado, y guarda
métricas + gráficas en `data/gold_standard/<video>/<modelo>/<condición>/<timestamp>/`.

```bash
python -m scripts.evaluate_model_variants \
    --video "data/gold_standard/<video>/<video>.mp4" \
    --maxtraq "data/gold_standard/<video>/<video>.TXT" \
    --offset -0.16 --joint right_knee --gt-marker-ids 1 2 3 \
    --models yolo26n mediapipe_heavy mediapipe_lite \
    --store-dir "data/gold_standard/<video>"
```

Con `--occlude-joint right_knee` (oclusión sintética auto-centrada en esa
articulación) y/o `--illumination dark` se generan las condiciones
`occlusion`/`illumination` en su propia carpeta, sin mezclarse con la
corrida base (`no_occlusion`).

`scripts/build_model_comparison.py` unifica, para un modelo dentro de un
video, las 3 condiciones ya guardadas en una sola gráfica comparativa
(`<video>/<modelo>/comparison/<timestamp>/`), y acepta `--metric
{mean_error_deg,rmse_deg}` para elegir qué métrica graficar.

## Decisiones ya tomadas (y por qué)

- **Esquema unificado de 17 puntos (formato COCO)**: MediaPipe (33
  landmarks) y YOLO-Pose (17, COCO) no comparten esquema nativo; se usa
  COCO-17 como vocabulario común interno. Ver `app/core/keypoint_schema.py`.
- **Modelos activos**: YOLO26-nano, MediaPipe Heavy y MediaPipe Lite.
  YOLOv8/YOLO11 (y sus variantes `_hiconf`) quedaron descartados tras el
  barrido inicial — MediaPipe los superó consistentemente y YOLO26-nano fue
  el único YOLO competitivo. El código de esas variantes se conserva en
  `evaluate_model_variants.py` por valor histórico, pero no se generan
  resultados nuevos con ellas salvo pedido explícito.
- **Umbrales de error angular**: ≤5° aceptable, 5-10° moderado, >10° no
  aceptable (`app/config.py`), basado en Gajdosik & Bohannon (1987) y
  revisiones de mocap markerless vs. marker-based. La métrica histórica del
  proyecto es el error absoluto medio (MAE); RMSE se añadió como métrica
  secundaria (más adoptada en la literatura de validación de pose
  estimation) sin reemplazar el criterio de clasificación, que sigue
  basado en MAE.
- **Simulación de iluminación con OpenCV**: mientras se confirma si el
  laboratorio puede grabar con luz real variable. Ver la advertencia de
  limitación metodológica en `app/core/illumination.py` — debe citarse como
  tal en el documento.
- **Sincronización video↔mocap por correlación**: los videos de laboratorio
  no tienen offset de arranque garantizado contra el archivo Maxtraq; se
  determina por búsqueda del desfase que maximiza la correlación de Pearson
  entre la señal de ángulo del gold standard y la del modelo
  (`app/core/ground_truth.py::find_best_time_offset`).

## Limitaciones abiertas

- Identidad anatómica de los marcadores Maxtraq más allá de lo ya
  confirmado (vértice = marcador #2 en las tomas de "TOMA FRONTAL", cadera/
  rodilla/tobillo = #1/#3/#5 en "Marcha P01") no está documentada por
  el laboratorio — se corroboró por correlación contra nuestro propio
  modelo, no por confirmación directa del laboratorio. Debe citarse así en
  el documento.
- Algunos videos de "TOMA FRONTAL" (pie derecho, y un subconjunto de
  "pierna derecha") no alcanzan correlación confiable con ningún candidato
  de articulación probado — quedan excluidos de la comparación cuantitativa
  por ahora, pendientes de una solución (mejor toma, o aceptar la
  limitación metodológica de cámara frontal para movimiento en profundidad).
