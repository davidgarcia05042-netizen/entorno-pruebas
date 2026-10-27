# Decisiones técnicas — Entorno de evaluación de modelos de estimación de pose humana

Reúne el contexto del entorno experimental, las decisiones de diseño con su
justificación, las limitaciones abiertas y las convenciones de trabajo. Cómo
correr los scripts está en el `README.md`.

## Contexto

Trabajo de grado (Universidad Antonio Nariño, Ingeniería de Sistemas) de Juan
David García Rubio y Camila Sandoval Ruiz, dirigido por Ph.D. David Alberto
Herrera Álvarez, en el marco del grupo de investigación LACSER/Bioingeniería.

**Objetivo del trabajo de grado:** desarrollar un servicio que compare
MediaPipe Pose con YOLOv8-Pose para estimación de pose humana, evaluando su
precisión bajo oclusión parcial e iluminación variable, con miras a
integrarse a una plataforma de telefisioterapia existente. Metodología SCRUM,
4 sprints, 12 semanas.

## Los tres repositorios de investigación

El 2026-09-14 el servicio real se separó a su propio repositorio.

- **[`entorno-pruebas`](https://github.com/davidgarcia05042-netizen/entorno-pruebas)**
  (este repositorio): entorno **experimental**. Compara MediaPipe Pose y
  YOLO-Pose contra el gold standard de laboratorio (mocap Maxtraq). Contiene
  los adaptadores de los modelos, el esquema unificado de keypoints, las
  métricas de precisión (RMSE como protocolo principal, MAE, MPJPE, PCK, tasa
  de detección), la simulación de oclusión e iluminación, y los scripts para
  procesar video y generar resultados y gráficas. **No expone ningún servicio
  ni API**: esa parte se eliminó de aquí a propósito, porque vive en el
  repositorio del servicio.
- **[`servicio-procesamiento-pose`](https://github.com/davidgarcia05042-netizen/servicio-procesamiento-pose)**:
  el servicio **real** que se integra con la plataforma de telefisioterapia
  (contrato webhook, cola asíncrona, despliegue en Render). Usa los modelos
  validados aquí (YOLO26-nano, MediaPipe Heavy, MediaPipe Lite), pero no
  reimplementa la lógica de comparación y evaluación, que vive solo en este
  repositorio.
- **[`pose-benchmark`](https://github.com/davidgarcia05042-netizen/pose-benchmark)**:
  entorno separado y deliberadamente mínimo, solo para medir tiempo, CPU,
  memoria y GPU de cada variante de cada modelo. Está separado para que la
  sobrecarga de un framework de servicio no contamine las mediciones de
  desempeño.

## Decisiones técnicas clave

- **Esquema unificado de keypoints:** MediaPipe (33 landmarks) y YOLOv8-Pose
  (17, formato COCO) no comparten esquema nativo. Internamente se usa COCO-17
  como vocabulario común. La plataforma de telefisioterapia, en cambio, usa
  los 33 índices de MediaPipe como convención (`LEFT_SHOULDER`=11, etc., en
  mayúsculas): cualquier salida hacia la plataforma debe usar esa convención,
  no COCO-17.
- **Umbral de error angular:** ≤5° aceptable, 5-10° moderado, >10° no
  aceptable. Fundamentado en Gajdosik & Bohannon (1987, *Physical Therapy*) y
  corroborado con revisiones sistemáticas recientes de captura de movimiento
  sin marcadores frente a captura con marcadores (constantes en
  `app/config.py`).
- **Oclusión:** diseño mixto. Oclusión **natural**, ya presente en las tomas
  del laboratorio, más oclusión **sintética** programada (máscaras sobre una
  región, `app/core/occlusion.py`).
- **Iluminación:** los videos del laboratorio se grabaron con luz constante.
  Como alternativa, ya implementada, se simula con OpenCV (gamma y brillo) en
  `app/core/illumination.py`. Una segunda tanda del laboratorio con luz real
  variable quedó sin confirmar.
- **Modelos activos:** YOLO26-nano, MediaPipe Heavy y MediaPipe Lite. YOLOv8
  se descartó tras el barrido inicial (ver README, "Decisiones ya tomadas").
- **El contrato con la plataforma de telefisioterapia** (webhook, hardware de
  despliegue, etc.) no es responsabilidad de este repositorio: se documenta en
  `docs/DECISIONES_TECNICAS.md` de `servicio-procesamiento-pose`. Aquí solo
  importa en cuanto fija la convención de keypoints que debería usar una
  exportación hacia la plataforma (33 índices de MediaPipe, no COCO-17).

## Limitaciones abiertas

Se citan como tales en el documento del trabajo de grado.

1. **Identidad anatómica exacta de los marcadores Maxtraq:** además de lo
   corroborado por correlación (vértice = marcador #2 en "TOMA FRONTAL";
   cadera, rodilla y tobillo = #1, #3 y #5 en "Marcha P01"), el
   laboratorio no ha confirmado directamente qué marcador corresponde a qué
   punto anatómico. Debe citarse como corroborado, no confirmado.
2. **Videos de "TOMA FRONTAL" sin correlación confiable** (pie derecho y un
   subconjunto de "pierna derecha"): quedan excluidos de la comparación
   cuantitativa.

## Estado del benchmarking registrado aquí

Corresponde a las primeras mediciones; el estado actual está en
`pose-benchmark`.

- YOLOv8 n/s/m/l/x se midió en CPU sin límite de hilos, lo que **no
  representa Render** (usaba ~700 % de CPU, y Render daba 2 núcleos, ≈ 200 %
  de techo).
- Se corrigió con `--cpu-threads 2` en `benchmark_yolov8.py`, para simular el
  límite real. También se corrigió la medición de GPU para que sea por PID del
  proceso (NVML vía `nvidia-ml-py`) y no de toda la tarjeta, que antes se
  contaminaba con otras aplicaciones del sistema.
- Pendiente en ese momento: correr con `--cpu-threads 2` y con
  `--device cuda` para tener el conjunto completo, y construir la fase 2
  (MediaPipe Lite/Full/Heavy, `model_complexity` 0/1/2).

## Convenciones del código

- Identificadores en inglés; docstrings y comentarios largos en español (el
  equipo sustenta en español).
- Entorno de desarrollo: Windows, PowerShell, VS Code, Python 3.12, venv por
  proyecto (`venv\Scripts\Activate.ps1`).
- Los scripts se corren como módulo: `python -m scripts.nombre_script`, no
  `python scripts\nombre_script.py`, por la resolución del paquete `app`.
- Todo cambio de código se verifica con `pytest tests/test_core.py -v` antes
  de entregarse.
