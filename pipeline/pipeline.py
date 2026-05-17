"""
Pipeline de generación de pares contrafactuales fotorrealistas
TFG - Explainabilidad XAI con dataset fotorrealista
Universitat de les Illes Balears

Flujo:
  1. Descarga imágenes de COCO con coches (con bounding boxes anotados)
  2. SAM 2 genera una máscara precisa del coche usando el bbox como guía
  3. Inpainting de la región enmascarada (Replicate API o LaMa local)
  4. Se calculan las métricas PCP, MAPD y SSIM para cada par
  5. Se guardan los pares validados en output/

Entornos:
  - xai_env  (conda, Python 3.10): SAM 2 + Replicate + métricas
  - lama_env (conda, Python 3.9):  LaMa con refinamiento

Uso:
  # Con Replicate (ejecutar desde xai_env):
  export REPLICATE_API_TOKEN=tu_token
  python pipeline/pipeline.py --inpainting replicate

  # Con LaMa local (ejecutar desde xai_env, requiere lama_env instalado):
  python pipeline/pipeline.py --inpainting lama

Setup inicial (una sola vez):
  # Checkpoint SAM 2:
  mkdir -p checkpoints
  curl -L -o checkpoints/sam2.1_hiera_small.pt \\
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

  # Checkpoint LaMa (dentro de models/lama/):
  cd models/lama
  curl -L "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip" -o big-lama.zip
  unzip big-lama.zip && rm big-lama.zip

  # Anotaciones COCO (una sola vez):
  curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip annotations_trainval2017.zip annotations/instances_val2017.json
  mv annotations/instances_val2017.json instances_val2017.json
"""

import os
import io
import sys
import json
import time
import base64
import argparse
import tempfile
import subprocess
import requests
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim_metric

import torch

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

PROJECT_ROOT    = Path(__file__).parent.parent
COCO_IMAGES_DIR = PROJECT_ROOT / "data" / "coco_images"
ANN_PATH        = PROJECT_ROOT / "data" / "instances_val2017.json"
OUTPUT_DIR      = PROJECT_ROOT / "output"
METRICS_FILE    = OUTPUT_DIR / "metrics.json"

SAM2_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sam2.1_hiera_small.pt"
SAM2_CONFIG     = "configs/sam2.1/sam2.1_hiera_s.yaml"

LAMA_DIR        = PROJECT_ROOT / "models" / "lama"
LAMA_MODEL      = LAMA_DIR / "big-lama"
LAMA_PYTHON     = Path(os.environ.get(
    "LAMA_PYTHON",
    # ruta por defecto al Python del entorno lama_env
    Path.home() / "miniconda3" / "envs" / "lama_env" / "bin" / "python3"
))

N_IMAGES        = 15     # imágenes a procesar
MIN_CAR_AREA    = 8000   # área mínima del bbox del coche en píxeles
MASK_MAX_PCT    = 20.0   # % máximo de la imagen que puede cubrir la máscara

# Umbrales de validación del par
SSIM_MIN        = 0.92
PCP_EXT_MAX     = 25.0
MAPD_EXT_MAX    = 8.0

# Modelo Replicate
REPLICATE_MODEL = "bria/eraser"


# ─── PASO 1: DESCARGA DE IMÁGENES COCO ────────────────────────────────────────

def download_coco_car_images(n: int, output_dir: Path) -> list[dict]:
    """
    Descarga n imágenes de COCO que contengan coches.
    Usa las anotaciones de instances_val2017.json (debe existir en data/).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if not ANN_PATH.exists():
        print(f"[ERROR] No se encontraron anotaciones en {ANN_PATH}")
        print("Descárgalas con:")
        print("  curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
        print("  unzip annotations_trainval2017.zip annotations/instances_val2017.json")
        print(f"  mv annotations/instances_val2017.json {ANN_PATH}")
        return []

    from pycocotools.coco import COCO
    print("Cargando anotaciones COCO...")
    coco    = COCO(str(ANN_PATH))
    cat_ids = coco.getCatIds(catNms=["car"])
    img_ids = coco.getImgIds(catIds=cat_ids)

    collected = []
    for img_id in img_ids:
        if len(collected) >= n:
            break

        img_info = coco.loadImgs(img_id)[0]
        ann_ids  = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns     = coco.loadAnns(ann_ids)

        valid_bboxes = [
            ann["bbox"] for ann in anns
            if ann["bbox"][2] * ann["bbox"][3] >= MIN_CAR_AREA
        ]
        if not valid_bboxes:
            continue

        img_path = output_dir / img_info["file_name"]
        if not img_path.exists():
            r = requests.get(img_info["coco_url"], timeout=15)
            if r.status_code != 200:
                continue
            img_path.write_bytes(r.content)

        collected.append({
            "image_path": str(img_path),
            "image_id":   img_id,
            "bboxes":     valid_bboxes,
        })
        print(f"  [{len(collected)}/{n}] {img_info['file_name']}")

    return collected


# ─── PASO 2: SEGMENTACIÓN CON SAM 2 ───────────────────────────────────────────

def load_sam2():
    """Carga SAM 2 small. Requiere xai_env."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if not SAM2_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint SAM 2 no encontrado en {SAM2_CHECKPOINT}\n"
            "Descárgalo con:\n"
            f"  curl -L -o {SAM2_CHECKPOINT} "
            "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        )

    device    = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"SAM 2 dispositivo: {device}")
    model     = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor


def generate_mask_sam2(predictor, image: np.ndarray, bbox_coco: list) -> np.ndarray:
    """
    Genera máscara binaria del objeto dado bbox en formato COCO [x,y,w,h].
    Devuelve máscara booleana (H, W).
    """
    x, y, w, h  = bbox_coco
    box_xyxy     = np.array([x, y, x + w, y + h])
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=box_xyxy[None, :],
        multimask_output=True
    )
    return masks[np.argmax(scores)]


# ─── PASO 3A: INPAINTING CON REPLICATE ────────────────────────────────────────

def _to_base64_png(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_inpainting_replicate(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """Inpainting via Replicate API (bria/eraser). Requiere REPLICATE_API_TOKEN."""
    import replicate

    try:
        img_b64  = _to_base64_png(image)
        mask_b64 = _to_base64_png((mask * 255).astype(np.uint8))

        output = replicate.run(
            REPLICATE_MODEL,
            input={
                "image": f"data:image/png;base64,{img_b64}",
                "mask":  f"data:image/png;base64,{mask_b64}",
            }
        )

        result_url = output[0] if isinstance(output, list) else output
        r          = requests.get(str(result_url), timeout=30)
        return np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))

    except Exception as e:
        print(f"  [ERROR] Replicate falló: {e}")
        return None


# ─── PASO 3B: INPAINTING CON LAMA (subproceso) ────────────────────────────────

def run_inpainting_lama(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """
    Inpainting con LaMa ejecutado como subproceso en lama_env.
    Usa archivos temporales para comunicar imagen y máscara.
    No requiere importar dependencias de LaMa en xai_env.
    """
    if not LAMA_PYTHON.exists():
        print(f"  [ERROR] Python de lama_env no encontrado en {LAMA_PYTHON}")
        print("  Ajusta la variable LAMA_PYTHON o el path en la configuración.")
        return None

    if not LAMA_MODEL.exists():
        print(f"  [ERROR] Checkpoint LaMa no encontrado en {LAMA_MODEL}")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Guardar inputs
        img_path  = tmpdir / "image.png"
        mask_path = tmpdir / "image_mask001.png"
        Image.fromarray(image).save(img_path)
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

        # Ejecutar LaMa
        cmd = [
            str(LAMA_PYTHON),
            str(LAMA_DIR / "bin" / "predict.py"),
            f"model.path={LAMA_MODEL}",
            f"indir={tmpdir}",
            f"outdir={tmpdir}/output",
            "model.checkpoint=best.ckpt",
        ]

        env = os.environ.copy()
        env["TORCH_HOME"]  = str(LAMA_DIR)
        env["PYTHONPATH"]  = str(LAMA_DIR)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(LAMA_DIR)
        )

        if result.returncode != 0:
            print(f"  [ERROR] LaMa falló:\n{result.stderr[-500:]}")
            return None

        # Leer resultado
        output_dir_path = tmpdir / "output"
        if output_dir_path.exists():
            files = list(output_dir_path.iterdir())
            print(f"  [DEBUG] Archivos en output: {[f.name for f in files]}")
        else:
            print("  [DEBUG] El directorio output no existe")
            return None

        output_path = tmpdir / "output" / "image_mask001.png"
        if not output_path.exists():
            print("  [ERROR] LaMa no generó output")
            return None

        return np.array(Image.open(output_path).convert("RGB"))

# ─── PASO 3C: INPAINTING CON NANOBANANA + COMPOSICIÓN ─────────────────────────

def run_inpainting_nanobanana(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """
    Usa NanoBanana (google/imagen-4) para editar la imagen completa,
    luego compone el resultado usando solo la región de la máscara SAM.
    El fondo queda 100% preservado por construcción.
    """
    from scipy.ndimage import gaussian_filter
    import replicate

    try:
        img_b64 = _to_base64_png(image)

        output = replicate.run(
            "google/imagen-4",
            input={
                "image":  f"data:image/png;base64,{img_b64}",
                "prompt": "Remove the car completely. Fill the area with natural background consistent with the surroundings. No artifacts, no traces of the car.",
            }
        )

        result_url = output[0] if isinstance(output, list) else output
        r          = requests.get(str(result_url), timeout=30)
        nb_result  = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))

        # Ajustar dimensiones si NanoBanana cambia el tamaño
        if nb_result.shape != image.shape:
            nb_result = np.array(
                Image.fromarray(nb_result).resize(
                    (image.shape[1], image.shape[0]),
                    Image.LANCZOS
                )
            )

        # Composición: pegar solo la región de la máscara
        mask_float  = mask.astype(np.float32)
        mask_smooth = gaussian_filter(mask_float, sigma=2)
        mask_3ch    = np.stack([mask_smooth] * 3, axis=2)

        result = (
            image * (1 - mask_3ch) +
            nb_result * mask_3ch
        ).astype(np.uint8)

        return result

    except Exception as e:
        print(f"  [ERROR] NanoBanana falló: {e}")
        return None

# ─── PASO 4: MÉTRICAS ─────────────────────────────────────────────────────────

def compute_metrics(original: np.ndarray, modified: np.ndarray, mask: np.ndarray) -> dict:
    """
    Calcula PCP, MAPD y SSIM globales y en la región exterior a la máscara.
    Las métricas exteriores miden cuánto cambió el fondo (idealmente 0).
    """
    orig_f = original.astype(np.float32)
    mod_f  = modified.astype(np.float32)
    diff   = np.abs(orig_f - mod_f).mean(axis=2)

    threshold  = 5.0
    pcp_global = float((diff > threshold).mean() * 100)
    mapd_global = float(diff.mean())
    ssim_global = float(ssim_metric(original, modified, channel_axis=2, data_range=255))

    exterior = mask == 0
    if exterior.sum() > 0:
        diff_ext   = diff[exterior]
        pcp_ext    = float((diff_ext > threshold).mean() * 100)
        mapd_ext   = float(diff_ext.mean())
    else:
        pcp_ext = mapd_ext = 0.0

    return {
        "pcp_global":    round(pcp_global,  4),
        "mapd_global":   round(mapd_global, 4),
        "ssim_global":   round(ssim_global, 4),
        "pcp_exterior":  round(pcp_ext,     4),
        "mapd_exterior": round(mapd_ext,    4),
    }


def validate_pair(metrics: dict) -> bool:
    return (
        metrics["ssim_global"]   >= SSIM_MIN     and
        metrics["pcp_exterior"]  <= PCP_EXT_MAX  and
        metrics["mapd_exterior"] <= MAPD_EXT_MAX
    )


# ─── PASO 5: GUARDADO ─────────────────────────────────────────────────────────

def save_pair(original: np.ndarray, modified: np.ndarray, mask: np.ndarray,
              metrics: dict, image_id: int, pair_idx: int):
    pair_dir = OUTPUT_DIR / f"pair_{pair_idx:03d}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(original).save(pair_dir / "original.png")
    Image.fromarray(modified).save(pair_dir / "inpainted.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(pair_dir / "mask.png")

    meta = {"image_id": image_id, "pair_idx": pair_idx, **metrics}
    (pair_dir / "metrics.json").write_text(json.dumps(meta, indent=2))

    print(f"  Guardado en {pair_dir}/")
    print(f"  SSIM: {metrics['ssim_global']:.4f}  |  "
          f"PCP global: {metrics['pcp_global']:.1f}%  |  "
          f"PCP exterior: {metrics['pcp_exterior']:.1f}%")


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

def run_pipeline(backend: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Validar backend
    if backend == "replicate":
        if not os.environ.get("REPLICATE_API_TOKEN"):
            print("[ERROR] Falta REPLICATE_API_TOKEN.")
            print("Ejecuta: export REPLICATE_API_TOKEN=tu_token")
            sys.exit(1)
        run_inpainting = run_inpainting_replicate
        print(f"Backend: Replicate ({REPLICATE_MODEL})")
    elif backend == "lama":
        run_inpainting = run_inpainting_lama
        print(f"Backend: LaMa local ({LAMA_MODEL})")

    elif backend == "nanobanana":
        if not os.environ.get("REPLICATE_API_TOKEN"):
            print("[ERROR] Falta REPLICATE_API_TOKEN.")
            sys.exit(1)
        run_inpainting = run_inpainting_nanobanana
        print("Backend: NanoBanana (google/imagen-4) + composición SAM")
    else:
        print(f"[ERROR] Backend desconocido: {backend}. Usa 'replicate' o 'lama'.")
        sys.exit(1)

    # 1. Descargar imágenes
    print("\n=== PASO 1: Descargando imágenes de COCO ===")
    image_records = download_coco_car_images(N_IMAGES, COCO_IMAGES_DIR)
    if not image_records:
        print("[STOP] No se pudieron obtener imágenes.")
        return

    # 2. Cargar SAM 2
    print("\n=== PASO 2: Cargando SAM 2 ===")
    predictor = load_sam2()

    all_metrics = []
    pair_idx    = 0

    for record in image_records:
        img_path = record["image_path"]
        bboxes   = record["bboxes"]
        print(f"\n--- Procesando: {Path(img_path).name} ---")

        image_np = np.array(Image.open(img_path).convert("RGB"))
        bbox     = max(bboxes, key=lambda b: b[2] * b[3])

        # 3. Segmentación
        print("  Generando máscara SAM 2...")
        try:
            mask = generate_mask_sam2(predictor, image_np, bbox)
        except Exception as e:
            print(f"  [ERROR] SAM 2 falló: {e}")
            continue

        mask_pct = mask.mean() * 100
        print(f"  Máscara: {mask_pct:.1f}% de la imagen")

        if mask_pct > MASK_MAX_PCT:
            print(f"  [SKIP] Máscara demasiado grande (>{MASK_MAX_PCT}%)")
            continue

        # 4. Inpainting
        print(f"  Inpainting con {backend}...")
        inpainted = run_inpainting(image_np, mask)
        if inpainted is None:
            continue

        # Ajustar dimensiones si es necesario
        if inpainted.shape != image_np.shape:
            inpainted = np.array(
                Image.fromarray(inpainted).resize(
                    (image_np.shape[1], image_np.shape[0]),
                    Image.LANCZOS
                )
            )

        # 5. Métricas y validación
        metrics = compute_metrics(image_np, inpainted, mask)
        valid   = validate_pair(metrics)
        print(f"  {'✓ VÁLIDO' if valid else '✗ DESCARTADO'}")

        all_metrics.append({
            "image":    img_path,
            "backend":  backend,
            "pair_idx": pair_idx if valid else None,
            "valid":    valid,
            **metrics
        })

        if valid:
            save_pair(image_np, inpainted, mask, metrics, record["image_id"], pair_idx)
            pair_idx += 1

        if backend == "replicate":
            time.sleep(12)  # respetar rate limit

    # 6. Resumen
    print("\n=== RESUMEN ===")
    valid_pairs = [m for m in all_metrics if m["valid"]]
    print(f"Procesadas:  {len(image_records)}")
    print(f"Válidas:     {len(valid_pairs)}")
    print(f"Descartadas: {len(image_records) - len(valid_pairs)}")

    if valid_pairs:
        print(f"\nMétricas medias (pares válidos):")
        for key in ["ssim_global", "pcp_global", "mapd_global", "pcp_exterior", "mapd_exterior"]:
            avg = np.mean([m[key] for m in valid_pairs])
            print(f"  {key}: {avg:.4f}")

    METRICS_FILE.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nMétricas guardadas en {METRICS_FILE}")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera pares contrafactuales para evaluación XAI"
    )
    parser.add_argument(
        "--inpainting",
        choices=["replicate", "lama", "nanobanana"],
        default="replicate",
        help="Backend de inpainting a usar (default: replicate)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N_IMAGES,
        help=f"Número de imágenes a procesar (default: {N_IMAGES})"
    )
    args = parser.parse_args()

    # Permitir sobreescribir N_IMAGES desde CLI
    if args.n != N_IMAGES:
        N_IMAGES = args.n

    run_pipeline(args.inpainting)
