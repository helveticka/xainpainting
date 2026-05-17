"""
pipeline_iopaint.py
-------------------
Pipeline completo para generar un dataset de pares counterfactuales usando:
  1. COCO API  →  descarga imágenes con coches (categoría 'car')
  2. Anotaciones de segmentación COCO  →  máscara binaria (sin SAM 2)
  3. Dilatación de máscara  →  evitar artefactos en bordes
  4. IOPaint (LaMa)  →  inpainting en batch via CLI subprocess
  5. Métricas  →  SSIM, MAPD, PCP exterior por par

IMPORTANTE: IOPaint funciona en un entorno separado del tuyo (xai_env),
igual que hacías con LaMa. Sigue el mismo patrón de subprocess.

Uso:
    conda activate xai_env
    python pipeline_iopaint.py --n 100 --output-dir data/iopaint_dataset

Requisitos en xai_env:
    pip install pycocotools requests opencv-python scikit-image tqdm

Requisitos en iopaint_env (entorno separado):
    pip install iopaint
    # Al primer arranque descarga big-lama automáticamente (~200MB)
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import requests
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# pycocotools: pip install pycocotools
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuración global
# ---------------------------------------------------------------------------

COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
COCO_CATEGORY = "car"       # categoría a eliminar
MASK_DILATION_PX = 20       # píxeles de dilatación de máscara (clave para calidad)
MIN_CAR_AREA_RATIO = 0.02   # el coche debe ocupar al menos el 2% de la imagen
MAX_CAR_AREA_RATIO = 0.40   # ni tampoco más del 60% (coches muy cercanos)
MAX_CARS_PER_IMAGE = 3      # filtrar imágenes con demasiados coches solapados

# IOPaint: ruta al Python del entorno donde está instalado
# Ajusta esta variable si tu entorno se llama diferente
IOPAINT_PYTHON = os.environ.get(
    "IOPAINT_PYTHON",
    os.path.expanduser("~/miniconda3/envs/iopaint_env/bin/python"),
)


# ---------------------------------------------------------------------------
# PASO 1: Descarga de anotaciones COCO
# ---------------------------------------------------------------------------

def download_coco_annotations(data_dir: Path) -> Path:
    """
    Obtiene las anotaciones de COCO 2017 (instances_val2017.json).

    Estrategia por orden de preferencia:
      1. Si el JSON ya existe (descarga previa o manual), lo usa directamente.
      2. Si existe un zip válido (descarga completa), lo descomprime.
      3. Intenta descargar solo el JSON directamente (~25MB) via curl/wget.
      4. Como último recurso, descarga el zip completo (~241MB) con resume.
    """
    ann_dir = data_dir / "annotations"
    ann_file = ann_dir / "instances_val2017.json"

    # ── Caso 1: JSON ya disponible ───────────────────────────────────────
    if ann_file.exists() and ann_file.stat().st_size > 1_000_000:
        logger.info(f"Anotaciones ya disponibles: {ann_file}")
        return ann_file

    ann_dir.mkdir(parents=True, exist_ok=True)

    # ── Caso 2: Zip ya descargado y completo ─────────────────────────────
    zip_path = data_dir / "annotations.zip"
    ZIP_EXPECTED_SIZE = 230_000_000  # ~230MB mínimo para considerarlo completo

    if zip_path.exists() and zip_path.stat().st_size > ZIP_EXPECTED_SIZE:
        logger.info("Zip encontrado, descomprimiendo...")
        _extract_zip(zip_path, data_dir)
        return ann_file

    # ── Caso 3: Descarga directa del JSON (~25MB, mucho más rápida) ──────
    JSON_URL = "http://images.cocodataset.org/annotations/instances_val2017.json"
    logger.info("Intentando descarga directa del JSON de anotaciones (~25MB)...")

    if _download_with_tool(JSON_URL, ann_file):
        if ann_file.exists() and ann_file.stat().st_size > 1_000_000:
            logger.info(f"JSON descargado correctamente: {ann_file}")
            return ann_file
        else:
            ann_file.unlink(missing_ok=True)

    # ── Caso 4: Descarga del zip con resume ──────────────────────────────
    logger.info("Descargando zip de anotaciones COCO 2017 (~241MB) con resume...")
    _download_with_tool(COCO_ANNOTATIONS_URL, zip_path, resume=True)

    if not zip_path.exists() or zip_path.stat().st_size < ZIP_EXPECTED_SIZE:
        raise RuntimeError(
            "La descarga del zip falló o quedó incompleta.\n"
            "Descárgalo manualmente y colócalo en:\n"
            f"  {zip_path}\n"
            "URL: http://images.cocodataset.org/annotations/annotations_trainval2017.zip\n"
            "\nO descarga solo el JSON (~25MB) y colócalo en:\n"
            f"  {ann_file}\n"
            "URL: http://images.cocodataset.org/annotations/instances_val2017.json"
        )

    _extract_zip(zip_path, data_dir)
    return ann_file


def _download_with_tool(url: str, dest: Path, resume: bool = False) -> bool:
    """
    Descarga una URL usando wget (con resume si está disponible) o curl.
    Devuelve True si el comando terminó sin error.
    """
    import shutil

    # Intentar primero con wget (tiene -c para resume)
    if shutil.which("wget"):
        cmd = ["wget", "-q", "--show-progress"]
        if resume:
            cmd.append("-c")
        cmd += [str(url), "-O", str(dest)]
        logger.info(f"Usando wget: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode == 0

    # Si no hay wget, usar curl (sin resume nativo en esta forma)
    if shutil.which("curl"):
        cmd = ["curl", "-L", "--progress-bar"]
        if resume:
            cmd += ["-C", "-"]
        cmd += [str(url), "-o", str(dest)]
        logger.info(f"Usando curl: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        return result.returncode == 0

    logger.warning("No se encontró wget ni curl. Intentando con urllib...")
    try:
        def _hook(count, block_size, total_size):
            if total_size > 0:
                pct = min(count * block_size * 100 // total_size, 100)
                print(f"\r  {pct}%", end="", flush=True)
        urllib.request.urlretrieve(str(url), str(dest), _hook)
        print()
        return True
    except Exception as e:
        logger.error(f"urllib falló: {e}")
        return False


def _extract_zip(zip_path: Path, dest_dir: Path):
    """Descomprime el zip de anotaciones COCO."""
    import zipfile
    logger.info(f"Descomprimiendo {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    logger.info("Descompresión completada.")


# ---------------------------------------------------------------------------
# PASO 2: Selección de imágenes y construcción de máscaras desde COCO
# ---------------------------------------------------------------------------

def load_coco_car_images(ann_file: Path, n: int) -> list[dict]:
    """
    Usa la API de COCO para obtener imágenes con coches bien segmentados.

    Retorna una lista de dicts con:
        - image_id, file_name, coco_url
        - annotations: lista de anotaciones de coches en esa imagen
        - best_ann: la anotación del coche más grande (la que usaremos)
    """
    logger.info("Cargando índice COCO...")
    coco = COCO(str(ann_file))

    # ID de la categoría 'car' en COCO
    cat_ids = coco.getCatIds(catNms=[COCO_CATEGORY])
    logger.info(f"Categoría '{COCO_CATEGORY}': cat_ids = {cat_ids}")

    # Imágenes que contienen al menos un coche
    img_ids = coco.getImgIds(catIds=cat_ids)
    logger.info(f"Imágenes con '{COCO_CATEGORY}' en COCO val2017: {len(img_ids)}")

    selected = []

    for img_id in img_ids:
        if len(selected) >= n:
            break

        img_info = coco.loadImgs(img_id)[0]
        W, H = img_info["width"], img_info["height"]
        img_area = W * H

        # Anotaciones de coches en esta imagen
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns = coco.loadAnns(ann_ids)

        if not anns:
            continue

        # Filtrar por tamaño relativo
        valid_anns = [
            a for a in anns
            if MIN_CAR_AREA_RATIO <= (a["area"] / img_area) <= MAX_CAR_AREA_RATIO
        ]

        if not valid_anns:
            continue

        # Limitar imágenes con demasiados coches solapados
        if len(valid_anns) > MAX_CARS_PER_IMAGE:
            continue

        # Elegir el coche más grande (mayor área) como objeto principal
        best_ann = max(valid_anns, key=lambda a: a["area"])

        selected.append({
            "image_id": img_id,
            "file_name": img_info["file_name"],
            "coco_url": img_info["coco_url"],
            "width": W,
            "height": H,
            "annotations": valid_anns,
            "best_ann": best_ann,
        })

    logger.info(f"Imágenes seleccionadas: {len(selected)}")
    return selected


def build_mask_from_annotation(ann: dict, height: int, width: int) -> np.ndarray:
    """
    Convierte la segmentación COCO (polígono o RLE) en máscara binaria numpy.
    Retorna array uint8 (H, W) con 255 en la región del objeto.
    """
    seg = ann["segmentation"]

    if isinstance(seg, list):
        # Formato polígono: lista de listas de coordenadas
        rle = coco_mask_utils.frPyObjects(seg, height, width)
        rle = coco_mask_utils.merge(rle)
    elif isinstance(seg, dict):
        # Formato RLE directamente
        rle = seg
    else:
        raise ValueError(f"Formato de segmentación desconocido: {type(seg)}")

    binary_mask = coco_mask_utils.decode(rle)  # (H, W) con valores 0/1
    return (binary_mask * 255).astype(np.uint8)


def dilate_mask(mask: np.ndarray, pixels: int = MASK_DILATION_PX) -> np.ndarray:
    """
    Dilata la máscara para dar margen al modelo de inpainting.
    Esto es lo que diferencia OptiClean de una implementación naïve:
    sin dilatación, LaMa genera artefactos visibles en los bordes del objeto.
    """
    kernel = np.ones((pixels, pixels), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


# ---------------------------------------------------------------------------
# PASO 3: Descarga de imágenes desde COCO URLs
# ---------------------------------------------------------------------------

def download_image(url: str, dest_path: Path) -> bool:
    """Descarga una imagen de COCO si no existe ya."""
    if dest_path.exists():
        return True
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        dest_path.write_bytes(response.content)
        return True
    except Exception as e:
        logger.warning(f"Error descargando {url}: {e}")
        return False


# ---------------------------------------------------------------------------
# PASO 4: IOPaint via subprocess (mismo patrón que tu workaround con LaMa)
# ---------------------------------------------------------------------------

def run_iopaint_batch(
    images_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    device: str = "cpu",
) -> bool:
    """
    Llama a IOPaint como subprocess, exactamente igual que hacías con LaMa.
    IOPaint debe estar instalado en su propio entorno (iopaint_env).

    La convención de IOPaint:
        - Las máscaras deben tener el MISMO nombre que las imágenes.
        - Blanco (255) = región a rellenar. Negro (0) = mantener.
    """
    cmd = [
        IOPAINT_PYTHON, "-m", "iopaint", "run",
        "--model", "lama",
        "--device", device,
        "--image", str(images_dir),
        "--mask", str(masks_dir),
        "--output", str(output_dir),
    ]

    logger.info(f"Ejecutando IOPaint: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=False,   # mostrar output en tiempo real
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"IOPaint falló con código {result.returncode}")
        return False

    logger.info("IOPaint completado correctamente.")
    return True


# ---------------------------------------------------------------------------
# PASO 5: Métricas post-inpainting
# ---------------------------------------------------------------------------

def compute_metrics(
    original_path: Path,
    inpainted_path: Path,
    mask: np.ndarray,
) -> dict:
    """
    Calcula SSIM global, MAPD exterior y PCP exterior.

    - SSIM global: calidad estructural global de la imagen inpaintada.
    - MAPD exterior: diferencia media en píxeles FUERA de la máscara.
      Un MAPD exterior bajo indica que el modelo no ha tocado el fondo.
    - PCP exterior: fracción de píxeles exteriores que han cambiado más
      de un umbral (10/255). Equivale a tu métrica de "contaminación".

    La hipótesis del TFG es que LaMa tendrá MAPD/PCP bajo (conserva el fondo)
    pero artefactos visibles, mientras que NanoBanana tendrá MAPD/PCP alto
    pero resultado perceptualmente convincente.
    """
    orig = cv2.imread(str(original_path))
    inp = cv2.imread(str(inpainted_path))

    if orig is None or inp is None:
        return {"error": "No se pudo leer alguna imagen"}

    if orig.shape != inp.shape:
        inp = cv2.resize(inp, (orig.shape[1], orig.shape[0]))

    # Máscara binaria booleana para exterior (fondo = lo que NO debería cambiar)
    mask_bool = mask > 127
    exterior = ~mask_bool  # True donde debería mantenerse igual

    # SSIM global (entre 0 y 1, mayor es mejor)
    ssim_val = ssim(orig, inp, channel_axis=2, data_range=255)

    # diff en float32 (H, W, 3)
    diff = np.abs(orig.astype(np.float32) - inp.astype(np.float32))
    # diff_max por píxel (H, W) — máximo cambio en cualquier canal
    diff_max_px = diff.max(axis=2)

    # ── Métricas EXTERIOR (fondo — no debería cambiar) ────────────────────
    # MAPD exterior: diff media por canal en el exterior, normalizada a [0,1]
    # diff[exterior] tiene shape (N, 3) → .mean() promedia todos los valores
    # Para obtener la media por píxel primero promediamos canales:
    diff_mean_px = diff.mean(axis=2)  # (H, W) — media de los 3 canales por px
    mapd_ext = float(diff_mean_px[exterior].mean() / 255.0) if exterior.any() else 0.0

    # PCP exterior: fracción de píxeles exteriores con cambio > umbral en algún canal
    threshold = 10
    pcp_ext = float((diff_max_px[exterior] > threshold).mean()) if exterior.any() else 0.0

    # ── Métricas INTERIOR (región del objeto — aquí sí debe cambiar) ──────
    interior = mask_bool
    mapd_int = float(diff_mean_px[interior].mean() / 255.0) if interior.any() else 0.0
    pcp_int  = float((diff_max_px[interior] > threshold).mean()) if interior.any() else 0.0

    # Cobertura de la máscara (qué porcentaje de imagen era el objeto)
    mask_coverage = float(mask_bool.mean()) * 100

    return {
        "ssim_global":       round(float(ssim_val), 4),
        "mapd_exterior":     round(mapd_ext, 4),   # ~0 en LaMa, ~alto en NanoBanana
        "pcp_exterior":      round(pcp_ext, 4),    # ~0 en LaMa, ~alto en NanaBanana
        "mapd_interior":     round(mapd_int, 4),   # cuánto cambió la región del objeto
        "pcp_interior":      round(pcp_int, 4),    # fracción de px del objeto modificados
        "mask_coverage_pct": round(mask_coverage, 2),
    }


# ---------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ---------------------------------------------------------------------------

def run_pipeline(n: int, output_dir: Path, device: str):
    """Ejecuta el pipeline completo de principio a fin."""

    # Estructura de directorios
    data_dir       = output_dir / "raw"
    images_dir     = output_dir / "images_original"
    masks_dir      = output_dir / "masks"
    inpainted_dir  = output_dir / "images_inpainted"
    metadata_file  = output_dir / "dataset_metadata.json"

    for d in [data_dir, images_dir, masks_dir, inpainted_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ── PASO 1: Anotaciones ──────────────────────────────────────────────
    ann_file = download_coco_annotations(data_dir)

    # ── PASO 2: Selección de imágenes ───────────────────────────────────
    selected = load_coco_car_images(ann_file, n)

    if not selected:
        logger.error("No se encontraron imágenes válidas. Revisa los filtros.")
        sys.exit(1)

    # ── PASO 3: Descarga + generación de máscaras ────────────────────────
    logger.info("Descargando imágenes y generando máscaras...")
    metadata = []
    valid_entries = []

    for entry in tqdm(selected, desc="Preparando pares"):
        img_name = entry["file_name"]
        img_path = images_dir / img_name
        mask_path = masks_dir / img_name  # MISMO nombre — requisito de IOPaint

        # Descargar imagen
        ok = download_image(entry["coco_url"], img_path)
        if not ok:
            continue

        # Construir máscara desde anotación COCO
        raw_mask = build_mask_from_annotation(
            entry["best_ann"], entry["height"], entry["width"]
        )
        dilated_mask = dilate_mask(raw_mask, pixels=MASK_DILATION_PX)

        # Guardar máscara (mismo nombre que la imagen)
        cv2.imwrite(str(mask_path), dilated_mask)

        entry["img_path"] = str(img_path)
        entry["mask_path"] = str(mask_path)
        # IOPaint siempre guarda en PNG independientemente del formato de entrada
        img_stem = Path(img_name).stem
        entry["inpainted_path"] = str(inpainted_dir / f"{img_stem}.png")
        valid_entries.append(entry)

    logger.info(f"Pares preparados: {len(valid_entries)}")

    # ── PASO 4: IOPaint en batch ─────────────────────────────────────────
    logger.info("Lanzando IOPaint (LaMa)...")
    ok = run_iopaint_batch(images_dir, masks_dir, inpainted_dir, device=device)

    if not ok:
        logger.error(
            "IOPaint falló. Verifica que está instalado en el entorno correcto.\n"
            f"  IOPAINT_PYTHON={IOPAINT_PYTHON}\n"
            "  Puedes sobrescribir esta ruta con: export IOPAINT_PYTHON=/ruta/a/python"
        )
        sys.exit(1)

    # ── PASO 5: Métricas ─────────────────────────────────────────────────
    logger.info("Calculando métricas por par...")

    for entry in tqdm(valid_entries, desc="Métricas"):
        # IOPaint guarda siempre en PNG; resolver extensión si hace falta
        inpainted_path = Path(entry["inpainted_path"])
        if not inpainted_path.exists():
            # Fallback: buscar con extensión original por si acaso
            alt = inpainted_path.with_suffix(Path(entry["file_name"]).suffix)
            if alt.exists():
                inpainted_path = alt
                entry["inpainted_path"] = str(alt)

        if not inpainted_path.exists():
            logger.warning(f"No encontrado resultado para {entry['file_name']}")
            entry["metrics"] = {"error": "inpainted not found"}
            continue

        # Reconstruir máscara (ya dilatada) para el cálculo
        mask_np = cv2.imread(entry["mask_path"], cv2.IMREAD_GRAYSCALE)

        entry["metrics"] = compute_metrics(
            original_path=Path(entry["img_path"]),
            inpainted_path=inpainted_path,
            mask=mask_np,
        )

    # ── GUARDAR METADATOS ────────────────────────────────────────────────
    # Serializar solo lo que es serializable (eliminar objetos numpy)
    clean_metadata = []
    for e in valid_entries:
        clean_metadata.append({
            "image_id": e["image_id"],
            "file_name": e["file_name"],
            "img_path": e["img_path"],
            "mask_path": e["mask_path"],
            "inpainted_path": e["inpainted_path"],
            "car_area_px": e["best_ann"]["area"],
            "image_wh": [e["width"], e["height"]],
            "num_cars_in_image": len(e["annotations"]),
            "metrics": e.get("metrics", {}),
        })

    with open(metadata_file, "w") as f:
        json.dump(clean_metadata, f, indent=2)

    # ── RESUMEN ──────────────────────────────────────────────────────────
    successful = [e for e in clean_metadata if "error" not in e.get("metrics", {})]

    logger.info("=" * 60)
    logger.info(f"Dataset generado: {output_dir}")
    logger.info(f"  Pares totales:    {len(valid_entries)}")
    logger.info(f"  Con métricas OK:  {len(successful)}")
    logger.info(f"  Metadatos:        {metadata_file}")

    if successful:
        ssim_vals  = [e["metrics"]["ssim_global"]   for e in successful]
        mapd_vals  = [e["metrics"]["mapd_exterior"]  for e in successful]
        pcp_vals   = [e["metrics"]["pcp_exterior"]   for e in successful]

        logger.info(f"  SSIM global:      {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}")
        logger.info(f"  MAPD exterior:    {np.mean(mapd_vals):.4f} ± {np.std(mapd_vals):.4f}")
        logger.info(f"  PCP exterior:     {np.mean(pcp_vals):.4f} ± {np.std(pcp_vals):.4f}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline COCO → Máscaras → IOPaint (LaMa) → Métricas"
    )
    parser.add_argument(
        "--n", type=int, default=15,
        help="Número de pares a generar (default: 15)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/iopaint_dataset",
        help="Directorio raíz de salida"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
        help="Device para IOPaint. En Mac Apple Silicon: mps"
    )
    args = parser.parse_args()

    run_pipeline(
        n=args.n,
        output_dir=Path(args.output_dir),
        device=args.device,
    )
