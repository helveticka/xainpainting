"""
Pipeline piloto: SAM 2 + Replicate Inpainting
TFG - Explainabilidad XAI con dataset fotorrealista

Flujo:
  1. Descarga imágenes de COCO con coches (con bounding boxes anotados)
  2. SAM 2 genera una máscara precisa del coche usando el bbox como guía
  3. Replicate API hace inpainting de la región enmascarada
  4. Se calculan las métricas PCP, MAPD y SSIM para cada par
  5. Se guardan los pares validados en output/

Requisitos:
  pip install replicate pycocotools Pillow numpy scikit-image requests torch torchvision

SAM 2 (instalar una sola vez):
  pip install git+https://github.com/facebookresearch/sam2.git
  # Descargar checkpoint:
  mkdir -p checkpoints
  wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

Uso:
  export REPLICATE_API_TOKEN=r8_F6hsd7IgL1p0W1k3XR5lRQc26HOXK5J33hlm0
  python pipeline_pilot.py
"""

import os
import io
import json
import time
import base64
import requests
import numpy as np
from pathlib import Path
from PIL import Image
from skimage.metrics import structural_similarity as ssim

import torch
import replicate

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────

COCO_IMAGES_DIR   = Path("coco_images")       # donde se guardan las imágenes descargadas
OUTPUT_DIR        = Path("output")             # pares generados
METRICS_FILE      = OUTPUT_DIR / "metrics.json"

N_IMAGES          = 15    # imágenes del piloto (empieza con 15, suficiente para validar)
MIN_CAR_AREA      = 8000  # píxeles mínimos del bbox del coche (evita coches muy pequeños)

# Umbrales de aceptación del par (basados en los resultados del informe como baseline)
SSIM_MIN          = 0.92  # por encima de esto, el fondo se preserva bien
PCP_MAX           = 25.0  # % máximo de píxeles modificados fuera de la máscara
MAPD_MAX          = 8.0   # diferencia absoluta media máxima aceptable

# Modelo de inpainting en Replicate
INPAINTING_MODEL = "bria/eraser"
# ─── PASO 1: DESCARGA DE IMÁGENES COCO CON COCHES ─────────────────────────────

def download_coco_car_images(n: int, output_dir: Path) -> list[dict]:
    """
    Descarga n imágenes de COCO que contengan coches usando la API pública.
    Devuelve lista de dicts con {image_path, bboxes}.
    No requiere descargar el dataset completo.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Obteniendo anotaciones de COCO...")
    # API pública de COCO: categoría 3 = car
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    # Usamos la API REST de COCO para no descargar todo el zip
    # Alternativa ligera: endpoint de instancias
    instances_url = "https://raw.githubusercontent.com/nightrome/cocostuff/master/dataset/annotations/instances_val2017.json"
    
    # Descargamos solo el JSON de validación (más pequeño que el de train)
    ann_path = Path("instances_val2017.json")
    if not ann_path.exists():
        print("Descargando anotaciones de COCO val2017 (~25MB)...")
        r = requests.get(
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            stream=True
        )
        # Más sencillo: descargar directamente el JSON de instancias
        r2 = requests.get(
            "https://storage.googleapis.com/tfds-data/downloads/manual/coco2017/annotations/instances_val2017.json",
            timeout=30
        )
        if r2.status_code != 200:
            # Fallback: usar pycocotools con descarga manual
            print("Descargando instancias val2017 directamente...")
            r3 = requests.get(
                "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            )
            # Guardamos instrucción para descarga manual
            print("\n[INFO] Descarga automática bloqueada por COCO. Ejecuta manualmente:")
            print("  cd coco_images && wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip")
            print("  unzip annotations_trainval2017.zip")
            return []

    from pycocotools.coco import COCO
    coco = COCO(str(ann_path))

    # ID de la categoría "car" en COCO
    cat_ids = coco.getCatIds(catNms=["car"])
    img_ids = coco.getImgIds(catIds=cat_ids)

    collected = []
    for img_id in img_ids:
        if len(collected) >= n:
            break

        img_info = coco.loadImgs(img_id)[0]
        ann_ids  = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=False)
        anns     = coco.loadAnns(ann_ids)

        # Filtrar coches con área suficiente
        valid_bboxes = [
            ann["bbox"] for ann in anns
            if ann["bbox"][2] * ann["bbox"][3] >= MIN_CAR_AREA
        ]
        if not valid_bboxes:
            continue

        # Descargar la imagen
        img_path = output_dir / img_info["file_name"]
        if not img_path.exists():
            img_url = img_info["coco_url"]
            r = requests.get(img_url, timeout=15)
            if r.status_code != 200:
                continue
            img_path.write_bytes(r.content)

        collected.append({
            "image_path": str(img_path),
            "image_id": img_id,
            "bboxes": valid_bboxes  # formato COCO: [x, y, width, height]
        })
        print(f"  [{len(collected)}/{n}] {img_info['file_name']}")

    return collected


# ─── PASO 2: SEGMENTACIÓN CON SAM 2 ───────────────────────────────────────────

def load_sam2():
    """Carga el modelo SAM 2 small (más ligero, suficiente para el piloto)."""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint = "checkpoints/sam2.1_hiera_small.pt"
    config     = "configs/sam2.1/sam2.1_hiera_s.yaml"

    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint de SAM 2 no encontrado en {checkpoint}.\n"
            "Descárgalo con:\n"
            "  mkdir -p checkpoints\n"
            "  wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"SAM 2 usando dispositivo: {device}")

    model     = build_sam2(config, checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor


def generate_mask_sam2(predictor, image: np.ndarray, bbox_coco: list) -> np.ndarray:
    """
    Genera máscara binaria del objeto dado un bounding box en formato COCO [x,y,w,h].
    Devuelve máscara booleana (H, W).
    """
    # Convertir bbox COCO [x, y, w, h] → [x1, y1, x2, y2] que espera SAM
    x, y, w, h = bbox_coco
    box_xyxy = np.array([x, y, x + w, y + h])

    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=box_xyxy[None, :],   # SAM espera (1, 4)
        multimask_output=True
    )

    # Tomar la máscara con mayor score
    best_idx = np.argmax(scores)
    mask = masks[best_idx]  # bool (H, W)
    return mask


# ─── PASO 3: INPAINTING CON REPLICATE ─────────────────────────────────────────

def mask_to_base64_png(mask: np.ndarray) -> str:
    """Convierte máscara booleana a PNG en base64 (blanco = rellenar)."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    pil_mask   = Image.fromarray(mask_uint8).convert("RGB")
    buf        = io.BytesIO()
    pil_mask.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def image_to_base64_png(image: np.ndarray) -> str:
    """Convierte imagen numpy RGB a PNG en base64."""
    pil_img = Image.fromarray(image)
    buf     = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def run_inpainting(image: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    """
    Llama a Replicate para hacer inpainting de la región enmascarada.
    Devuelve imagen resultado como numpy array RGB, o None si falla.
    """
    img_b64  = image_to_base64_png(image)
    mask_b64 = mask_to_base64_png(mask)

    try:
        output = replicate.run(
        INPAINTING_MODEL,
        input={
            "image": f"data:image/png;base64,{img_b64}",
            "mask":  f"data:image/png;base64,{mask_b64}",
        }
        )

        # output es una URL con la imagen resultado
        result_url = output[0] if isinstance(output, list) else output
        r = requests.get(result_url, timeout=30)
        result_img = np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
        return result_img

    except Exception as e:
        print(f"  [ERROR] Inpainting fallido: {e}")
        return None


# ─── PASO 4: MÉTRICAS ─────────────────────────────────────────────────────────

def compute_metrics(original: np.ndarray, modified: np.ndarray, mask: np.ndarray) -> dict:
    """
    Calcula PCP, MAPD y SSIM entre imagen original y modificada.
    También calcula métricas solo en la región exterior a la máscara
    para verificar que el fondo no ha cambiado.
    """
    orig_f   = original.astype(np.float32)
    mod_f    = modified.astype(np.float32)
    diff     = np.abs(orig_f - mod_f).mean(axis=2)  # (H, W) diferencia por canal promediada

    # Métricas globales (igual que en el informe)
    threshold      = 5.0
    pcp_global     = float((diff > threshold).mean() * 100)
    mapd_global    = float(diff.mean())

    # SSIM global
    ssim_global = float(ssim(
        original, modified,
        channel_axis=2,
        data_range=255
    ))

    # Métricas en la región EXTERIOR a la máscara (el fondo)
    # Esto es lo clave: idealmente el fondo no debería cambiar nada
    exterior = mask == 0
    if exterior.sum() > 0:
        diff_ext   = diff[exterior]
        pcp_ext    = float((diff_ext > threshold).mean() * 100)
        mapd_ext   = float(diff_ext.mean())
    else:
        pcp_ext  = 0.0
        mapd_ext = 0.0

    return {
        "pcp_global":  round(pcp_global,  4),
        "mapd_global": round(mapd_global, 4),
        "ssim_global": round(ssim_global, 4),
        "pcp_exterior":  round(pcp_ext,  4),  # cuánto cambió el fondo (debería ser ~0)
        "mapd_exterior": round(mapd_ext, 4),
    }


def validate_pair(metrics: dict) -> bool:
    """Devuelve True si el par supera los umbrales de calidad."""
    return (
        metrics["ssim_global"]  >= SSIM_MIN and
        metrics["pcp_exterior"] <= PCP_MAX  and
        metrics["mapd_exterior"] <= MAPD_MAX
    )


# ─── PASO 5: GUARDADO ─────────────────────────────────────────────────────────

def save_pair(original: np.ndarray, modified: np.ndarray, mask: np.ndarray,
              metrics: dict, image_id: int, pair_idx: int):
    """Guarda el par original/modificado y la máscara con sus métricas."""
    pair_dir = OUTPUT_DIR / f"pair_{pair_idx:03d}"
    pair_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(original).save(pair_dir / "original.png")
    Image.fromarray(modified).save(pair_dir / "inpainted.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(pair_dir / "mask.png")

    meta = {"image_id": image_id, "pair_idx": pair_idx, **metrics}
    (pair_dir / "metrics.json").write_text(json.dumps(meta, indent=2))

    print(f"  Guardado en {pair_dir}/")
    print(f"  PCP global: {metrics['pcp_global']:.2f}%  |  "
          f"MAPD global: {metrics['mapd_global']:.2f}  |  "
          f"SSIM: {metrics['ssim_global']:.4f}")
    print(f"  PCP exterior (fondo): {metrics['pcp_exterior']:.2f}%  |  "
          f"MAPD exterior: {metrics['mapd_exterior']:.2f}")


# ─── PIPELINE PRINCIPAL ───────────────────────────────────────────────────────

def run_pipeline():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Descargar imágenes
    print("\n=== PASO 1: Descargando imágenes de COCO ===")
    image_records = download_coco_car_images(N_IMAGES, COCO_IMAGES_DIR)

    if not image_records:
        print("[STOP] No se pudieron obtener imágenes. Revisa la descarga manual de anotaciones.")
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

        image_pil = Image.open(img_path).convert("RGB")
        image_np  = np.array(image_pil)

        # Procesar el coche más grande de la imagen (bbox con mayor área)
        bbox = max(bboxes, key=lambda b: b[2] * b[3])

        # 3. Segmentación con SAM 2
        print("  Generando máscara SAM 2...")
        try:
            mask = generate_mask_sam2(predictor, image_np, bbox)
        except Exception as e:
            print(f"  [ERROR] SAM 2 falló: {e}")
            continue

        mask_coverage = mask.mean() * 100
        print(f"  Máscara generada: {mask_coverage:.1f}% de la imagen cubierta")

        # Sanity check: la máscara no debe cubrir más del 40% de la imagen
        if mask_coverage > 40:
            print("  [SKIP] Máscara demasiado grande, posible error de segmentación")
            continue

        # 4. Inpainting con Replicate
        print("  Llamando a Replicate inpainting...")
        inpainted = run_inpainting(image_np, mask)

        if inpainted is None:
            continue

        # Asegurarse de que las dimensiones coinciden
        if inpainted.shape != image_np.shape:
            inpainted = np.array(
                Image.fromarray(inpainted).resize(
                    (image_np.shape[1], image_np.shape[0]),
                    Image.LANCZOS
                )
            )

        # 5. Métricas
        metrics = compute_metrics(image_np, inpainted, mask)
        valid   = validate_pair(metrics)

        status = "✓ VÁLIDO" if valid else "✗ DESCARTADO"
        print(f"  {status}")

        all_metrics.append({
            "image": img_path,
            "pair_idx": pair_idx if valid else None,
            "valid": valid,
            **metrics
        })

        if valid:
            save_pair(image_np, inpainted, mask, metrics, record["image_id"], pair_idx)
            pair_idx += 1

        time.sleep(12)  # respetar rate limit de Replicate

    # 6. Resumen final
    print("\n=== RESUMEN DEL PILOTO ===")
    valid_pairs = [m for m in all_metrics if m["valid"]]
    print(f"Pares generados:  {len(image_records)}")
    print(f"Pares válidos:    {len(valid_pairs)}")
    print(f"Pares descartados: {len(image_records) - len(valid_pairs)}")

    if valid_pairs:
        avg_ssim  = np.mean([m["ssim_global"]    for m in valid_pairs])
        avg_pcp   = np.mean([m["pcp_global"]      for m in valid_pairs])
        avg_mapd  = np.mean([m["mapd_global"]     for m in valid_pairs])
        avg_pcp_e = np.mean([m["pcp_exterior"]    for m in valid_pairs])
        print(f"\nMétricas medias (pares válidos):")
        print(f"  SSIM global:        {avg_ssim:.4f}   (baseline informe: 0.8011)")
        print(f"  PCP global:         {avg_pcp:.2f}%  (baseline informe: 98.1%)")
        print(f"  MAPD global:        {avg_mapd:.2f}   (baseline informe: 14.31)")
        print(f"  PCP exterior fondo: {avg_pcp_e:.2f}%  (ideal: ~0%)")

    # Guardar métricas completas
    METRICS_FILE.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nMétricas guardadas en {METRICS_FILE}")


if __name__ == "__main__":
    token = os.environ.get("REPLICATE_API_TOKEN")
    if not token:
        print("[ERROR] Falta REPLICATE_API_TOKEN.")
        print("Ejecuta: export REPLICATE_API_TOKEN=tu_token_aqui")
        exit(1)

    run_pipeline()
