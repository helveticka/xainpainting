# genera parells contrafactuals: coco (1 cotxe) -> mascara dilatada -> iopaint (lama)

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import requests
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # data/ es a l'arrel del repo, no dins src/

N_CARS       = 1            # exactament 1 cotxe per imatge, evita ambiguitat semantica
MIN_CAR_AREA = 0.10         # prou gran per ser rellevant
MAX_CAR_AREA = 0.30         # prou petit per reconstruir be el fons
DILATION_PX  = 15           # dilatacio de la mascara: evita vores residuals del cotxe
SEED         = 42           # seleccio reproduible

IOPAINT_PYTHON = os.environ.get(  # python de l'entorn amb iopaint instal·lat
    "IOPAINT_PYTHON",
    "/opt/homebrew/Caskroom/miniconda/base/envs/iopaint/bin/python")


def seleccionar_imatges(ann_file: Path, n: int) -> list[dict]:
    print(f"Carregant {ann_file.name}...")
    coco    = COCO(str(ann_file))
    car_ids = coco.getCatIds(catNms=["car"])
    img_ids = coco.getImgIds(catIds=car_ids)
    random.seed(SEED)
    random.shuffle(img_ids)

    triades, n_multi, n_area = [], 0, 0
    for img_id in img_ids:
        if len(triades) >= n:
            break
        img  = coco.loadImgs(img_id)[0]
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=car_ids, iscrowd=False))

        if len(anns) != N_CARS:                       # filtre 1: un sol cotxe
            n_multi += 1
            continue
        ratio = anns[0]["area"] / (img["width"] * img["height"])
        if not (MIN_CAR_AREA <= ratio <= MAX_CAR_AREA):  # filtre 2: mida
            n_area += 1
            continue

        triades.append({
            "image_id": img_id,
            "file_name": img["file_name"],
            "coco_url":  img["coco_url"],
            "width":     img["width"],
            "height":    img["height"],
            "ann":       anns[0],
            "area_pct":  round(ratio * 100, 1),
        })

    print(f"  Candidates: {len(img_ids)}  ·  descartades >1 cotxe: {n_multi}  "
          f"·  mida fora de rang: {n_area}  ·  seleccionades: {len(triades)}")
    return triades


def construir_mascara(ann: dict, h: int, w: int) -> np.ndarray:
    # anotacio coco -> mascara binaria uint8 dilatada (255 = cotxe)
    seg = ann["segmentation"]
    rle = coco_mask.merge(coco_mask.frPyObjects(seg, h, w)) if isinstance(seg, list) else seg
    mask = (coco_mask.decode(rle) * 255).astype(np.uint8)
    kernel = np.ones((DILATION_PX, DILATION_PX), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


def executar_iopaint(images_dir: Path, masks_dir: Path, out_dir: Path, device: str) -> bool:
    # cada mascara ha de dur el mateix nom que la seva imatge (requisit d'iopaint)
    cmd = [IOPAINT_PYTHON, "-m", "iopaint", "run",
           "--model", "lama", "--device", device,
           "--image", str(images_dir), "--mask", str(masks_dir), "--output", str(out_dir)]
    print(f"\nExecutant IOPaint (LaMa, device={device})...")
    return subprocess.run(cmd).returncode == 0


def descarregar(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        dest.write_bytes(requests.get(url, timeout=15).content)
        return True
    except Exception as e:
        print(f"  Error descarregant {dest.name}: {e}")
        return False


def run(ann_file: Path, n: int, output_dir: Path, device: str):
    images_dir    = output_dir / "images_original"
    masks_dir     = output_dir / "masks"
    inpainted_dir = output_dir / "images_inpainted"
    for d in (images_dir, masks_dir, inpainted_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Selecció
    triades = seleccionar_imatges(ann_file, n)
    if not triades:
        sys.exit("No s'han trobat imatges amb els filtres actuals. Amplia MIN/MAX_CAR_AREA.")

    # 2. Descàrrega d'imatges + generació de màscares
    print("\nDescarregant imatges i generant màscares...")
    parells = []
    for e in tqdm(triades):
        nom = e["file_name"]
        img_path = images_dir / nom
        if not descarregar(e["coco_url"], img_path):
            continue
        mask = construir_mascara(e["ann"], e["height"], e["width"])
        cv2.imwrite(str(masks_dir / nom), mask)          # mateix nom → requisit IOPaint
        parells.append({
            "image_id":       e["image_id"],
            "file_name":      nom,
            "img_path":       str(img_path),
            "mask_path":      str(masks_dir / nom),
            "inpainted_path": str(inpainted_dir / f"{Path(nom).stem}.png"),
            "car_area_pct":   e["area_pct"],
        })
    print(f"Parells preparats: {len(parells)}")

    # 3. Inpainting (batch)
    if not executar_iopaint(images_dir, masks_dir, inpainted_dir, device):
        sys.exit(f"IOPaint ha fallat. Comprova IOPAINT_PYTHON: {IOPAINT_PYTHON}")

    # 4. Verificació i desat de metadades
    valids = [p for p in parells if Path(p["inpainted_path"]).exists()]
    if len(valids) < len(parells):
        print(f"  Avís: {len(parells) - len(valids)} inpaintings no trobats.")
    (output_dir / "dataset_metadata.json").write_text(json.dumps(valids, indent=2))

    print(f"\n{'='*50}\nDataset generat a: {output_dir}")
    print(f"  Parells vàlids: {len(valids)}  "
          f"(originals + contrafactuals = {len(valids)*2} imatges)\n{'='*50}")
    print("Pròxim pas: fine-tuning de ResNet-18 (02_finetune_resnet.py)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Genera dataset contrafactual: COCO → màscares → IOPaint")
    p.add_argument("--ann-file",   type=Path, default=DATA_DIR / "annotations/instances_train2017.json")
    p.add_argument("--n",          type=int,  default=500, help="parells a generar (500 → 1000 imatges)")
    p.add_argument("--output-dir", type=Path, default=DATA_DIR / "iopaint_dataset")
    p.add_argument("--device",     type=str,  default="mps", choices=["cpu", "cuda", "mps"])
    a = p.parse_args()
    run(a.ann_file, a.n, a.output_dir, a.device)
