"""
pipeline_iopaint.py
-------------------
Genera un dataset de parells contrafactuals per a fine-tuning i XAI:
    imatge original (amb 1 cotxe)  →  imatge inpaintada (sense cotxe)

L'objectiu és generar ~500 parells (1000 imatges) per entrenar un
classificador binari ResNet-18 i després aplicar Integrated Gradients.

Filtres de selecció d'imatges:
  - Exactament 1 cotxe a tota la imatge (cap ambigüitat semàntica)
  - El cotxe ocupa entre MIN_CAR_AREA i MAX_CAR_AREA de la imatge
    (prou gran per ser rellevant, prou petit per inpaintar bé)

Ús:
    export IOPAINT_PYTHON=/ruta/a/iopaint_env/bin/python
    python pipeline_iopaint.py --ann-file data/annotations/instances_train2017.json --n 500
"""

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

# ---------------------------------------------------------------------------
# Filtres (ajusta si en surten massa poc o massa)
# ---------------------------------------------------------------------------

# Exactament 1 cotxe per imatge → garanteix causa única per a XAI
N_CARS = 1

# El cotxe ha d'ocupar entre el 5% i el 30% de la imatge.
# < 5%  → massa petit, LaMa no necessita gaire esforç i IG no el detectarà
# > 30% → massa gran, el inpainting és difícil i el fons queda estrany
MIN_CAR_AREA = 0.10
MAX_CAR_AREA = 0.30

# Dilatació de la màscara en píxels.
# Evita que quedin píxels del cotxe visibles als vores després del inpainting.
DILATION_PX = 15

# Ruta al Python de l'entorn amb IOPaint instal·lat.
IOPAINT_PYTHON = os.environ.get(
    "IOPAINT_PYTHON",
    "/opt/homebrew/Caskroom/miniconda/base/envs/iopaint_env/bin/python",
)


# ---------------------------------------------------------------------------
# Selecció d'imatges
# ---------------------------------------------------------------------------

def seleccionar_imatges(ann_file: Path, n: int) -> list[dict]:
    """
    Retorna fins a n imatges de COCO que compleixen els filtres.

    Per cada imatge retorna:
        image_id, file_name, coco_url, width, height, ann (anotació del cotxe)
    """
    print(f"Carregant {ann_file.name}...")
    coco    = COCO(str(ann_file))
    car_ids = coco.getCatIds(catNms=["car"])

    # Totes les imatges que tenen almenys un cotxe
    img_ids = coco.getImgIds(catIds=car_ids)
    random.seed(42)         # ordre reproduïble
    random.shuffle(img_ids)

    seleccionades    = []
    desc_ncotxes     = 0    # més d'1 cotxe
    desc_area        = 0    # cotxe massa petit o massa gran

    for img_id in img_ids:
        if len(seleccionades) >= n:
            break

        img  = coco.loadImgs(img_id)[0]
        area = img["width"] * img["height"]

        # Totes les anotacions de cotxe d'aquesta imatge (sense crowds)
        anns = coco.loadAnns(
            coco.getAnnIds(imgIds=img_id, catIds=car_ids, iscrowd=False)
        )

        # Filtre 1: exactament 1 cotxe
        if len(anns) != N_CARS:
            desc_ncotxes += 1
            continue

        ann          = anns[0]
        area_ratio   = ann["area"] / area

        # Filtre 2: mida acceptable
        if not (MIN_CAR_AREA <= area_ratio <= MAX_CAR_AREA):
            desc_area += 1
            continue

        seleccionades.append({
            "image_id": img_id,
            "file_name": img["file_name"],
            "coco_url":  img["coco_url"],
            "width":     img["width"],
            "height":    img["height"],
            "ann":       ann,
            "area_pct":  round(area_ratio * 100, 1),
        })

    total = len(img_ids)
    print(f"  Imatges candidates:              {total}")
    print(f"  Descartades (més d'1 cotxe):     {desc_ncotxes}")
    print(f"  Descartades (mida fora de rang): {desc_area}")
    print(f"  Seleccionades:                   {len(seleccionades)}")
    return seleccionades


# ---------------------------------------------------------------------------
# Màscara
# ---------------------------------------------------------------------------

def construir_mascara(ann: dict, height: int, width: int) -> np.ndarray:
    """Anotació COCO → màscara binària uint8 (255 = cotxe, 0 = fons)."""
    seg = ann["segmentation"]
    rle = coco_mask.merge(coco_mask.frPyObjects(seg, height, width)) \
          if isinstance(seg, list) else seg
    return (coco_mask.decode(rle) * 255).astype(np.uint8)


def dilatar_mascara(mask: np.ndarray) -> np.ndarray:
    """Dilata la màscara DILATION_PX píxels per evitar vores visibles."""
    kernel = np.ones((DILATION_PX, DILATION_PX), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1)


# ---------------------------------------------------------------------------
# Descàrrega
# ---------------------------------------------------------------------------

def descarregar(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"  Error descarregant {dest.name}: {e}")
        return False


# ---------------------------------------------------------------------------
# IOPaint
# ---------------------------------------------------------------------------

def executar_iopaint(images_dir: Path, masks_dir: Path, output_dir: Path, device: str) -> bool:
    """
    Executa IOPaint (LaMa) en mode batch.
    Requereix que cada màscara tengui el MATEIX nom que la seva imatge.
    """
    cmd = [
        IOPAINT_PYTHON, "-m", "iopaint", "run",
        "--model",  "lama",
        "--device", device,
        "--image",  str(images_dir),
        "--mask",   str(masks_dir),
        "--output", str(output_dir),
    ]
    print(f"\nExecutant IOPaint (LaMa) amb device={device}...")
    result = subprocess.run(cmd)
    return result.returncode == 0


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(ann_file: Path, n: int, output_dir: Path, device: str):

    images_dir    = output_dir / "images_original"
    masks_dir     = output_dir / "masks"
    inpainted_dir = output_dir / "images_inpainted"
    metadata_file = output_dir / "dataset_metadata.json"

    for d in [images_dir, masks_dir, inpainted_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 1. Seleccionar imatges
    seleccionades = seleccionar_imatges(ann_file, n)
    if not seleccionades:
        print("\nNo s'han trobat imatges amb els filtres actuals.")
        print("Prova a ampliar MIN_CAR_AREA o MAX_CAR_AREA.")
        sys.exit(1)

    # 2. Descarregar imatges i generar màscares
    print("\nDescarregant imatges i generant màscares...")
    parells = []
    for entry in tqdm(seleccionades):
        nom      = entry["file_name"]
        img_path = images_dir / nom

        if not descarregar(entry["coco_url"], img_path):
            continue

        mask     = construir_mascara(entry["ann"], entry["height"], entry["width"])
        mask_dil = dilatar_mascara(mask)
        mask_path = masks_dir / nom          # MATEIX nom → requisit IOPaint
        cv2.imwrite(str(mask_path), mask_dil)

        parells.append({
            "image_id":       entry["image_id"],
            "file_name":      nom,
            "img_path":       str(img_path),
            "mask_path":      str(mask_path),
            "inpainted_path": str(inpainted_dir / (Path(nom).stem + ".png")),
            "car_area_pct":   entry["area_pct"],
        })

    print(f"Parells preparats: {len(parells)}")

    # 3. IOPaint
    if not executar_iopaint(images_dir, masks_dir, inpainted_dir, device):
        print(f"\nIOPaint ha fallat.")
        print(f"Comprova que IOPAINT_PYTHON és correcte: {IOPAINT_PYTHON}")
        print("Pots sobreescriure'l amb: export IOPAINT_PYTHON=/ruta/al/python")
        sys.exit(1)

    # 4. Verificar i desar metadades
    valids = [p for p in parells if Path(p["inpainted_path"]).exists()]
    no_trobats = len(parells) - len(valids)
    if no_trobats:
        print(f"  Avís: {no_trobats} inpaintings no trobats (possible error de IOPaint)")

    with open(metadata_file, "w") as f:
        json.dump(valids, f, indent=2)

    # Resum
    print(f"\n{'='*50}")
    print(f"Dataset generat a: {output_dir}")
    print(f"  Parells vàlids:  {len(valids)}")
    print(f"  → images_original/   ({len(valids)} imatges amb cotxe)")
    print(f"  → images_inpainted/  ({len(valids)} imatges sense cotxe)")
    print(f"  → masks/             ({len(valids)} màscares)")
    print(f"  → dataset_metadata.json")
    print(f"{'='*50}")
    print(f"\nPròxim pas: fine-tuning de ResNet-18")
    print(f"  Positius (label=1): images_original/")
    print(f"  Negatius (label=0): images_inpainted/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera dataset contrafactual: COCO → màscares → IOPaint"
    )
    parser.add_argument(
        "--ann-file", type=str,
        default="data/annotations/instances_train2017.json",
        help="Fitxer d'anotacions COCO (train2017 recomanat per volum)"
    )
    parser.add_argument(
        "--n", type=int, default=500,
        help="Nombre de parells a generar (default: 500 → 1000 imatges)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/iopaint_dataset",
        help="Directori de sortida"
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        choices=["cpu", "cuda", "mps"],
        help="Device per a IOPaint (mps = Apple Silicon)"
    )
    args = parser.parse_args()

    run(
        ann_file=Path(args.ann_file),
        n=args.n,
        output_dir=Path(args.output_dir),
        device=args.device,
    )
