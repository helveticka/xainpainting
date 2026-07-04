# PCP/MAPD/SSIM entre original i contrafactual, global i fora de la mascara
# (exterior ha de ser ~0 si LaMa no toca res fora del vehicle)

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"  # data/ es a l'arrel del repo, no dins src/


def carregar_parell(entry: dict):
    # retorna (orig, inp, mask) alineats a la mida de l'original; mask 255 = vehicle
    orig = np.array(Image.open(entry["img_path"]).convert("RGB"))
    inp  = np.array(Image.open(entry["inpainted_path"]).convert("RGB"))
    mask = cv2.imread(entry["mask_path"], cv2.IMREAD_GRAYSCALE)

    if inp.shape != orig.shape:  # iopaint pot tornar una mida lleugerament distinta
        inp = np.array(Image.fromarray(inp).resize((orig.shape[1], orig.shape[0])))
    if mask.shape != orig.shape[:2]:
        mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
    return orig, inp, mask


def metriques_parell(orig, inp, mask, tau):
    diff = np.abs(orig.astype(float) - inp.astype(float)).mean(axis=2)  # (H,W)
    ext  = mask == 0  # exterior = pixels fora de la mascara del vehicle
    return {
        "pcp_global":  float((diff > tau).mean() * 100),
        "mapd_global": float(diff.mean()),
        "ssim_global": float(ssim(orig, inp, channel_axis=2, data_range=255)),
        "pcp_ext":     float((diff[ext] > tau).mean() * 100),
        "mapd_ext":    float(diff[ext].mean()),
    }


def agregar(valors):  # mitjana i desviacio estandard mostral (ddof=1)
    a = np.array(valors, dtype=float)
    return {"mean": float(a.mean()),
            "std":  float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def run(dataset_dir: Path, output_dir: Path, tau: float):
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((dataset_dir / "dataset_metadata.json").read_text())
    parells  = [e for e in metadata
                if Path(e["img_path"]).exists()
                and Path(e["inpainted_path"]).exists()
                and Path(e["mask_path"]).exists()]
    print(f"Parells disponibles: {len(parells)} / {len(metadata)}")

    per_parell, n_skip = [], 0
    for entry in tqdm(parells, desc="Calculant mètriques de qualitat"):
        pair_id = Path(entry["file_name"]).stem
        try:
            orig, inp, mask = carregar_parell(entry)
            m = metriques_parell(orig, inp, mask, tau)
        except Exception as e:
            print(f"  [skip] {pair_id}: {e}")
            n_skip += 1
            continue
        m["pair_id"] = pair_id
        per_parell.append(m)

    if not per_parell:
        raise SystemExit("No s'ha pogut calcular cap parell. Revisa les rutes.")

    claus = ["pcp_global", "mapd_global", "ssim_global", "pcp_ext", "mapd_ext"]
    agregat = {k: agregar([p[k] for p in per_parell]) for k in claus}

    resultat = {
        "n_parells": len(per_parell),
        "n_descartats": n_skip,
        "tau": tau,
        "nota": "ddof=1; PCP en %, exterior = pixels amb mascara == 0",
        "agregat": agregat,
        "per_parell": per_parell,
    }
    (output_dir / "quality_metrics.json").write_text(json.dumps(resultat, indent=2))

    def fmt(k): return f"{agregat[k]['mean']:.4f} ± {agregat[k]['std']:.4f}"
    print(f"\n{'='*60}\nMÈTRIQUES DE QUALITAT — {len(per_parell)} parells (τ={tau})\n{'='*60}")
    print(f"  PCP  global    : {fmt('pcp_global')} %")
    print(f"  MAPD global    : {fmt('mapd_global')}")
    print(f"  SSIM global    : {fmt('ssim_global')}")
    print(f"  PCP  exterior  : {fmt('pcp_ext')} %")
    print(f"  MAPD exterior  : {fmt('mapd_ext')}")

    g = lambda k: agregat[k]["mean"]
    print(f"\n  Fila LaTeX (global | exterior):")
    print(f"    PCP  & {g('pcp_global'):.2f} & {g('pcp_ext'):.3f} \\\\")
    print(f"    MAPD & {g('mapd_global'):.2f} & {g('mapd_ext'):.3f} \\\\")
    print(f"    SSIM & {g('ssim_global'):.3f} & --- \\\\")
    print(f"\n  Desat a: {output_dir / 'quality_metrics.json'}\n{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Mètriques de qualitat (PCP/MAPD/SSIM) global i exterior sobre el dataset")
    p.add_argument("--dataset-dir", type=Path, default=DATA_DIR / "iopaint_dataset")
    p.add_argument("--output-dir",  type=Path, default=None,
                   help="per defecte, el mateix dataset-dir")
    p.add_argument("--tau", type=float, default=5.0,
                   help="llindar de PCP sobre [0,255] (notebook: 5.0)")
    a = p.parse_args()
    run(a.dataset_dir, a.output_dir or a.dataset_dir, a.tau)
