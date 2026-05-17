"""
xai_analysis.py
---------------
Aplica Integrated Gradients (IG) a cada par de imágenes del dataset counterfactual
(original + inpainted) y genera:
  - Mapas de atribución visualizados (heatmaps)
  - Métricas de cambio de explicación por par
  - Un JSON con los resultados agregados
  - Una figura resumen comparativa

Uso:
    conda activate xai_env   (o deep_blue311, el que tenga captum)
    python xai_analysis.py --dataset-dir data/iopaint_dataset --output-dir data/xai_results

Requisitos:
    pip install torch torchvision captum matplotlib opencv-python scikit-image tqdm
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

# Clases ImageNet relacionadas con vehículos (para diagnóstico)
# No filtramos por ellas, pero las mostramos en los resultados
CAR_RELATED_IMAGENET = {
    817: "sports_car", 511: "convertible", 656: "minivan",
    627: "limousine", 468: "cab", 751: "racer", 779: "school_bus",
    829: "streetcar", 654: "minibus", 407: "ambulance",
}

# Umbral para considerar que un píxel "cambió" en la explicación
ATTR_CHANGE_THRESHOLD = 0.1   # sobre mapa normalizado [0, 1]


# ---------------------------------------------------------------------------
# Preprocesado (idéntico al tutorial del tutor)
# ---------------------------------------------------------------------------

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

transform_normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


def load_image(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Carga una imagen y devuelve:
      - input_tensor:     tensor normalizado listo para el modelo  (1, 3, 224, 224)
      - transformed_img:  tensor sin normalizar para visualización (3, 224, 224)
    """
    img = Image.open(path).convert("RGB")
    transformed_img = transform(img)
    input_tensor = transform_normalize(transformed_img).unsqueeze(0)
    return input_tensor, transformed_img


# ---------------------------------------------------------------------------
# Funciones XAI (basadas en el tutorial del tutor)
# ---------------------------------------------------------------------------

def normalize_data(arr: np.ndarray) -> np.ndarray:
    """Normaliza un array a [0, 1]."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def viz_xai(attributions: torch.Tensor) -> np.ndarray:
    """
    Convierte las atribuciones IG a un mapa 2D visualizable.
    Sigue exactamente la función del tutorial del tutor:
      - Toma valor absoluto
      - Normaliza a [0, 1]
      - Promedia los 3 canales → (224, 224)
    """
    arr_np = attributions.cpu().detach().numpy()[0]   # (3, 224, 224)
    return np.mean(normalize_data(np.abs(arr_np)), axis=0)  # (224, 224)


def compute_ig(model, input_tensor: torch.Tensor, target_class: int,
               n_steps: int = 50) -> torch.Tensor:
    """
    Aplica Integrated Gradients con baseline negro (igual que en el tutorial).
    """
    ig = IntegratedGradients(model)
    attributions = ig.attribute(input_tensor, target=target_class, n_steps=n_steps)
    return attributions


# ---------------------------------------------------------------------------
# Métricas de cambio de explicación
# ---------------------------------------------------------------------------

def explanation_change_metrics(attr_orig: np.ndarray, attr_inp: np.ndarray,
                                mask_224: np.ndarray) -> dict:
    """
    Calcula cuánto cambia la explicación entre la imagen original e inpainted,
    especialmente en la región de la máscara (donde estaba el coche).

    Parámetros
    ----------
    attr_orig : mapa de atribución de la imagen original   (224, 224) en [0, 1]
    attr_inp  : mapa de atribución de la imagen inpainted  (224, 224) en [0, 1]
    mask_224  : máscara binaria redimensionada a 224x224   (224, 224) bool

    Métricas devueltas
    ------------------
    - attr_diff_mean_mask   : cambio medio de atribución DENTRO de la máscara
    - attr_diff_mean_outside: cambio medio de atribución FUERA de la máscara
    - attr_orig_in_mask     : atribución media original dentro de la máscara
    - attr_inp_in_mask      : atribución media inpainted dentro de la máscara
    - relative_drop         : caída relativa de atribución en la región del coche
                              (1 = desaparece completamente, 0 = no cambia)
    - ssim_attr             : SSIM entre los dos mapas de atribución
    """
    diff = np.abs(attr_orig - attr_inp)
    mask_bool = mask_224.astype(bool)
    outside = ~mask_bool

    attr_orig_in  = float(attr_orig[mask_bool].mean()) if mask_bool.any() else 0.0
    attr_inp_in   = float(attr_inp[mask_bool].mean())  if mask_bool.any() else 0.0
    diff_in       = float(diff[mask_bool].mean())       if mask_bool.any() else 0.0
    diff_out      = float(diff[outside].mean())          if outside.any() else 0.0

    # Caída relativa: cuánto cayó la atribución en la zona del coche
    relative_drop = float((attr_orig_in - attr_inp_in) / (attr_orig_in + 1e-8))

    # SSIM entre mapas de atribución (mide similitud estructural de las explicaciones)
    ssim_attr = float(ssim(attr_orig, attr_inp, data_range=1.0))

    return {
        "attr_orig_in_mask":    round(attr_orig_in, 4),
        "attr_inp_in_mask":     round(attr_inp_in, 4),
        "attr_diff_mean_mask":  round(diff_in, 4),
        "attr_diff_mean_outside": round(diff_out, 4),
        "relative_drop":        round(relative_drop, 4),
        "ssim_attr":            round(ssim_attr, 4),
    }


# ---------------------------------------------------------------------------
# Visualización de un par
# ---------------------------------------------------------------------------

def visualize_pair(img_orig_t: torch.Tensor, img_inp_t: torch.Tensor,
                   attr_orig: np.ndarray, attr_inp: np.ndarray,
                   mask_224: np.ndarray,
                   pred_orig: int, pred_inp: int,
                   pair_id: str, output_dir: Path):
    """
    Genera una figura con 6 paneles para un par:
      [orig] [inpainted] [mask]
      [IG orig] [IG inpainted] [diff IG]
    """
    def t2img(t):
        """Tensor (3, H, W) → numpy (H, W, 3) en [0, 1]."""
        arr = t.permute(1, 2, 0).numpy()
        return np.clip(arr, 0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(f"Par: {pair_id}\n"
                 f"Pred original: {pred_orig}  |  Pred inpainted: {pred_inp}",
                 fontsize=11)

    # Fila 1: imágenes
    axes[0, 0].imshow(t2img(img_orig_t))
    axes[0, 0].set_title("Imagen original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(t2img(img_inp_t))
    axes[0, 1].set_title("Imagen inpainted (sin coche)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mask_224, cmap="gray")
    axes[0, 2].set_title("Máscara (región del coche)")
    axes[0, 2].axis("off")

    # Fila 2: explicaciones
    im1 = axes[1, 0].imshow(attr_orig, cmap="hot", vmin=0, vmax=1)
    axes[1, 0].set_title("IG — imagen original")
    axes[1, 0].axis("off")
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)

    im2 = axes[1, 1].imshow(attr_inp, cmap="hot", vmin=0, vmax=1)
    axes[1, 1].set_title("IG — imagen inpainted")
    axes[1, 1].axis("off")
    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)

    diff = attr_orig - attr_inp
    im3 = axes[1, 2].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 2].set_title("Diferencia IG (orig − inp)\nRojo = atención perdida")
    axes[1, 2].axis("off")
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    out_path = output_dir / f"{pair_id}_xai.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_xai_pipeline(dataset_dir: Path, output_dir: Path, n_steps: int,
                     device_str: str, max_pairs: int):

    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = output_dir / "pair_figures"
    pairs_dir.mkdir(exist_ok=True)

    # ── Cargar modelo (ResNet-18, igual que el tutorial) ─────────────────
    device = torch.device(device_str)
    print(f"[INFO] Cargando ResNet-18 en {device}...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    model.to(device)

    # ── Leer metadatos del dataset ────────────────────────────────────────
    metadata_file = dataset_dir / "dataset_metadata.json"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"No se encontró dataset_metadata.json en {dataset_dir}.\n"
            "Asegúrate de haber ejecutado pipeline_iopaint.py primero."
        )

    with open(metadata_file) as f:
        metadata = json.load(f)

    # Filtrar pares con inpainted disponible y sin error
    valid_pairs = [
        e for e in metadata
        if "error" not in e.get("metrics", {})
        and Path(e["inpainted_path"]).exists()
        and Path(e["img_path"]).exists()
    ]

    if not valid_pairs:
        raise RuntimeError("No hay pares válidos en el dataset. Revisa los paths.")

    if max_pairs > 0:
        valid_pairs = valid_pairs[:max_pairs]

    print(f"[INFO] Procesando {len(valid_pairs)} pares...")

    results = []

    for entry in tqdm(valid_pairs, desc="Aplicando IG"):
        pair_id = Path(entry["file_name"]).stem

        # ── Cargar imágenes ───────────────────────────────────────────────
        orig_path = Path(entry["img_path"])
        inp_path  = Path(entry["inpainted_path"])

        try:
            input_orig, img_orig_t = load_image(orig_path)
            input_inp,  img_inp_t  = load_image(inp_path)
        except Exception as ex:
            print(f"[WARN] Error cargando {pair_id}: {ex}")
            continue

        input_orig = input_orig.to(device)
        input_inp  = input_inp.to(device)

        # ── Predicciones ─────────────────────────────────────────────────
        with torch.no_grad():
            out_orig = model(input_orig)
            out_inp  = model(input_inp)

        pred_orig = int(torch.argmax(out_orig, dim=1).item())
        pred_inp  = int(torch.argmax(out_inp,  dim=1).item())

        prob_orig = float(F.softmax(out_orig, dim=1).max().item())
        prob_inp  = float(F.softmax(out_inp,  dim=1).max().item())

        # ── Integrated Gradients ─────────────────────────────────────────
        try:
            attr_orig_t = compute_ig(model, input_orig, pred_orig, n_steps)
            attr_inp_t  = compute_ig(model, input_inp,  pred_inp,  n_steps)
        except Exception as ex:
            print(f"[WARN] Error en IG para {pair_id}: {ex}")
            continue

        attr_orig = viz_xai(attr_orig_t)
        attr_inp  = viz_xai(attr_inp_t)

        # ── Máscara redimensionada a 224×224 ─────────────────────────────
        mask_raw = cv2.imread(entry["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask_224 = cv2.resize(mask_raw, (224, 224), interpolation=cv2.INTER_NEAREST)
        mask_224 = (mask_224 > 127).astype(np.uint8)

        # ── Métricas de cambio de explicación ────────────────────────────
        xai_metrics = explanation_change_metrics(attr_orig, attr_inp, mask_224)

        # ── Visualización ─────────────────────────────────────────────────
        fig_path = visualize_pair(
            img_orig_t, img_inp_t,
            attr_orig, attr_inp,
            mask_224,
            pred_orig, pred_inp,
            pair_id, pairs_dir,
        )

        # ── Acumular resultado ────────────────────────────────────────────
        results.append({
            "pair_id":       pair_id,
            "pred_orig":     pred_orig,
            "pred_inp":      pred_inp,
            "prob_orig":     round(prob_orig, 4),
            "prob_inp":      round(prob_inp, 4),
            "pred_changed":  pred_orig != pred_inp,
            "is_car_pred":   pred_orig in CAR_RELATED_IMAGENET,
            "xai_metrics":   xai_metrics,
            "inpaint_metrics": entry.get("metrics", {}),
            "figure_path":   str(fig_path),
        })

    # ── Guardar resultados JSON ───────────────────────────────────────────
    results_file = output_dir / "xai_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # ── Figura resumen ────────────────────────────────────────────────────
    _plot_summary(results, output_dir)

    # ── Resumen por pantalla ──────────────────────────────────────────────
    n = len(results)
    print("\n" + "=" * 60)
    print(f"RESULTADOS XAI — {n} pares procesados")
    print("=" * 60)
    if n > 0:
        drops      = [r["xai_metrics"]["relative_drop"] for r in results]
        ssims_attr = [r["xai_metrics"]["ssim_attr"]     for r in results]
        changed    = sum(r["pred_changed"] for r in results)
        car_preds  = sum(r["is_car_pred"]  for r in results)

        print(f"  Predicción cambia al inpaintar:  {changed}/{n} ({100*changed/n:.1f}%)")
        print(f"  Pred. original era un coche:     {car_preds}/{n} ({100*car_preds/n:.1f}%)")
        print(f"  Caída relativa de atribución")
        print(f"    en la región del coche:        {np.mean(drops):.3f} ± {np.std(drops):.3f}")
        print(f"  SSIM entre mapas de atribución:  {np.mean(ssims_attr):.3f} ± {np.std(ssims_attr):.3f}")
        print(f"\n  Resultados guardados en: {output_dir}")
    print("=" * 60)


def _plot_summary(results: list, output_dir: Path):
    """Genera una figura de resumen con distribuciones de las métricas clave."""
    if not results:
        return

    drops   = [r["xai_metrics"]["relative_drop"]     for r in results]
    ssims_a = [r["xai_metrics"]["ssim_attr"]          for r in results]
    diffs_m = [r["xai_metrics"]["attr_diff_mean_mask"] for r in results]
    ssims_i = [r["inpaint_metrics"].get("ssim_global", None) for r in results]
    ssims_i = [v for v in ssims_i if v is not None]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Resumen del experimento XAI — Integrated Gradients", fontsize=13)

    axes[0].hist(drops, bins=10, color="steelblue", edgecolor="white")
    axes[0].axvline(np.mean(drops), color="red", linestyle="--",
                    label=f"Media={np.mean(drops):.2f}")
    axes[0].set_title("Caída relativa de atribución\nen la región del coche")
    axes[0].set_xlabel("relative_drop  (1 = desaparece del todo)")
    axes[0].set_ylabel("Nº de pares")
    axes[0].legend()

    axes[1].hist(ssims_a, bins=10, color="coral", edgecolor="white")
    axes[1].axvline(np.mean(ssims_a), color="red", linestyle="--",
                    label=f"Media={np.mean(ssims_a):.2f}")
    axes[1].set_title("SSIM entre mapas de atribución\n(orig vs inpainted)")
    axes[1].set_xlabel("SSIM  (1 = explicaciones idénticas)")
    axes[1].legend()

    if ssims_i:
        axes[2].scatter(ssims_i[:len(drops)], drops[:len(ssims_i)],
                        alpha=0.7, color="seagreen", edgecolors="white")
        axes[2].set_xlabel("SSIM inpainting (calidad pixel-level)")
        axes[2].set_ylabel("Caída de atribución (utilidad XAI)")
        axes[2].set_title("¿Correlaciona la calidad\nde inpainting con la utilidad XAI?")
    else:
        axes[2].text(0.5, 0.5, "Sin datos SSIM inpainting",
                     ha="center", va="center", transform=axes[2].transAxes)

    plt.tight_layout()
    out_path = output_dir / "xai_summary.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Figura resumen guardada: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aplica Integrated Gradients al dataset counterfactual"
    )
    parser.add_argument(
        "--dataset-dir", type=str, default="data/iopaint_dataset",
        help="Directorio raíz del dataset generado por pipeline_iopaint.py"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/xai_results",
        help="Directorio donde guardar los resultados XAI"
    )
    parser.add_argument(
        "--n-steps", type=int, default=50,
        help="Pasos de integración para IG (más pasos = más preciso, más lento)"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "mps"],
        help="Device: cpu / cuda / mps (Apple Silicon)"
    )
    parser.add_argument(
        "--max-pairs", type=int, default=0,
        help="Limitar a N pares (0 = todos)"
    )
    args = parser.parse_args()

    run_xai_pipeline(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        n_steps=args.n_steps,
        device_str=args.device,
        max_pairs=args.max_pairs,
    )
