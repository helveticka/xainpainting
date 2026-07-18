import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

IG_TARGET = 1          # classe explicada: "amb cotxe"
DIFF_TAU  = 0.05       # llindar [0,1] per a la regio de diferencia (~13/255: ignora soroll de compressio)

preprocess = transforms.Compose([  # entrada del model, normalitzada amb estadistiques d'imagenet
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

preprocess_vis = transforms.Compose([  # mateixa geometria en [0,1] sense normalitzar, per a diff i figures
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


def carregar_model(model_path: Path, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.eval().to(device)


def carregar_imatge(path: Path):
    # retorna (tensor_model, tensor_vis): mateixa imatge amb els dos preprocessats
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0), preprocess_vis(img)


def rellevancia_positiva(attr: torch.Tensor) -> np.ndarray:
    # (1,3,224,224) -> mapa (224,224); nomes rellevancia positiva, sense normalitzar
    # (el focus es una fraccio del total, i min-max introduiria biaix en restar el minim global)
    a = attr.squeeze(0).detach().cpu().numpy()
    a = np.clip(a, 0.0, None)
    return a.sum(axis=0)


def regio_diferencia(vis_orig: torch.Tensor, vis_inp: torch.Tensor,
                     tau: float = DIFF_TAU) -> np.ndarray:
    # regio que distingeix el parell: |original - contrafactual| > tau
    diff = (vis_orig - vis_inp).abs().mean(dim=0).numpy()   # (224,224) en [0,1]
    return diff > tau


def focus_score(rellevancia: np.ndarray, regio: np.ndarray) -> float:
    total = rellevancia.sum()
    if total < 1e-12:
        return 0.0
    return float(rellevancia[regio].sum() / total)


def figura_resum(res: list, output_dir: Path) -> Path:
    n      = len(res)
    canvia = [r for r in res if r["pred_changed"]]      # subconjunt on es reporta el Focus
    nc     = len(canvia)

    # sobre tots: per als scatters diagnostics i la distribucio de prob.
    pdrop     = [r["prob_car_drop"]  for r in res]
    forig     = [r["focus_orig"]     for r in res]
    fdrop_all = [r["focus_drop"]     for r in res]
    base      = [r["focus_baseline"] for r in res]
    cols      = ["#c0392b" if r["pred_changed"] else "#3498db" for r in res]

    fdrop_ch  = [r["focus_drop"] for r in canvia] or [0.0]  # distribucio nomes sobre els que canvien

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"Resum XAI — {n} parells · canvia {nc}/{n} ({100*nc/n:.0f}%) · "
        f"Focus reportat sobre els {nc} que canvien",
        fontsize=12)

    ax[0, 0].hist(fdrop_ch, bins=20, color="steelblue", edgecolor="white")
    ax[0, 0].axvline(np.mean(fdrop_ch), color="red", ls="--", label=f"Mitjana={np.mean(fdrop_ch):.3f}")
    ax[0, 0].axvline(0, color="black", lw=1)
    ax[0, 0].set_title(f"Caiguda de Focus · només canvis (N={nc})"); ax[0, 0].legend(fontsize=9)

    ax[0, 1].hist(pdrop, bins=20, color="coral", edgecolor="white")
    ax[0, 1].axvline(np.mean(pdrop), color="red", ls="--", label=f"Mitjana={np.mean(pdrop):.3f}")
    ax[0, 1].axvline(0, color="black", lw=1)
    ax[0, 1].set_title(f"Caiguda de P(cotxe) · tots (N={n})"); ax[0, 1].legend(fontsize=9)

    ax[1, 0].scatter(base, forig, c=cols, alpha=0.5, s=20, edgecolors="none")
    lo, hi = min(base + forig), max(base + forig)
    ax[1, 0].plot([lo, hi], [lo, hi], "k--", lw=1, label="Focus = baseline")
    ax[1, 0].set_xlabel("Baseline (àrea regió diferència / àrea total)")
    ax[1, 0].set_ylabel("Focus (imatge original)")
    ax[1, 0].set_title("Focus vs baseline · vermell = canvia"); ax[1, 0].legend(fontsize=8)

    ax[1, 1].scatter(pdrop, fdrop_all, c=cols, alpha=0.5, s=20, edgecolors="none")
    ax[1, 1].axhline(0, color="black", lw=0.8); ax[1, 1].axvline(0, color="black", lw=0.8)
    ax[1, 1].set_xlabel("Δ P(cotxe)"); ax[1, 1].set_ylabel("Caiguda de Focus")
    ax[1, 1].set_title("Δ prob vs caiguda Focus · vermell = canvia")

    plt.tight_layout()
    path = output_dir / "xai_summary.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path


def run(model_path, dataset_dir, output_dir, n_steps, device_str):
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((dataset_dir / "dataset_metadata.json").read_text())
    parells  = [e for e in metadata
                if Path(e["img_path"]).exists() and Path(e["inpainted_path"]).exists()]
    print(f"Parells disponibles: {len(parells)}")

    device = torch.device(device_str)
    model  = carregar_model(model_path, device)
    ig     = IntegratedGradients(model)
    print(f"Model: {model_path.name} · target IG = classe {IG_TARGET} (amb cotxe)\n")

    res = []
    for entry in tqdm(parells, desc="Calculant mètriques IG"):
        pair_id = Path(entry["file_name"]).stem
        try:
            t_orig, v_orig = carregar_imatge(Path(entry["img_path"]))
            t_inp,  v_inp  = carregar_imatge(Path(entry["inpainted_path"]))
        except Exception as e:
            print(f"  [skip] {pair_id}: {e}")
            continue
        t_orig, t_inp = t_orig.to(device), t_inp.to(device)

        with torch.no_grad():
            p_orig = float(F.softmax(model(t_orig), dim=1)[0][IG_TARGET])
            p_inp  = float(F.softmax(model(t_inp),  dim=1)[0][IG_TARGET])

        rel_orig = rellevancia_positiva(ig.attribute(t_orig, target=IG_TARGET, n_steps=n_steps))
        rel_inp  = rellevancia_positiva(ig.attribute(t_inp,  target=IG_TARGET, n_steps=n_steps))

        regio    = regio_diferencia(v_orig, v_inp)  # regio de referencia, no la mascara coco
        baseline = float(regio.mean())                 # P(píxel aleatori dins l'evidència)

        res.append({
            "pair_id":        pair_id,
            "prob_car_orig":  round(p_orig, 4),
            "prob_car_inp":   round(p_inp,  4),
            "prob_car_drop":  round(p_orig - p_inp, 4),
            "pred_changed":   (p_orig >= 0.5) != (p_inp >= 0.5),
            "focus_orig":     round(focus_score(rel_orig, regio), 4),
            "focus_inp":      round(focus_score(rel_inp,  regio), 4),
            "focus_baseline": round(baseline, 4),
            "focus_drop":     round(focus_score(rel_orig, regio) - focus_score(rel_inp, regio), 4),
            "car_area_pct":   entry.get("car_area_pct"),
        })

    (output_dir / "xai_results.json").write_text(json.dumps(res, indent=2))

    summary_path = figura_resum(res, output_dir)

    # prob(cotxe): tots els parells; focus: nomes els que canvien la prediccio
    n      = len(res)
    canvia = [r for r in res if r["pred_changed"]]
    nc     = len(canvia)
    pdrop  = np.array([r["prob_car_drop"] for r in res])
    fd     = np.array([r["focus_drop"] for r in canvia] or [0.0])
    above  = sum(r["focus_orig"] > r["focus_baseline"] for r in canvia)
    print(f"\n{'='*60}\nRESULTATS XAI — {n} parells\n{'='*60}")
    print(f"  Predicció canvia:      {nc}/{n} ({100*nc/n:.0f}%)")
    print(f"  Caiguda P(cotxe):      {pdrop.mean():+.3f} ± {pdrop.std():.3f}   (tots els parells)")
    print(f"  Caiguda Focus:         {fd.mean():+.3f} ± {fd.std():.3f}   (només els {nc} que canvien)")
    print(f"  Focus orig > baseline: {above}/{nc} ({100*above/nc if nc else 0:.0f}%)   (només canvis)")
    print(f"\n  Resultats: {output_dir / 'xai_results.json'}\n  Figura:    {summary_path}\n{'='*60}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Integrated Gradients + Focus (regió contrafactual)")
    p.add_argument("--model",       type=Path, default=DATA_DIR / "finetune/resnet18_car_classifier.pth")
    p.add_argument("--dataset-dir", type=Path, default=DATA_DIR / "iopaint_dataset")
    p.add_argument("--output-dir",  type=Path, default=DATA_DIR / "xai_results")
    p.add_argument("--n-steps",     type=int,  default=50)
    p.add_argument("--device",      type=str,  default="mps", choices=["cpu", "cuda", "mps"])
    a = p.parse_args()
    run(a.model, a.dataset_dir, a.output_dir, a.n_steps, a.device)
