"""
xai_analysis.py
---------------
Aplica Integrated Gradients (IG) sobre el classificador binari entrenat
(amb cotxe=1 / sense cotxe=0).

Sortides:
  - xai_results.json        → mètriques numèriques de tots els parells
  - xai_summary.png         → figura resum agregada (4 gràfics)
  - figures/                → figures individuals NOMÉS dels N casos
                               més representatius (--top-n, default 20)
  - informe.html            → informe lleuger: estadístiques + taula +
                               figures dels casos seleccionats (~2-5 MB)

La figura individual per parell s'ha eliminat de l'informe massiu.
Amb 410 parells, generar una figura per a cada un produïa ~580 MB.

Ús:
    python xai_analysis.py \
        --model data/finetune/resnet18_car_classifier.pth \
        --dataset-dir data/iopaint_dataset \
        --output-dir data/xai_results \
        --device mps
"""

import argparse
import base64
import json
import torch.nn.functional as F
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuració
# ---------------------------------------------------------------------------

IG_TARGET = 1   # classe "amb cotxe"

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

preprocess_vis = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def carregar_model(model_path: Path, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)
    print(f"Model carregat: {model_path.name}")
    return model


# ---------------------------------------------------------------------------
# Funcions XAI
# ---------------------------------------------------------------------------

def carregar_imatge(path: Path):
    img = Image.open(path).convert("RGB")
    return preprocess(img).unsqueeze(0), preprocess_vis(img)


def mapa_ig(atribucions: torch.Tensor) -> np.ndarray:
    """(1,3,224,224) → (224,224) en [0,1]. Idèntic al tutorial del tutor."""
    arr = atribucions.cpu().detach().numpy()[0]
    arr = np.abs(arr)
    mn, mx = arr.min(), arr.max()
    if mx - mn > 1e-8:
        arr = (arr - mn) / (mx - mn)
    return arr.mean(axis=0)


def focus_score(attr: np.ndarray, mask: np.ndarray) -> float:
    """Fracció de l'atribució total dins la màscara del cotxe."""
    total = attr.sum()
    if total < 1e-8:
        return 0.0
    return float(attr[mask.astype(bool)].sum() / total)


def overlay_heatmap(img_tensor, attr_map: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    img     = np.clip(img_tensor.permute(1, 2, 0).numpy(), 0, 1)
    heatmap = plt.cm.hot(attr_map)[:, :, :3]
    return np.clip(img * (1 - alpha) + heatmap * alpha, 0, 1)


# ---------------------------------------------------------------------------
# Figura d'un parell individual (es desa en disc, NO s'incrusta massivament)
# ---------------------------------------------------------------------------

def figura_parell(entry: dict, model, ig, device: torch.device,
                  figures_dir: Path) -> Path:
    """Genera i desa la figura de 6 panells per a un parell concret."""
    pair_id = Path(entry["file_name"]).stem

    t_orig, v_orig = carregar_imatge(Path(entry["img_path"]))
    t_inp,  v_inp  = carregar_imatge(Path(entry["inpainted_path"]))
    t_orig, t_inp  = t_orig.to(device), t_inp.to(device)

    with torch.no_grad():
        out_orig = model(t_orig)
        out_inp  = model(t_inp)

    prob_orig = float(F.softmax(out_orig, dim=1)[0][IG_TARGET])
    prob_inp  = float(F.softmax(out_inp,  dim=1)[0][IG_TARGET])

    attr_orig = mapa_ig(ig.attribute(t_orig, target=IG_TARGET, n_steps=50))
    attr_inp  = mapa_ig(ig.attribute(t_inp,  target=IG_TARGET, n_steps=50))

    mask_raw = cv2.imread(entry["mask_path"], cv2.IMREAD_GRAYSCALE)
    mask_224 = (cv2.resize(mask_raw, (224, 224),
                           interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)

    f_orig = focus_score(attr_orig, mask_224)
    f_inp  = focus_score(attr_inp,  mask_224)

    def t2img(t):
        return np.clip(t.permute(1, 2, 0).numpy(), 0, 1)

    canvi_str = "✓ CANVIA" if (prob_orig >= 0.5) != (prob_inp >= 0.5) else "✗ no canvia"

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        f"Parell: {pair_id}  [{canvi_str}]\n"
        f"Prob(cotxe): {prob_orig:.3f} → {prob_inp:.3f}   "
        f"Focus: {f_orig:.3f} → {f_inp:.3f}  (Δ={f_orig-f_inp:+.3f})",
        fontsize=10
    )

    axes[0, 0].imshow(t2img(v_orig)); axes[0, 0].set_title("Original");             axes[0, 0].axis("off")
    axes[0, 1].imshow(t2img(v_inp));  axes[0, 1].set_title("Inpaintat (sense cotxe)"); axes[0, 1].axis("off")
    axes[0, 2].imshow(mask_224, cmap="gray"); axes[0, 2].set_title("Màscara");       axes[0, 2].axis("off")

    axes[1, 0].imshow(overlay_heatmap(v_orig, attr_orig))
    axes[1, 0].set_title(f"IG original  Focus={f_orig:.3f}"); axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay_heatmap(v_inp, attr_inp))
    axes[1, 1].set_title(f"IG inpaintat  Focus={f_inp:.3f}"); axes[1, 1].axis("off")

    diff = attr_orig - attr_inp
    im   = axes[1, 2].imshow(diff, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[1, 2].set_title("Diferència IG\nVermell = atenció perduda"); axes[1, 2].axis("off")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    path = figures_dir / f"{pair_id}_xai.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Figura resum agregada
# ---------------------------------------------------------------------------

def figura_resum(resultats: list, output_dir: Path) -> Path:
    n          = len(resultats)
    focus_drop = [r["focus_drop"]    for r in resultats]
    prob_drop  = [r["prob_car_drop"] for r in resultats]
    focus_orig = [r["focus_orig"]    for r in resultats]
    baselines  = [r["focus_baseline"] for r in resultats]
    canvis     = [r["pred_changed"]  for r in resultats]

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(
        f"Resum XAI — {n} parells  |  ResNet-18 fine-tuned  |  "
        f"Predicció canvia: {sum(canvis)}/{n} ({100*sum(canvis)/n:.0f}%)",
        fontsize=12
    )

    # 1. Distribució caiguda Focus
    axes[0, 0].hist(focus_drop, bins=20, color="steelblue", edgecolor="white")
    axes[0, 0].axvline(np.mean(focus_drop), color="red", linestyle="--",
                       label=f"Mitjana = {np.mean(focus_drop):.3f}")
    axes[0, 0].axvline(0, color="black", linewidth=1)
    axes[0, 0].set_title("Distribució caiguda Focus (orig − inp)")
    axes[0, 0].set_xlabel("focus_drop  (>0 = model mirava el cotxe)")
    axes[0, 0].set_ylabel("Nº parells"); axes[0, 0].legend(fontsize=9)

    # 2. Distribució caiguda prob(cotxe)
    axes[0, 1].hist(prob_drop, bins=20, color="coral", edgecolor="white")
    axes[0, 1].axvline(np.mean(prob_drop), color="red", linestyle="--",
                       label=f"Mitjana = {np.mean(prob_drop):.3f}")
    axes[0, 1].axvline(0, color="black", linewidth=1)
    axes[0, 1].set_title("Distribució caiguda prob(cotxe)")
    axes[0, 1].set_xlabel("Δ prob(cotxe)  (>0 = predicció s'allunya del cotxe)")
    axes[0, 1].set_ylabel("Nº parells"); axes[0, 1].legend(fontsize=9)

    # 3. Focus orig vs baseline (scatter)
    colors = ["#c0392b" if c else "#3498db" for c in canvis]
    axes[1, 0].scatter(baselines, focus_orig, c=colors, alpha=0.5,
                       edgecolors="none", s=20)
    mn = min(min(baselines), min(focus_orig))
    mx = max(max(baselines), max(focus_orig))
    axes[1, 0].plot([mn, mx], [mn, mx], "k--", linewidth=1, label="Focus = baseline")
    axes[1, 0].set_xlabel("Baseline (àrea màscara / àrea total)")
    axes[1, 0].set_ylabel("Focus score (imatge original)")
    axes[1, 0].set_title("Focus vs baseline\nRoig = predicció canvia")
    axes[1, 0].legend(fontsize=8)

    # 4. Focus drop vs prob drop (scatter)
    axes[1, 1].scatter(prob_drop, focus_drop, c=colors, alpha=0.5,
                       edgecolors="none", s=20)
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].axvline(0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("Δ prob(cotxe)")
    axes[1, 1].set_ylabel("Caiguda Focus")
    axes[1, 1].set_title("Δ prob vs caiguda Focus\nRoig = predicció canvia")

    plt.tight_layout()
    path = output_dir / "xai_summary.png"
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    return path


# ---------------------------------------------------------------------------
# Selecció de casos representatius per a l'informe
# ---------------------------------------------------------------------------

def seleccionar_casos(resultats: list, metadata: list, top_n: int) -> list:
    """
    Selecciona top_n casos representatius per mostrar a l'informe:
      - Els millors per focus_drop (IG detecta clarament el cotxe)
      - Els millors per prob_drop (predicció canvia molt)
      - Alguns casos amb focus_drop negatiu (comportament paradoxal)

    Evita duplicats. Retorna les entrades de metadata corresponents.
    """
    meta_per_id = {Path(e["file_name"]).stem: e for e in metadata}

    per_focus = sorted(resultats, key=lambda r: r["focus_drop"], reverse=True)
    per_prob  = sorted(resultats, key=lambda r: r["prob_car_drop"], reverse=True)
    paradoxes = sorted(
        [r for r in resultats if r["focus_drop"] < -0.05],
        key=lambda r: r["focus_drop"]
    )

    vistos = set()
    seleccionats = []

    quota_focus    = max(1, top_n // 2)
    quota_prob     = max(1, top_n // 3)
    quota_paradox  = max(1, top_n // 6)

    for r in per_focus[:quota_focus]:
        if r["pair_id"] not in vistos and r["pair_id"] in meta_per_id:
            seleccionats.append((r, meta_per_id[r["pair_id"]]))
            vistos.add(r["pair_id"])

    for r in per_prob[:quota_prob]:
        if r["pair_id"] not in vistos and r["pair_id"] in meta_per_id:
            seleccionats.append((r, meta_per_id[r["pair_id"]]))
            vistos.add(r["pair_id"])

    for r in paradoxes[:quota_paradox]:
        if r["pair_id"] not in vistos and r["pair_id"] in meta_per_id:
            seleccionats.append((r, meta_per_id[r["pair_id"]]))
            vistos.add(r["pair_id"])

    return seleccionats[:top_n]


# ---------------------------------------------------------------------------
# Informe HTML lleuger
# ---------------------------------------------------------------------------

def img_b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generar_informe(resultats: list, summary_path: Path,
                    casos_figures: list,   # [(resultat, path_figura)]
                    output_dir: Path):

    n          = len(resultats)
    n_canvis   = sum(r["pred_changed"] for r in resultats)
    mean_fdrop = np.mean([r["focus_drop"]     for r in resultats])
    std_fdrop  = np.std( [r["focus_drop"]     for r in resultats])
    mean_pdrop = np.mean([r["prob_car_drop"]  for r in resultats])
    std_pdrop  = np.std( [r["prob_car_drop"]  for r in resultats])
    mean_forig = np.mean([r["focus_orig"]     for r in resultats])
    mean_finp  = np.mean([r["focus_inp"]      for r in resultats])
    mean_base  = np.mean([r["focus_baseline"] for r in resultats])
    above_base = sum(1 for r in resultats if r["focus_orig"] > r["focus_baseline"])

    summary_b64 = img_b64(summary_path)

    # Taula completa (sense imatges → lleugera)
    files_taula = ""
    for r in sorted(resultats, key=lambda x: x["focus_drop"], reverse=True):
        canvi_cls = "si" if r["pred_changed"] else "no"
        canvi_str = "✓" if r["pred_changed"] else "✗"
        files_taula += (
            f'<tr class="c{canvi_cls}">'
            f'<td>{r["pair_id"]}</td>'
            f'<td>{r["prob_car_orig"]:.3f}</td>'
            f'<td>{r["prob_car_inp"]:.3f}</td>'
            f'<td>{r["prob_car_drop"]:+.3f}</td>'
            f'<td>{r["focus_orig"]:.3f}</td>'
            f'<td>{r["focus_inp"]:.3f}</td>'
            f'<td><b>{r["focus_drop"]:+.3f}</b></td>'
            f'<td>{r["focus_baseline"]:.3f}</td>'
            f'<td class="c{canvi_cls}">{canvi_str}</td>'
            f'</tr>\n'
        )

    # Seccions dels casos representatius (amb figura)
    seccions = ""
    for r, fig_path in casos_figures:
        b64       = img_b64(fig_path)
        canvi_str = "✓ Predicció canvia" if r["pred_changed"] else "✗ Predicció no canvia"
        badge_cls = "si" if r["pred_changed"] else "no"

        if r["focus_drop"] > 0.05:
            interpretacio = (f"IG perd {r['focus_drop']:.3f} de Focus sobre la zona del cotxe. "
                             "El model mirava clarament el cotxe i deixa de fer-ho quan s'elimina.")
        elif r["focus_drop"] < -0.02:
            interpretacio = (f"Focus augmenta {abs(r['focus_drop']):.3f} sobre la zona inpaintada. "
                             "Possible artefacte: LaMa ha generat textura que activa features de cotxe.")
        else:
            interpretacio = ("Canvi de Focus mínim. El cotxe no era la característica "
                             "central per a la decisió del model en aquest cas.")

        seccions += f"""
<div class="cas">
  <div class="cas-cap">
    <h3>{r['pair_id']}</h3>
    <span class="badge b{badge_cls}">{canvi_str}</span>
  </div>
  <div class="nums">
    <span>prob orig: <b>{r['prob_car_orig']:.3f}</b></span>
    <span>prob inp: <b>{r['prob_car_inp']:.3f}</b></span>
    <span>Δ prob: <b>{r['prob_car_drop']:+.3f}</b></span>
    <span>Focus orig: <b>{r['focus_orig']:.3f}</b></span>
    <span>Focus inp: <b>{r['focus_inp']:.3f}</b></span>
    <span>Caiguda Focus: <b>{r['focus_drop']:+.3f}</b></span>
  </div>
  <p class="interp">{interpretacio}</p>
  <img src="data:image/png;base64,{b64}" style="width:100%">
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="ca">
<head>
<meta charset="UTF-8">
<title>Informe XAI — Integrated Gradients</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
     background:#f5f5f7;color:#1d1d1f;line-height:1.5}}
.wrap{{max-width:1100px;margin:0 auto;padding:2rem 1rem}}
h1{{font-size:1.8rem;font-weight:700;margin-bottom:.3rem}}
h2{{font-size:1.15rem;font-weight:600;margin:2rem 0 .8rem;
   border-bottom:2px solid #e0e0e0;padding-bottom:.3rem}}
h3{{font-size:1rem;font-weight:600}}
.sub{{color:#666;margin-bottom:1.5rem;font-size:.9rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
       gap:.8rem;margin:1rem 0}}
.card{{background:#fff;border-radius:10px;padding:.9rem 1.1rem;
       box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.val{{font-size:1.5rem;font-weight:700}}
.lbl{{font-size:.75rem;color:#888;margin-top:.15rem}}
.resum img{{width:100%;border-radius:10px;box-shadow:0 2px 8px rgba(0,0,0,.1)}}
.wrap-t{{overflow-x:auto;margin:.8rem 0}}
table{{width:100%;border-collapse:collapse;background:#fff;font-size:.82rem;
       border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
th{{background:#f0f0f5;padding:.5rem .7rem;text-align:left;font-weight:600;color:#555}}
td{{padding:.45rem .7rem;border-top:1px solid #f0f0f0}}
tr.csi td{{background:#fff8f8}}
td.csi{{color:#c0392b;font-weight:600}}
td.cno{{color:#888}}
.cas{{background:#fff;border-radius:12px;padding:1.3rem;margin:1.2rem 0;
      box-shadow:0 1px 6px rgba(0,0,0,.08)}}
.cas-cap{{display:flex;align-items:center;gap:.8rem;margin-bottom:.6rem}}
.badge{{padding:.2rem .6rem;border-radius:20px;font-size:.78rem;font-weight:600}}
.bsi{{background:#fde8e8;color:#c0392b}}
.bno{{background:#eef;color:#556}}
.nums{{display:flex;flex-wrap:wrap;gap:.4rem 1.2rem;
       font-size:.82rem;color:#555;margin-bottom:.5rem}}
.interp{{font-size:.85rem;color:#444;background:#f9f9fb;
          border-left:3px solid #0071e3;padding:.45rem .7rem;
          border-radius:0 6px 6px 0;margin-bottom:.7rem}}
footer{{margin-top:2.5rem;text-align:center;color:#aaa;font-size:.78rem}}
</style>
</head>
<body>
<div class="wrap">
<h1>Informe XAI — Integrated Gradients</h1>
<p class="sub">Model: ResNet-18 fine-tuned (binari cotxe/sense cotxe) &nbsp;·&nbsp;
Dataset: COCO 2017 train &nbsp;·&nbsp; {n} parells analitzats</p>

<h2>Mètriques globals</h2>
<div class="grid">
  <div class="card"><div class="val">{n}</div><div class="lbl">Parells analitzats</div></div>
  <div class="card"><div class="val">{n_canvis}/{n}</div><div class="lbl">Predicció canvia ({100*n_canvis/n:.0f}%)</div></div>
  <div class="card"><div class="val">{mean_forig:.3f}</div><div class="lbl">Focus mitjà original</div></div>
  <div class="card"><div class="val">{mean_finp:.3f}</div><div class="lbl">Focus mitjà inpaintat</div></div>
  <div class="card"><div class="val">{mean_base:.3f}</div><div class="lbl">Baseline aleatori</div></div>
  <div class="card"><div class="val">{mean_fdrop:+.3f}</div><div class="lbl">Caiguda Focus (±{std_fdrop:.3f})</div></div>
  <div class="card"><div class="val">{mean_pdrop:+.3f}</div><div class="lbl">Δ prob(cotxe) (±{std_pdrop:.3f})</div></div>
  <div class="card"><div class="val">{above_base}/{n}</div><div class="lbl">Focus orig &gt; baseline ({100*above_base/n:.0f}%)</div></div>
</div>

<h2>Figura resum</h2>
<div class="resum"><img src="data:image/png;base64,{summary_b64}" alt="Resum XAI"></div>

<h2>Taula de resultats — tots els parells</h2>
<div class="wrap-t">
<table>
<thead><tr>
  <th>Parell</th><th>prob orig</th><th>prob inp</th><th>Δ prob</th>
  <th>Focus orig</th><th>Focus inp</th><th>Δ Focus</th><th>Baseline</th><th>Canvi</th>
</tr></thead>
<tbody>{files_taula}</tbody>
</table>
</div>

<h2>Casos representatius ({len(casos_figures)} seleccionats)</h2>
<p class="sub">Selecció automàtica: millors per caiguda Focus, millors per Δ prob, i casos paradoxals (Focus puja però predicció canvia).</p>
{seccions}

<footer>Generat per xai_analysis.py &nbsp;·&nbsp; ResNet-18 fine-tuned &nbsp;·&nbsp; Integrated Gradients (Captum)</footer>
</div></body></html>"""

    path = output_dir / "informe.html"
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(model_path: Path, dataset_dir: Path, output_dir: Path,
        n_steps: int, device_str: str, top_n: int):

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Metadades
    with open(dataset_dir / "dataset_metadata.json") as f:
        metadata = json.load(f)

    parells = [
        e for e in metadata
        if Path(e["img_path"]).exists() and Path(e["inpainted_path"]).exists()
    ]
    print(f"Parells disponibles: {len(parells)}")

    # Model
    device = torch.device(device_str)
    model  = carregar_model(model_path, device)
    ig     = IntegratedGradients(model)
    print(f"Target IG: classe {IG_TARGET} (amb cotxe)\n")

    # ── Bucle principal: càlcul de mètriques (sense generar figures) ────────
    resultats = []

    for entry in tqdm(parells, desc="Calculant mètriques IG"):
        pair_id = Path(entry["file_name"]).stem
        try:
            t_orig, _ = carregar_imatge(Path(entry["img_path"]))
            t_inp,  _ = carregar_imatge(Path(entry["inpainted_path"]))
        except Exception as e:
            print(f"  Error carregant {pair_id}: {e}")
            continue

        t_orig, t_inp = t_orig.to(device), t_inp.to(device)

        with torch.no_grad():
            out_orig = model(t_orig)
            out_inp  = model(t_inp)

        probs_orig    = F.softmax(out_orig, dim=1)[0]
        probs_inp     = F.softmax(out_inp,  dim=1)[0]
        prob_car_orig = float(probs_orig[IG_TARGET])
        prob_car_inp  = float(probs_inp[IG_TARGET])
        pred_changed  = (prob_car_orig >= 0.5) != (prob_car_inp >= 0.5)

        try:
            attr_orig = mapa_ig(ig.attribute(t_orig, target=IG_TARGET, n_steps=n_steps))
            attr_inp  = mapa_ig(ig.attribute(t_inp,  target=IG_TARGET, n_steps=n_steps))
        except Exception as e:
            print(f"  Error IG {pair_id}: {e}")
            continue

        mask_raw = cv2.imread(entry["mask_path"], cv2.IMREAD_GRAYSCALE)
        mask_224 = (cv2.resize(mask_raw, (224, 224),
                               interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)

        f_orig     = focus_score(attr_orig, mask_224)
        f_inp      = focus_score(attr_inp,  mask_224)
        f_baseline = float(mask_224.astype(bool).mean())

        resultats.append({
            "pair_id":        pair_id,
            "prob_car_orig":  round(prob_car_orig, 4),
            "prob_car_inp":   round(prob_car_inp,  4),
            "prob_car_drop":  round(prob_car_orig - prob_car_inp, 4),
            "pred_changed":   pred_changed,
            "focus_orig":     round(f_orig,     4),
            "focus_inp":      round(f_inp,      4),
            "focus_baseline": round(f_baseline, 4),
            "focus_drop":     round(f_orig - f_inp, 4),
            "car_area_pct":   entry.get("car_area_pct"),
        })

    # Desar JSON
    with open(output_dir / "xai_results.json", "w") as f:
        json.dump(resultats, f, indent=2)

    # Figura resum
    summary_path = figura_resum(resultats, output_dir)

    # ── Generar figures NOMÉS dels casos seleccionats ───────────────────────
    meta_per_id  = {Path(e["file_name"]).stem: e for e in metadata}
    casos        = seleccionar_casos(resultats, metadata, top_n)

    print(f"\nGenerant figures dels {len(casos)} casos representatius...")
    casos_figures = []
    for r, entry in tqdm(casos, desc="Figures seleccionades"):
        try:
            fig_path = figura_parell(entry, model, ig, device, figures_dir)
            casos_figures.append((r, fig_path))
        except Exception as e:
            print(f"  Error figura {r['pair_id']}: {e}")

    # Informe HTML
    informe_path = generar_informe(resultats, summary_path, casos_figures, output_dir)

    # Resum
    n = len(resultats)
    canvis      = sum(r["pred_changed"]  for r in resultats)
    focus_drops = [r["focus_drop"]       for r in resultats]
    prob_drops  = [r["prob_car_drop"]    for r in resultats]

    print(f"\n{'='*55}")
    print(f"RESULTATS XAI — {n} parells")
    print(f"{'='*55}")
    print(f"  Predicció canvia:        {canvis}/{n} ({100*canvis/n:.0f}%)")
    print(f"  Caiguda Focus (mitjana): {np.mean(focus_drops):.3f} ± {np.std(focus_drops):.3f}")
    print(f"  Caiguda prob(cotxe):     {np.mean(prob_drops):.4f} ± {np.std(prob_drops):.4f}")
    print(f"\n  Informe HTML:  {informe_path}")
    print(f"  Figura resum:  {summary_path}")
    print(f"  JSON:          {output_dir / 'xai_results.json'}")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aplica IG al classificador binari i genera informe lleuger"
    )
    parser.add_argument("--model",       type=str, default="data/finetune/resnet18_car_classifier.pth")
    parser.add_argument("--dataset-dir", type=str, default="data/iopaint_dataset")
    parser.add_argument("--output-dir",  type=str, default="data/xai_results")
    parser.add_argument("--n-steps",     type=int, default=50)
    parser.add_argument("--device",      type=str, default="mps", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--top-n",       type=int, default=20,
                        help="Nombre de casos representatius a incloure a l'informe (default: 20)")
    args = parser.parse_args()

    run(
        model_path=Path(args.model),
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        n_steps=args.n_steps,
        device_str=args.device,
        top_n=args.top_n,
    )
