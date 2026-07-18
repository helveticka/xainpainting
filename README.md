# XAINPAINTING
**Fidelitat d'Integrated Gradients sobre parells contrafactuals**

Pipeline per generar parells d'imatges contrafactuals (amb cotxe / sense cotxe) i mesurar si les explicacions d'Integrated Gradients (IG) sobre un classificador binari són fidels a la decisió real del model.

## Resultat central

Sobre 410 parells generats (91% amb canvi de predicció en eliminar el cotxe):

| Mètrica | Valor |
|---|---|
| Caiguda mitjana de P(cotxe) — fidelitat *conductual* | +0.825 ± 0.267 |
| Caiguda mitjana de Focus — fidelitat *atribucional* | +0.023 ± 0.125 |

Eliminar l'objecte gairebé sempre canvia la predicció, però l'atribució d'IG amb prou feines es desplaça fora de la regió de l'objecte: una dissociació entre fidelitat conductual i atribucional. Detall metodològic i discussió a la memòria (`thesis/memoria.pdf`).

## Pipeline

```
COCO (1 cotxe, 10-30% àrea, seed=42)
  → màscara (anotació + dilatació)
  → IOPaint (LaMa) → parell contrafactual
  → fine-tuning ResNet-18 binari (amb cotxe / sense cotxe)
  → Integrated Gradients + Focus
```

El Focus es calcula sobre la regió *empírica* de diferència entre el parell (no la màscara COCO) i s'agrega només sobre els parells on canvia la predicció — són decisions metodològiques documentades als comentaris de `src/explanation.py` i a la memòria.

## Instal·lació

Per revisar la qualitat de les imatges, fer fine-tuning i l'anàlisi XAI sobre el dataset ja generat (`metrics`, `classification`, `explanation`), només fa falta un entorn:

```bash
python3 -m venv .venv && source .venv/bin/activate   # o conda create -n xai_env python=3.10
pip install -r requirements.txt
```

Regenerar el dataset des de zero (`generation.py`) també crida IOPaint com a procés extern, en un entorn a part perquè arrossega dependències pesades (gradio, diffusers) que no fan falta a la resta del codi:

```bash
./envs/setup.sh
export IOPAINT_PYTHON=$(conda run -n iopaint which python)
```

L'script crea l'entorn a partir d'`envs/iopaint.yml`, que reprodueix les versions exactes usades als resultats.

### Dades externes (no incloses al repositori)

```bash
mkdir -p data
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip annotations/instances_train2017.json
mv annotations/instances_train2017.json data/annotations/instances_train2017.json
rm -rf annotations_trainval2017.zip annotations
```

Les imatges COCO es descarreguen automàticament (via `coco_url`) durant `generation.py`.

## Reproducció

```bash
source .venv/bin/activate
python src/generation.py --ann-file data/annotations/instances_train2017.json --n 500
python src/metrics.py
python src/classification.py
python src/explanation.py
```

## Notebook

`notebooks/xainpainting.ipynb` recorre el pipeline sencer sobre un únic parell (selecció → màscara → inpainting → classificació → IG → Focus). És el punt d'entrada recomanat per inspeccionar cada pas sense regenerar tot el dataset.

## Estructura del repositori

```
xainpainting/
├── requirements.txt               # entorn principal
├── src/
│   ├── generation.py              # COCO → màscares → IOPaint
│   ├── metrics.py                 # PCP/MAPD/SSIM
│   ├── classification.py          # fine-tuning binari
│   └── explanation.py             # Integrated Gradients + Focus
├── notebooks/
│   └── pipeline_walkthrough.ipynb
├── thesis/
│   └── memoria.pdf                # memòria del TFG
└── envs/
    ├── iopaint.yml                # nomes per regenerar el dataset
    └── setup.sh                   # idem
```
