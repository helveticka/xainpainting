# XAI Dataset Generation

Pipeline para la generación automática de pares contrafactuales fotorrealistes per a l'avaluació de mètodes d'Explainable AI (XAI).

## Descripció

El pipeline genera parells d'imatges (original / sense objecte) usant:
- **SAM 2** per a la segmentació precisa de l'objecte
- **LaMa** (inpainting local) o **Replicate API** (bria/eraser) per eliminar l'objecte
- Mètriques de qualitat: PCP, MAPD, SSIM

## Requisits previs

### 1. Entorns conda

```bash
# Entorn principal (SAM 2 + Replicate + mètriques)
conda env create -f envs/environment.yml
conda activate xai_env

# Entorn LaMa (només necessari per a --inpainting lama)
conda env create -f envs/lama_env.yml
conda activate lama_env
pip install scikit-learn requests tensorboard==2.11.0
pip install albumentations==1.1.0 imgaug==0.4.0
```

### 2. Submòdul LaMa

```bash
git submodule update --init --recursive
```

Aplicar pedaços de compatibilitat dins de `lama_env`:

```bash
conda activate lama_env

# Pedaç imgaug (incompatible amb numpy >= 1.24)
python3 -c "
import imgaug.imgaug as f, inspect, pathlib
path = pathlib.Path(inspect.getfile(f))
text = path.read_text()
text = text.replace(
    'NP_FLOAT_TYPES = set(np.sctypes[\"float\"])',
    'NP_FLOAT_TYPES = set([np.float16, np.float32, np.float64])'
)
path.write_text(text)
print('Pedaç imgaug aplicat correctament')
"

# Pedaç torch.load (PyTorch >= 2.6)
sed -i '' 's/torch.load(path, map_location=map_location)/torch.load(path, map_location=map_location, weights_only=False)/' models/lama/saicinpainting/training/trainers/__init__.py

# Pedaç refiner per a CPU (sense CUDA)
sed -i '' "s/gpu_ids = \[f'cuda:{gpuid}' for gpuid in gpu_ids.replace(\" \",\"\").split(\",\") if gpuid.isdigit()\]/gpu_ids = ['cpu']/" models/lama/saicinpainting/evaluation/refinement.py
```

### 3. Checkpoint SAM 2

```bash
mkdir -p checkpoints
curl -L -o checkpoints/sam2.1_hiera_small.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
```

### 4. Checkpoint LaMa (big-lama)

```bash
cd models/lama
curl -L "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip" -o big-lama.zip
unzip big-lama.zip && rm big-lama.zip
cd ../..
```

### 5. Anotacions COCO

```bash
mkdir -p data
curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip annotations/instances_val2017.json
mv annotations/instances_val2017.json data/instances_val2017.json
rm annotations_trainval2017.zip
```

## Ús

```bash
conda activate xai_env

# Amb LaMa local (recomanat)
export LAMA_PYTHON=/opt/homebrew/Caskroom/miniconda/base/envs/lama_env/bin/python3
python pipeline/pipeline.py --inpainting lama --n 15

# Amb Replicate API
export REPLICATE_API_TOKEN=el_teu_token
python pipeline/pipeline.py --inpainting replicate --n 15
```

> **Nota:** El path de `LAMA_PYTHON` pot variar segons la instal·lació de conda.  
> Troba'l amb: `conda activate lama_env && which python3`

## Resultats del pilot (15 imatges)

| Backend | Parells vàlids | SSIM mig | PCP exterior |
|---|---|---|---|
| Replicate (bria/eraser) | 6/15 | 0.9423 | 8.80% |
| LaMa (big-lama) | 9/15 | 0.9514 | 0.00% |

LaMa preserva el fons de manera perfecta (PCP exterior = 0%), cosa essencial per generar parells contrafactuals vàlids per a l'avaluació de mètodes XAI.

## Estructura del repositori

```
xai-dataset/
├── pipeline/
│   └── pipeline.py          # Script principal
├── envs/
│   ├── environment.yml      # Entorn xai_env (Python 3.10)
│   └── lama_env.yml         # Entorn lama_env (Python 3.9)
├── models/
│   └── lama/                # Submòdul LaMa (git submodule)
├── checkpoints/             # Checkpoint SAM 2 (no inclòs al Git)
├── data/                    # Imatges COCO i anotacions (no inclòs al Git)
└── output/                  # Parells generats (no inclòs al Git)
```

## Dependències externes (no incloses al repositori)

Els següents fitxers s'han de descarregar manualment (vegeu els passos anteriors):

| Fitxer | Mida aproximada | Font |
|---|---|---|
| `checkpoints/sam2.1_hiera_small.pt` | ~180 MB | Meta AI |
| `models/lama/big-lama/` | ~200 MB | Hugging Face |
| `data/instances_val2017.json` | ~25 MB | COCO Dataset |
| `data/coco_images/` | Variable | COCO Dataset (descarregat automàticament) |
