"""
finetune_resnet18.py
--------------------
Fine-tuning de ResNet-18 per a classificació binària:
    label 1  →  imatge original  (amb cotxe)
    label 0  →  imatge inpaintada (sense cotxe)

El dataset d'entrenament són directament els parells contrafactuals generats
pel pipeline_iopaint.py. Cada parell aporta un exemple positiu i un negatiu
de la mateixa escena, cosa que elimina qualsevol biaix de fons.

Divisió:
    80% dels PARELLS  →  entrenament  (orig + inp de cada parell)
    20% dels PARELLS  →  validació    (orig + inp de cada parell)

    Nota important: la divisió es fa per PARELLS, no per imatges individuals.
    Això evita que la mateixa escena aparegui als dos subconjunts (data leakage).

Fine-tuning:
    - Es congelen totes les capes excepte la última (layer4 + fc).
    - S'entrena la capa fc nova (512 → 2) des de zero.
    - Opcionalment es descongela layer4 per a un ajust més fi.

Ús:
    conda activate xai_env
    pip install torch torchvision tqdm matplotlib scikit-learn

    python finetune_resnet18.py
    python finetune_resnet18.py --epochs 20 --lr 1e-4 --unfreeze-layer4
"""

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CarDataset(Dataset):
    """
    Dataset binari construït a partir dels parells contrafactuals.

    Per a cada parell:
        (img_path,       label=1)  →  imatge original  (amb cotxe)
        (inpainted_path, label=0)  →  imatge inpaintada (sense cotxe)
    """

    # Transformació per a entrenament: augmentació lleugera
    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Transformació per a validació: sense augmentació
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, parells: list[dict], mode: str = "train"):
        """
        parells: llista d'entrades del dataset_metadata.json
        mode:    "train" o "val"
        """
        self.mode = mode
        self.transform = self.transform_train if mode == "train" else self.transform_val

        # Cada parell genera 2 exemples: original (1) i inpaintat (0)
        self.exemples = []
        for p in parells:
            self.exemples.append((Path(p["img_path"]),       1))  # amb cotxe
            self.exemples.append((Path(p["inpainted_path"]), 0))  # sense cotxe

    def __len__(self):
        return len(self.exemples)

    def __getitem__(self, idx):
        path, label = self.exemples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def construir_model(unfreeze_layer4: bool = False) -> nn.Module:
    """
    ResNet-18 preentrenat a ImageNet amb la capa fc substituïda per
    una sortida binària (2 classes: amb cotxe / sense cotxe).

    Estratègia de congelació:
        - Totes les capes es congelen (els pesos d'ImageNet es preserven).
        - La capa fc nova s'entrena des de zero.
        - Si unfreeze_layer4=True, també s'entrena layer4 (últim bloc convolucional).
          Recomanat quan es té >200 parells.
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Congelar totes les capes
    for param in model.parameters():
        param.requires_grad = False

    # Descongelar layer4 si es demana
    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    # Substituir la capa de classificació final
    # ResNet-18: fc té entrada 512, sortida 1000 (ImageNet)
    # Nosaltres volem sortida 2 (binari)
    model.fc = nn.Linear(512, 2)   # sempre entrenable (és nova)

    return model


# ---------------------------------------------------------------------------
# Entrenament i validació
# ---------------------------------------------------------------------------

def entrenar_epoca(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correctes, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        correctes  += (outputs.argmax(1) == labels).sum().item()
        total      += imgs.size(0)

    return total_loss / total, correctes / total


def validar(model, loader, criterion, device):
    model.eval()
    total_loss, correctes, total = 0.0, 0, 0
    totes_preds, totes_labels    = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs      = model(imgs)
            loss         = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            preds       = outputs.argmax(1)
            correctes  += (preds == labels).sum().item()
            total      += imgs.size(0)

            totes_preds.extend(preds.cpu().numpy())
            totes_labels.extend(labels.cpu().numpy())

    acc = correctes / total
    return total_loss / total, acc, totes_preds, totes_labels


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def guardar_corbes(historial: dict, output_dir: Path):
    """Guarda la figura de corbes d'accuracy i loss."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(historial["train_acc"]) + 1)

    axes[0].plot(epochs, historial["train_acc"], label="Train")
    axes[0].plot(epochs, historial["val_acc"],   label="Val")
    axes[0].set_title("Accuracy per època")
    axes[0].set_xlabel("Època"); axes[0].set_ylabel("Accuracy")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, historial["train_loss"], label="Train")
    axes[1].plot(epochs, historial["val_loss"],   label="Val")
    axes[1].set_title("Loss per època")
    axes[1].set_xlabel("Època"); axes[1].set_ylabel("Cross-entropy loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = output_dir / "training_curves.png"
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Corbes d'entrenament: {path}")


def guardar_matriu_confusio(preds, labels, output_dir: Path):
    """Guarda la matriu de confusió del conjunt de validació."""
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Sense cotxe (0)", "Amb cotxe (1)"]).plot(ax=ax)
    ax.set_title("Matriu de confusió — Validació")
    plt.tight_layout()
    path = output_dir / "confusion_matrix.png"
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"Matriu de confusió:   {path}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run(metadata_file: Path, output_dir: Path, epochs: int, lr: float,
        batch_size: int, unfreeze_layer4: bool, device_str: str):

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carregar metadades i dividir per parells (80/20)
    with open(metadata_file) as f:
        metadata = json.load(f)

    # Verificar que els fitxers existeixen
    metadata = [
        p for p in metadata
        if Path(p["img_path"]).exists() and Path(p["inpainted_path"]).exists()
    ]
    print(f"Parells disponibles: {len(metadata)}")

    # Divisió per PARELLS per evitar data leakage
    random.seed(42)
    random.shuffle(metadata)
    tall          = int(len(metadata) * 0.8)
    parells_train = metadata[:tall]
    parells_val   = metadata[tall:]

    print(f"  Train: {len(parells_train)} parells ({len(parells_train)*2} imatges)")
    print(f"  Val:   {len(parells_val)} parells ({len(parells_val)*2} imatges)")

    # 2. Datasets i DataLoaders
    ds_train = CarDataset(parells_train, mode="train")
    ds_val   = CarDataset(parells_val,   mode="val")

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=2)
    dl_val   = DataLoader(ds_val,   batch_size=batch_size, shuffle=False, num_workers=2)

    # 3. Model, loss i optimitzador
    device = torch.device(device_str)
    model  = construir_model(unfreeze_layer4=unfreeze_layer4).to(device)

    # Només s'optimitzen els paràmetres que requereixen gradient
    params_entrenables = [p for p in model.parameters() if p.requires_grad]
    print(f"\nParàmetres entrenables: {sum(p.numel() for p in params_entrenables):,}")
    print(f"  (layer4={unfreeze_layer4}, fc=True)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params_entrenables, lr=lr)

    # Scheduler: redueix lr si val_loss no millora en 3 èpoques
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5
    )

    # 4. Bucle d'entrenament
    print(f"\nEntrenant {epochs} èpoques  |  device={device_str}  |  lr={lr}")
    print("-" * 55)

    historial   = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    millor_acc  = 0.0
    millor_path = output_dir / "resnet18_car_classifier.pth"

    for epoca in range(1, epochs + 1):
        train_loss, train_acc = entrenar_epoca(model, dl_train, criterion, optimizer, device)
        val_loss,   val_acc, val_preds, val_labels = validar(model, dl_val, criterion, device)

        scheduler.step(val_loss)

        historial["train_loss"].append(train_loss)
        historial["train_acc"].append(train_acc)
        historial["val_loss"].append(val_loss)
        historial["val_acc"].append(val_acc)

        # Desar el millor model
        if val_acc > millor_acc:
            millor_acc = val_acc
            torch.save(model.state_dict(), millor_path)
            mark = " ← millor"
        else:
            mark = ""

        print(f"  Època {epoca:3d}/{epochs}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}{mark}")

    # 5. Avaluació final amb el millor model
    print(f"\nCarregant millor model ({millor_path.name})...")
    model.load_state_dict(torch.load(millor_path, map_location=device))
    _, final_acc, final_preds, final_labels = validar(model, dl_val, criterion, device)

    print(f"\n{'='*55}")
    print(f"RESULTATS FINALS — Conjunt de validació")
    print(f"{'='*55}")
    print(f"  Accuracy:  {final_acc:.4f}  ({final_acc*100:.1f}%)")
    print()
    print(classification_report(
        final_labels, final_preds,
        target_names=["Sense cotxe (0)", "Amb cotxe (1)"]
    ))

    # 6. Figures i historial
    guardar_corbes(historial, output_dir)
    guardar_matriu_confusio(final_preds, final_labels, output_dir)

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(historial, f, indent=2)

    print(f"\nModel guardat:  {millor_path}")
    print(f"Resultats a:    {output_dir}/")
    print(f"\nPròxim pas: aplicar Integrated Gradients sobre aquest model")
    print(f"  python xai_analysis.py --model {millor_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tuning ResNet-18 binari (amb cotxe / sense cotxe)"
    )
    parser.add_argument(
        "--metadata", type=str,
        default="data/iopaint_dataset/dataset_metadata.json",
        help="Fitxer de metadades generat per pipeline_iopaint.py"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/finetune",
        help="Directori on guardar el model i les figures"
    )
    parser.add_argument(
        "--epochs", type=int, default=15,
        help="Nombre d'èpoques (default: 15)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--unfreeze-layer4", action="store_true",
        help="Descongelar layer4 de ResNet-18 (recomanat amb >200 parells)"
    )
    parser.add_argument(
        "--device", type=str, default="mps",
        choices=["cpu", "cuda", "mps"],
        help="Device (mps = Apple Silicon)"
    )
    args = parser.parse_args()

    run(
        metadata_file=Path(args.metadata),
        output_dir=Path(args.output_dir),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        unfreeze_layer4=args.unfreeze_layer4,
        device_str=args.device,
    )
