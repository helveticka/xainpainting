import os
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from diffusers import StableDiffusionInpaintPipeline


# -----------------------------
# CONFIG
# -----------------------------
INPUT_IMAGE = "scene_with_car.jpeg"
SAM_CHECKPOINT = "checkpoints/sam_vit_b_01ec64.pth"   # ajusta si usas otro
SAM_MODEL_TYPE = "vit_b"                               # "vit_h" | "vit_l" | "vit_b"

OUTPUT_MASK = "mask_car.png"
OUTPUT_OVERLAY = "overlay_mask.png"
OUTPUT_INPAINT = "scene_without_car.png"

# Heurística para escoger el "coche" entre las máscaras detectadas por SAM:
# - suele estar en la mitad inferior (carretera/suelo)
# - tamaño medio (ni enorme como el cielo, ni minúsculo como detalles)
# Puedes ajustar estos rangos si tu escena es distinta.
MIN_AREA_RATIO = 0.002   # 0.2% del área de la imagen
MAX_AREA_RATIO = 0.20    # 20% del área de la imagen
PREFERRED_Y_CENTER_MIN = 0.45  # priorizar máscaras cuyo centro esté en la mitad inferior

# Stable Diffusion Inpainting
SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"
PROMPT = (
    "A photorealistic countryside landscape in daylight, natural lighting, "
    "realistic terrain and vegetation, no vehicles, preserve composition."
)
NEGATIVE_PROMPT = (
    "car, vehicle, wheels, bumper, headlights, artifact, blur, distortion, "
    "unrealistic texture, lighting change, extra objects"
)

GUIDANCE_SCALE = 7.5
STRENGTH = 0.25          # bajo = cambios mínimos fuera del coche
NUM_STEPS = 30
SEED = 42


# -----------------------------
# UTILS
# -----------------------------
def device_select():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def save_mask_png(mask_bool: np.ndarray, path: str):
    mask_u8 = (mask_bool.astype(np.uint8) * 255)
    Image.fromarray(mask_u8, mode="L").save(path)


def overlay_mask(image_rgb: np.ndarray, mask_bool: np.ndarray, out_path: str):
    overlay = image_rgb.copy()
    # Pintar máscara en rojo
    overlay[mask_bool] = (255, 0, 0)
    blended = (0.65 * image_rgb + 0.35 * overlay).astype(np.uint8)
    Image.fromarray(blended).save(out_path)


def choose_best_mask(masks, h, w):
    img_area = h * w
    best = None
    best_score = -1e9

    for m in masks:
        seg = m["segmentation"]  # bool mask
        area = m["area"]
        area_ratio = area / img_area

        # Filtros por tamaño para evitar cielo/terreno global y detalles minúsculos
        if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
            continue

        ys, xs = np.where(seg)
        if len(xs) == 0:
            continue

        x_center = xs.mean() / w
        y_center = ys.mean() / h

        # Heurística: preferir centro en parte baja
        y_bonus = 1.0 if y_center >= PREFERRED_Y_CENTER_MIN else 0.0

        # Preferir máscaras relativamente compactas (menos dispersas)
        x_span = (xs.max() - xs.min()) / w
        y_span = (ys.max() - ys.min()) / h
        compactness_penalty = (x_span * y_span)  # bbox area ratio (más grande = menos compacto)

        # Score: centro bajo + tamaño medio - penalizar dispersión
        # Ajustable si hace falta
        score = (2.0 * y_bonus) + (1.0 - abs(area_ratio - 0.03)) - 0.5 * compactness_penalty

        if score > best_score:
            best_score = score
            best = seg

    if best is None:
        raise RuntimeError(
            "SAM no encontró una máscara candidata con los filtros actuales. "
            "Prueba a relajar MIN_AREA_RATIO/MAX_AREA_RATIO."
        )
    return best


# -----------------------------
# MAIN
# -----------------------------
def main():
    device = device_select()
    print(f"[INFO] Device: {device}")
    sam_device = "cpu" #degut a usar Apple

    if not os.path.exists(INPUT_IMAGE):
        raise FileNotFoundError(f"Falta {INPUT_IMAGE}")

    if not os.path.exists(SAM_CHECKPOINT):
        raise FileNotFoundError(
            f"Falta el checkpoint de SAM: {SAM_CHECKPOINT}\n"
            f"Descárgalo y colócalo ahí, o actualiza la ruta en el script."
        )

    # 1) Cargar imagen
    img_rgb = load_rgb(INPUT_IMAGE)
    h, w, _ = img_rgb.shape
    print(f"[INFO] Image size: {w}x{h}")

    # 2) SAM: generar máscaras automáticas
    print("[INFO] Loading SAM...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device=sam_device)

    # Parámetros del generador: puedes tocar points_per_side si quieres más/menos máscaras
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100
    )

    print("[INFO] Generating masks with SAM...")
    masks = mask_generator.generate(img_rgb)
    print(f"[INFO] Masks found: {len(masks)}")

    # 3) Escoger la máscara del coche (heurística)
    print("[INFO] Selecting best candidate mask (heuristic)...")
    car_mask = choose_best_mask(masks, h, w)

    # 4) Guardar máscara y overlay para verificación
    save_mask_png(car_mask, OUTPUT_MASK)
    overlay_mask(img_rgb, car_mask, OUTPUT_OVERLAY)
    print(f"[OK] Saved mask: {OUTPUT_MASK}")
    print(f"[OK] Saved overlay preview: {OUTPUT_OVERLAY}")

    # 5) Inpainting con Stable Diffusion usando la máscara de SAM
    print("[INFO] Loading Stable Diffusion Inpaint pipeline...")
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=dtype
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(SEED)

    image_pil = Image.open(INPUT_IMAGE).convert("RGB")
    mask_pil = Image.open(OUTPUT_MASK).convert("L")

    print("[INFO] Running inpainting...")
    result = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=image_pil,
        mask_image=mask_pil,
        guidance_scale=GUIDANCE_SCALE,
        strength=STRENGTH,
        num_inference_steps=NUM_STEPS,
        generator=generator
    ).images[0]

    result.save(OUTPUT_INPAINT)
    print(f"[OK] Saved inpaint result: {OUTPUT_INPAINT}")

    # 6) Mostrar rápidamente el overlay (opcional)
    try:
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.imshow(Image.open(OUTPUT_OVERLAY))
        ax1.set_title("Mask overlay (SAM)")
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.imshow(Image.open(OUTPUT_INPAINT))
        ax2.set_title("Inpaint result (no car)")
        ax2.axis("off")

        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()