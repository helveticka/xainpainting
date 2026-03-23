import os
import cv2
import torch
import numpy as np
from PIL import Image

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from diffusers import StableDiffusionInpaintPipeline


# -----------------------------
# CONFIG
# -----------------------------
INPUT_IMAGE = "scene_with_car.jpeg"
SAM_CHECKPOINT = "checkpoints/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

OUTPUT_MASK = "mask_auto.png"
OUTPUT_RESULT = "scene_without_car_proportional.png"

SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"

PROMPT = (
    "A photorealistic countryside landscape in daylight, natural lighting, "
    "realistic terrain and vegetation, no vehicles, preserve composition."
)
NEGATIVE_PROMPT = "car, vehicle, artifact, distortion"

GUIDANCE_SCALE = 7.5
STRENGTH = 0.25
STEPS = 30
SEED = 42


# -----------------------------
# UTILS
# -----------------------------
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resize_to_multiple_of_8(img: Image.Image):
    w, h = img.size
    new_w = (w // 8) * 8
    new_h = (h // 8) * 8
    resized = img.resize((new_w, new_h), Image.BICUBIC)
    return resized, (w, h)


# -----------------------------
# MAIN
# -----------------------------
def main():
    device = get_device()

    # -------- SAM (CPU) --------
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device="cpu")

    img_bgr = cv2.imread(INPUT_IMAGE)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16
    )

    masks = mask_generator.generate(img_rgb)

    # Heurística simple: máscara de tamaño medio
    img_area = h * w
    masks = sorted(
        masks,
        key=lambda m: abs((m["area"] / img_area) - 0.03)
    )

    mask = masks[0]["segmentation"]
    Image.fromarray((mask * 255).astype(np.uint8)).save(OUTPUT_MASK)

    # -------- Stable Diffusion --------
    image_orig = Image.open(INPUT_IMAGE).convert("RGB")
    mask_orig = Image.open(OUTPUT_MASK).convert("L")

    image_resized, original_size = resize_to_multiple_of_8(image_orig)
    mask_resized, _ = resize_to_multiple_of_8(mask_orig)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        SD_MODEL_ID,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(SEED)

    result_resized = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        image=image_resized,
        mask_image=mask_resized,
        guidance_scale=GUIDANCE_SCALE,
        strength=STRENGTH,
        num_inference_steps=STEPS,
        generator=generator
    ).images[0]

    result = result_resized.resize(original_size, Image.BICUBIC)
    result.save(OUTPUT_RESULT)

    print("[OK] Result saved:", OUTPUT_RESULT)


if __name__ == "__main__":
    main()