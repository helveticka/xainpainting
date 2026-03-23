import sys
import os

GROUNDING_DINO_PATH = os.path.join(os.path.dirname(__file__), "GroundingDINO")
sys.path.append(GROUNDING_DINO_PATH)

import cv2
import torch
import numpy as np
from PIL import Image

from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline


# -----------------------------
# CONFIG
# -----------------------------
IMAGE_PATH = "scene_with_car.jpeg"

# GroundingDINO
DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
TEXT_PROMPT = "car"
BOX_THRESHOLD = 0.3
TEXT_THRESHOLD = 0.25

# SAM
SAM_CHECKPOINT = "checkpoints/sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

# Stable Diffusion
SD_MODEL_ID = "runwayml/stable-diffusion-inpainting"
OUTPUT_MASK = "mask_car.png"
OUTPUT_RESULT = "scene_without_car.png"

PROMPT = (
    "A photorealistic countryside landscape in daylight, natural lighting, "
    "realistic terrain and vegetation, no vehicles, preserve composition."
)
NEGATIVE_PROMPT = "car, vehicle, artifact"

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

    # -------- GroundingDINO --------
    model = load_model(DINO_CONFIG, DINO_CHECKPOINT)
    image_source, image = load_image(IMAGE_PATH)

    boxes, _, _ = predict(
        model=model,
        image=image,
        caption=TEXT_PROMPT,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device="cpu"
    )

    if len(boxes) == 0:
        raise RuntimeError("No object detected")

    h, w, _ = image_source.shape
    box = boxes[0] * torch.tensor([w, h, w, h])
    box = box.cpu().numpy().astype(int)

    # -------- SAM (CPU) --------
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=SAM_CHECKPOINT)
    sam.to(device="cpu")
    predictor = SamPredictor(sam)

    predictor.set_image(image_source)
    masks, _, _ = predictor.predict(
        box=box[None, :],
        multimask_output=False
    )

    mask = masks[0]
    Image.fromarray((mask * 255).astype(np.uint8)).save(OUTPUT_MASK)

    # -------- Stable Diffusion --------
    image_orig = Image.open(IMAGE_PATH).convert("RGB")
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