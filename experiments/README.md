# Experiments preliminars

Exploracions inicials de mètodes de detecció/segmentació i inpainting abans
d'arribar al pipeline final (SAM2 + LaMa/IOPaint, a `src/`):

- `remove_object_sam_only.py` — SAM sense priors, heurística per àrea.
- `remove_object_gorundingdino_sam.py` — GroundingDINO (detecció per text) + SAM.
- `remove_car_sam_inpaint.py` — SAM + Stable Diffusion inpaint.

No formen part del pipeline reproduïble ni s'utilitzen als resultats de la memòria.
Es conserven com a registre del procés.
