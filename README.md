# XAI Dataset Generation

Pipeline para la generación automática de pares contrafactuales fotorrealistas
para la evaluación de métodos de Explainable AI (XAI).

## Descripción
- Segmentación de objetos con SAM 2
- Inpainting con Replicate API (bria/eraser) o LaMa con refinamiento (local)
- Evaluación con métricas PCP, MAPD, SSIM

## Entorno
```bash
conda env create -f envs/environment.yml
conda activate xai_env
```

## Uso
```bash
python pipeline/pipeline.py --inpainting replicate
python pipeline/pipeline.py --inpainting lama