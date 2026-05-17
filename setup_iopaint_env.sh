#!/bin/bash
# setup_iopaint_env.sh
# --------------------
# Crea el entorno conda para IOPaint y verifica la instalación.
# Ejecutar UNA SOLA VEZ antes de correr el pipeline.
#
# Uso:
#   chmod +x setup_iopaint_env.sh
#   ./setup_iopaint_env.sh

set -e  # salir si cualquier comando falla

ENV_NAME="iopaint_env"
PYTHON_VERSION="3.10"

echo "========================================"
echo " Setup IOPaint environment"
echo "========================================"

# ── Crear entorno conda ──────────────────────────────────────────────────────
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[OK] Entorno '${ENV_NAME}' ya existe, saltando creación."
else
    echo "[1/3] Creando entorno conda '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

# ── Instalar IOPaint ─────────────────────────────────────────────────────────
echo "[2/3] Instalando IOPaint en '${ENV_NAME}'..."
conda run -n "${ENV_NAME}" pip install iopaint

# ── Verificar instalación ────────────────────────────────────────────────────
echo "[3/3] Verificando instalación..."
IOPAINT_PYTHON=$(conda run -n "${ENV_NAME}" which python)

echo ""
echo "  Python del entorno: ${IOPAINT_PYTHON}"
conda run -n "${ENV_NAME}" python -c "import iopaint; print(f'  IOPaint version: {iopaint.__version__}')"
conda run -n "${ENV_NAME}" python -m iopaint --help | head -5

echo ""
echo "========================================"
echo " Instalación completada."
echo ""
echo " Para usar el pipeline, exporta esta variable:"
echo "   export IOPAINT_PYTHON=${IOPAINT_PYTHON}"
echo ""
echo " O añádela a tu ~/.zshrc / ~/.bashrc para que persista:"
echo "   echo 'export IOPAINT_PYTHON=${IOPAINT_PYTHON}' >> ~/.zshrc"
echo ""
echo " Luego ejecuta:"
echo "   conda activate xai_env"
echo "   python pipeline_iopaint.py --n 20 --device mps"
echo "========================================"
