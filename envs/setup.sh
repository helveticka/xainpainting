#!/bin/bash
# Crea l'entorn conda "iopaint" des d'iopaint.yml. Executar una vegada.
set -e

ENV_NAME="iopaint"
ENV_FILE="$(cd "$(dirname "$0")" && pwd)/iopaint.yml"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "L'entorn '${ENV_NAME}' ja existeix."
else
    conda env create -f "${ENV_FILE}"
fi

IOPAINT_PYTHON=$(conda run -n "${ENV_NAME}" which python)
conda run -n "${ENV_NAME}" python -c "import iopaint; print('IOPaint', iopaint.__version__)"

echo ""
echo "Entorn llest. Abans d'executar el pipeline:"
echo "  export IOPAINT_PYTHON=${IOPAINT_PYTHON}"
