#!/bin/bash
# Offline Augmentation Runner

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="python" 

# 증강 스크립트 경로
AUG_SCRIPT="${PROJECT_ROOT}/src/data/augmentation.py"
CSV_PATH="${PROJECT_ROOT}/dataset/roboflow/info.csv" # CSV 파일 경로

IMG_SIZE=640 

echo "========================================"
echo "Offline Augmentation Start"
echo "Project root : ${PROJECT_ROOT}"
echo "CSV path     : ${CSV_PATH}"
echo "Image size   : ${IMG_SIZE}"
echo "========================================"

# 증강 실행
PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_BIN}" "${AUG_SCRIPT}" \
  --csv_path "${CSV_PATH}" \
  --img_size ${IMG_SIZE} \
  --save_dir "${PROJECT_ROOT}/dataset/roboflow/train"

EXIT_CODE=$?

if [ ${EXIT_CODE} -ne 0 ]; then
  echo "❌ Offline augmentation failed (exit code=${EXIT_CODE})"
  exit ${EXIT_CODE}
else
  echo "✅ Offline augmentation completed successfully"
fi