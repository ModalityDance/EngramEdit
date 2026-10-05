#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON=${PYTHON:-python}
MODEL_NAME=${MODEL_NAME:-data/LongCat-Flash-Lite}
RUN_DIR=${RUN_DIR:-results/mcf}
DATASET_SIZE_LIMIT=${DATASET_SIZE_LIMIT:-2000}
PARAPHRASE_PATH=${PARAPHRASE_PATH:-data/mcf_longcat_before_subject.jsonl}
NGRAM_FREQUENCY_CACHE=${NGRAM_FREQUENCY_CACHE:-data/ngram_frequency/LongCat-Flash-Lite_mcf2000_para4.pt}

"${PYTHON}" -m experiments.evaluate \
  --model_name "${MODEL_NAME}" \
  --ds_name mcf \
  --dataset_size_limit "${DATASET_SIZE_LIMIT}" \
  --num_edits 100 \
  --paraphrase_path "${PARAPHRASE_PATH}" \
  --paraphrase_append_count 4 \
  --ngram_frequency_cache "${NGRAM_FREQUENCY_CACHE}" \
  --downstream_eval_steps 5 \
  --run_dir "${RUN_DIR}" \
  "$@"
