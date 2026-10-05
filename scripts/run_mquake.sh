#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON=${PYTHON:-python}
MODEL_NAME=${MODEL_NAME:-data/LongCat-Flash-Lite}
RUN_DIR=${RUN_DIR:-results/mquake}
DATASET_SIZE_LIMIT=${DATASET_SIZE_LIMIT:-3000}
PARAPHRASE_PATH=${PARAPHRASE_PATH:-data/paraphrases/mquake_longcat_before_subject.jsonl}
NGRAM_FREQUENCY_CACHE=${NGRAM_FREQUENCY_CACHE:-data/ngram_frequency/LongCat-Flash-Lite_mquake3000_para4.pt}

"${PYTHON}" -m experiments.evaluate \
  --model_name "${MODEL_NAME}" \
  --ds_name mquake \
  --dataset_size_limit "${DATASET_SIZE_LIMIT}" \
  --num_edits 100 \
  --paraphrase_path "${PARAPHRASE_PATH}" \
  --paraphrase_append_count 4 \
  --ngram_frequency_cache "${NGRAM_FREQUENCY_CACHE}" \
  --downstream_eval_steps 0 \
  --run_dir "${RUN_DIR}" \
  "$@"
