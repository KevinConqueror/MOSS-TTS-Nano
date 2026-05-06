#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RAW_JSONL="${RAW_JSONL:-${REPO_ROOT}/datasets/aishell3_subset_finetune/train_ref.raw.jsonl}"
PREPARED_JSONL="${PREPARED_JSONL:-${REPO_ROOT}/datasets/aishell3_subset_finetune/train_ref.with_codes.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/moss_tts_nano_sft_ref_lora}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/MOSS-TTS-Nano}"
CODEC_PATH="${CODEC_PATH:-${REPO_ROOT}/models/MOSS-Audio-Tokenizer-Nano}"
PREP_DEVICE="${PREP_DEVICE:-auto}"

PREP_ACCELERATE_ARGS_STR="${PREP_ACCELERATE_ARGS_STR:-}"
TRAIN_ACCELERATE_ARGS_STR="${TRAIN_ACCELERATE_ARGS_STR:-}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-}"
LORA_EXTRA_ARGS_STR="${LORA_EXTRA_ARGS_STR:---use-lora --save-merged-lora-checkpoint --lora-r 16 --lora-alpha 32 --lora-dropout 0.05}"
TRAIN_EXTRA_ARGS_STR="${TRAIN_EXTRA_ARGS_STR:---per-device-batch-size 1 --gradient-accumulation-steps 8 --learning-rate 1e-4 --num-epochs 5 --mixed-precision bf16 --max-length 1024 --channelwise-loss-weight 1,32}"

echo "[train-ref-lora] repo_root=${REPO_ROOT}"
echo "[train-ref-lora] raw_jsonl=${RAW_JSONL}"
echo "[train-ref-lora] prepared_jsonl=${PREPARED_JSONL}"
echo "[train-ref-lora] output_dir=${OUTPUT_DIR}"

RAW_JSONL="${RAW_JSONL}" \
PREPARED_JSONL="${PREPARED_JSONL}" \
TRAIN_JSONL="${TRAIN_JSONL}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
MODEL_PATH="${MODEL_PATH}" \
CODEC_PATH="${CODEC_PATH}" \
PREP_DEVICE="${PREP_DEVICE}" \
PREP_ACCELERATE_ARGS_STR="${PREP_ACCELERATE_ARGS_STR}" \
TRAIN_ACCELERATE_ARGS_STR="${TRAIN_ACCELERATE_ARGS_STR}" \
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE}" \
TRAIN_EXTRA_ARGS_STR="${TRAIN_EXTRA_ARGS_STR} ${LORA_EXTRA_ARGS_STR}" \
bash "${REPO_ROOT}/finetuning/run_train.sh"
