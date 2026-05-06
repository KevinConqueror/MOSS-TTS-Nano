#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RAW_JSONL="${RAW_JSONL:-${REPO_ROOT}/datasets/aishell3_subset_finetune/train_plain.raw.jsonl}"
PREPARED_JSONL="${PREPARED_JSONL:-${REPO_ROOT}/datasets/aishell3_subset_finetune/train_plain.with_codes.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/moss_tts_nano_sft_plain}"

MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/models/MOSS-TTS-Nano}"
CODEC_PATH="${CODEC_PATH:-${REPO_ROOT}/models/MOSS-Audio-Tokenizer-Nano}"
PREP_DEVICE="${PREP_DEVICE:-auto}"

PREP_ACCELERATE_ARGS_STR="${PREP_ACCELERATE_ARGS_STR:-}"
TRAIN_ACCELERATE_ARGS_STR="${TRAIN_ACCELERATE_ARGS_STR:-}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-}"
TRAIN_EXTRA_ARGS_STR="${TRAIN_EXTRA_ARGS_STR:---per-device-batch-size 1 --gradient-accumulation-steps 8 --learning-rate 1e-5 --num-epochs 3 --mixed-precision bf16 --max-length 1024 --channelwise-loss-weight 1,32}"

echo "[train-plain] repo_root=${REPO_ROOT}"
echo "[train-plain] raw_jsonl=${RAW_JSONL}"
echo "[train-plain] prepared_jsonl=${PREPARED_JSONL}"
echo "[train-plain] output_dir=${OUTPUT_DIR}"
echo "[train-plain] using plain TTS finetuning without ref_audio"

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
PREP_EXTRA_ARGS_STR="--skip-reference-audio-codes" \
TRAIN_EXTRA_ARGS_STR="${TRAIN_EXTRA_ARGS_STR}" \
bash "${REPO_ROOT}/finetuning/run_train.sh"
