#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/output/moss_tts_nano_sft_ref/checkpoint-last}"
AUDIO_TOKENIZER_PATH="${AUDIO_TOKENIZER_PATH:-${REPO_ROOT}/models/MOSS-Audio-Tokenizer-Nano}"
SPEAKER_MODEL="${SPEAKER_MODEL:-microsoft/wavlm-base-plus-sv}"
ASR_MODEL="${ASR_MODEL:-openai/whisper-large-v3}"
ASR_LANGUAGE="${ASR_LANGUAGE:-zh}"

TEST_MANIFEST="${TEST_MANIFEST:-${REPO_ROOT}/datasets/aishell3_subset/test.jsonl}"

TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR:-${REPO_ROOT}/output/aishell3_ref_test_eval}"

COMMON_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --audio-tokenizer-pretrained-name-or-path "${AUDIO_TOKENIZER_PATH}"
  --speaker-model "${SPEAKER_MODEL}"
  --asr-model "${ASR_MODEL}"
  --asr-language "${ASR_LANGUAGE}"
  --mode voice_clone
  --skip-existing
)

echo "[eval-ref] checkpoint=${CHECKPOINT}"
echo "[eval-ref] running test evaluation"
python -m scripts.run_aishell3_eval \
  --manifest "${TEST_MANIFEST}" \
  --output-dir "${TEST_OUTPUT_DIR}" \
  "${COMMON_ARGS[@]}"
