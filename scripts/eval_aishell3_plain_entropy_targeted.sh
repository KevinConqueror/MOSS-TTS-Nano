#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/models/MOSS-TTS-Nano}"
AUDIO_TOKENIZER_PATH="${AUDIO_TOKENIZER_PATH:-${REPO_ROOT}/models/MOSS-Audio-Tokenizer-Nano}"
SPEAKER_MODEL="${SPEAKER_MODEL:-microsoft/wavlm-base-plus-sv}"
ASR_MODEL="${ASR_MODEL:-openai/whisper-large-v3}"
ASR_LANGUAGE="${ASR_LANGUAGE:-zh}"
TEST_MANIFEST="${TEST_MANIFEST:-${REPO_ROOT}/datasets/aishell3_subset/test.jsonl}"

CONFIG_DIR="${CONFIG_DIR:-${REPO_ROOT}/configs/decoding}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/output/decoding_ablation/plain_entropy_targeted}"

run_eval() {
  local config_name="$1"
  local output_name="$2"
  local config_path="${CONFIG_DIR}/${config_name}"
  local output_dir="${OUTPUT_ROOT}/${output_name}"

  echo "[plain-entropy-targeted] config=${config_path}"
  echo "[plain-entropy-targeted] output=${output_dir}"
  python -m scripts.run_aishell3_eval \
    --manifest "${TEST_MANIFEST}" \
    --output-dir "${output_dir}" \
    --checkpoint "${CHECKPOINT}" \
    --audio-tokenizer-pretrained-name-or-path "${AUDIO_TOKENIZER_PATH}" \
    --speaker-model "${SPEAKER_MODEL}" \
    --asr-model "${ASR_MODEL}" \
    --asr-language "${ASR_LANGUAGE}" \
    --mode continuation \
    --ignore-prompt-audio \
    --decoding-config "${config_path}" \
    --skip-existing
}

run_eval "entropy_aware_decoding_v3_gentle_hysteresis_low_threshold.yaml" "entropy_v3_gentle_hysteresis_low_threshold"
run_eval "entropy_triggered_short_horizon_branching_low_threshold.yaml" "entropy_branching_low_threshold"
