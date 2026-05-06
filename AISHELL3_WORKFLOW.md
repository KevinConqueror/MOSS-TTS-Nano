# AISHELL-3 Workflow

This document summarizes the AISHELL-3 test-subset sampling and decoding-time evaluation workflow used in this repository.

## 1. Directory Layout

Expected paths:

- Full AISHELL-3 dataset: `datasets/aishell3`
- Sampled subset manifests: `datasets/aishell3_subset`

Current subset outputs:

- `datasets/aishell3_subset/test.jsonl`
- `datasets/aishell3_subset/summary.json`

## 2. Sample an AISHELL-3 Subset

This samples:

- `15` test speakers
- a fixed `N` eval utterances per test speaker

Command:

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano

python -m scripts.sample_aishell3_subset \
  --num-test-speakers 15 \
  --test-utts-per-speaker 10
```

Notes:

- The script uses `datasets/aishell3` by default.
- The script writes to `datasets/aishell3_subset` by default.
- The script prints stage logs and uses `tqdm`.

## 3. Prepare a Full Evaluation Manifest from Raw AISHELL-3

If you want a full test-set manifest instead of using the sampled subset:

```bash
python -m scripts.prepare_aishell3_manifest \
  --target-root datasets/aishell3/test \
  --prompt-root datasets/aishell3/train \
  --output-manifest output/aishell3_eval_manifest.jsonl
```

## 4. Evaluate a Model on the Sampled Subset

The core evaluation entrypoint is:

```bash
python -m scripts.run_aishell3_eval \
  --manifest <manifest.jsonl> \
  --output-dir <output_dir> \
  --checkpoint <checkpoint_dir> \
  --audio-tokenizer-pretrained-name-or-path ./models/MOSS-Audio-Tokenizer-Nano \
  --speaker-model microsoft/wavlm-base-plus-sv \
  --asr-model openai/whisper-large-v3 \
  --asr-language zh
```

Metrics:

- `SIM`: speaker embedding cosine similarity
- `CER`: character error rate computed from Whisper transcription

### 4.1 Plain Evaluation

Wrapper script:

```bash
bash scripts/eval_aishell3_plain.sh
```

Default behavior:

- evaluates `test.jsonl`
- uses `mode continuation`
- ignores prompt audio during synthesis
- uses `target_audio_path` as the SIM reference

Override checkpoint:

```bash
CHECKPOINT=./output/moss_tts_nano_sft_plain/checkpoint-last \
bash scripts/eval_aishell3_plain.sh
```

Default output dirs:

- `output/aishell3_plain_test_eval`

### 4.2 Reference-Conditioned Evaluation

Wrapper script:

```bash
bash scripts/eval_aishell3_ref.sh
```

Default behavior:

- evaluates `test.jsonl`
- uses `mode voice_clone`
- uses manifest `prompt_audio_path` during synthesis

Override checkpoint:

```bash
CHECKPOINT=./output/moss_tts_nano_sft_ref/checkpoint-last \
bash scripts/eval_aishell3_ref.sh
```

Default output dirs:

- `output/aishell3_ref_test_eval`

## 5. Important Files

Evaluation helpers:

- [evaluation/eval_utils.py](/inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano/evaluation/eval_utils.py:1)

Scripts:

- [scripts/sample_aishell3_subset.py](/inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano/scripts/sample_aishell3_subset.py:1)
- [scripts/prepare_aishell3_manifest.py](/inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano/scripts/prepare_aishell3_manifest.py:1)
- [scripts/run_aishell3_eval.py](/inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano/scripts/run_aishell3_eval.py:1)
- [scripts/eval_aishell3_plain.sh](/inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano/scripts/eval_aishell3_plain.sh:1)
- [scripts/eval_aishell3_ref.sh](/inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano/scripts/eval_aishell3_ref.sh:1)
