# Evaluation Scripts

This directory contains a lightweight AISHELL-3 evaluation pipeline for `MOSS-TTS-Nano`.

These scripts use `tqdm` to report progress during manifest preparation and evaluation.

## 0. Sample a Test Subset

If you do not want to evaluate on the full AISHELL-3 test split, you can first sample a controlled subset.

The script below matches this setup:

- test speakers: `15`
- eval utterances per test speaker: fixed count, for example `10`

Command:

```bash
python -m scripts.sample_aishell3_subset \
  --num-test-speakers 15 \
  --test-utts-per-speaker 10
```

By default, it reads from `datasets/aishell3` and writes to `datasets/aishell3_subset`.

Outputs:

- `datasets/aishell3_subset/test.jsonl`
- `datasets/aishell3_subset/summary.json`

`test.jsonl` can be fed directly into `python -m scripts.run_aishell3_eval`.

## 1. Prepare a Manifest

Use a target split and a prompt split to build voice-cloning pairs:

```bash
python -m scripts.prepare_aishell3_manifest \
  --target-root datasets/aishell3/test \
  --prompt-root datasets/aishell3/train \
  --output-manifest output/aishell3_eval_manifest.jsonl
```

If your local AISHELL-3 layout differs, override transcript files with:

```bash
--target-transcript-file /path/to/content.txt
--prompt-transcript-file /path/to/content.txt
```

Each line in the manifest looks like:

```json
{
  "utt_id": "SSB00050001",
  "speaker_id": "SSB0005",
  "text": "卡尔普陪外孙玩滑梯。",
  "target_audio_path": "/abs/path/to/target.wav",
  "prompt_audio_path": "/abs/path/to/prompt.wav",
  "prompt_text": "这是提示音频对应的文本。"
}
```

## 2. Run Evaluation

The evaluation script generates audio with the repository's PyTorch inference path, then computes:

- `SIM`: cosine similarity between speaker embeddings
- `CER`: character error rate between reference text and Whisper transcription

Example:

```bash
python -m scripts.run_aishell3_eval \
  --manifest output/aishell3_eval_manifest.jsonl \
  --output-dir output/aishell3_eval \
  --checkpoint ./models/MOSS-TTS-Nano \
  --audio-tokenizer-pretrained-name-or-path ./models/MOSS-Audio-Tokenizer-Nano \
  --speaker-model microsoft/wavlm-base-plus-sv \
  --asr-model openai/whisper-large-v3 \
  --asr-language zh \
  --skip-existing
```

Outputs:

- `output/aishell3_eval/audio/*.wav`: generated waveforms
- `output/aishell3_eval/results.jsonl`: per-utterance metrics and errors
- `output/aishell3_eval/summary.json`: aggregated `SIM` and `CER`
