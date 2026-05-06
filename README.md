# 基于 MOSS-TTS-Nano 的解码侧改进实验

本项目基于 `MOSS-TTS-Nano` 做了面向中文 TTS 的解码侧改进与评测流程整理，围绕 **推理阶段的解码策略** 做系统化实验。

当前仓库主要用于：

- 在不改动主模型权重的前提下优化解码效果
- 比较不同解码策略对 CER 和 speaker similarity 的影响

## 项目简介

原始 `MOSS-TTS-Nano` 是一个轻量级自回归 TTS 模型。本仓库在其基础上新增了以下能力：

- AISHELL-3 测试集采样与评测脚本
- `plain` / `ref` 两种评测模式
- 基于 YAML 的解码参数配置
- 多种解码侧改进策略
- 批量消融实验脚本
- 自动 CER / speaker similarity 评测

本项目关注的问题包括：

1. 模型在中文测试集上的解码稳定性
2. 重复生成、发散生成和长尾坏例的抑制
3. 不同采样约束和熵感知策略的效果差异

## 当前实现的改进内容

目前仓库中已经实现的主要策略包括：

- `baseline`
  当前默认解码策略
- `text_greedy_audio_sampling`
  `text` 分支 greedy，`audio` 分支继续采样
- `conservative_audio`
  收紧 `audio` 侧采样参数
- `adaptive_antidegeneration`
  自适应反退化解码
- `repetition_aware_logit_shaping`
  重复感知的 logit shaping
- `entropy_aware_decoding`
  熵感知解码
- `entropy_aware_decoding_v2 / v3 / v4`
  不同强度与阈值的熵感知配置
- `entropy_aware_decoding_*_hysteresis`
  带迟滞机制的熵感知解码
- `entropy_triggered_short_horizon_branching`
  高熵触发的短视野分支策略

这些配置文件统一位于：

[`configs/decoding/`](./configs/decoding)

## 目录结构

- [`infer.py`](./infer.py)
  主推理入口，支持从 YAML 读取解码参数
- [`evaluation/eval_utils.py`](./evaluation/eval_utils.py)
  CER / speaker similarity 计算逻辑
- [`scripts/run_aishell3_eval.py`](./scripts/run_aishell3_eval.py)
  AISHELL-3 单次评测入口
- [`scripts/sample_aishell3_subset.py`](./scripts/sample_aishell3_subset.py)
  AISHELL-3 测试子集采样脚本
- [`scripts/eval_aishell3_plain_decoding_ablation.sh`](./scripts/eval_aishell3_plain_decoding_ablation.sh)
  `plain` 模式批量解码策略消融
- [`scripts/eval_aishell3_plain_entropy_ablation.sh`](./scripts/eval_aishell3_plain_entropy_ablation.sh)
  熵感知系列批量实验
- [`scripts/eval_aishell3_plain_entropy_targeted.sh`](./scripts/eval_aishell3_plain_entropy_targeted.sh)
  低阈值定向熵实验
- [`models/MOSS-TTS-Nano/modeling_moss_tts_nano.py`](./models/MOSS-TTS-Nano/modeling_moss_tts_nano.py)
  当前所有解码改进逻辑所在的核心文件

## 环境配置

建议使用独立 Conda 环境：

```bash
conda create -n moss-tts-nano python=3.12 -y
conda activate moss-tts-nano

pip install -r requirements.txt
pip install -e .
```

如果 `WeTextProcessing` 或 `pynini` 安装失败，建议优先先安装 `pynini`，再安装其余依赖。

## 数据准备

当前仓库默认只关注 **测试集评测**，不要求先构造训练集。

如果已经准备好 AISHELL-3 原始数据，可以先采样测试子集：

```bash
python -m scripts.sample_aishell3_subset \
  --num-test-speakers 15 \
  --test-utts-per-speaker 10
```

默认输出：

```bash
datasets/aishell3_subset/test.jsonl
```

## 单次评测用法

### `plain` 模式

`plain` 模式表示只根据文本合成，不使用参考音频作为输入。

```bash
python -m scripts.run_aishell3_eval \
  --manifest datasets/aishell3_subset/test.jsonl \
  --output-dir output/plain_baseline_eval \
  --checkpoint ./models/MOSS-TTS-Nano \
  --audio-tokenizer-pretrained-name-or-path ./models/MOSS-Audio-Tokenizer-Nano \
  --speaker-model microsoft/wavlm-base-plus-sv \
  --asr-model openai/whisper-large-v3 \
  --asr-language zh \
  --mode continuation \
  --ignore-prompt-audio \
  --decoding-config configs/decoding/baseline.yaml \
  --skip-existing
```

### `ref` 模式

`ref` 模式表示使用参考音频执行 voice clone。

```bash
python -m scripts.run_aishell3_eval \
  --manifest datasets/aishell3_subset/test.jsonl \
  --output-dir output/ref_baseline_eval \
  --checkpoint ./models/MOSS-TTS-Nano \
  --audio-tokenizer-pretrained-name-or-path ./models/MOSS-Audio-Tokenizer-Nano \
  --speaker-model microsoft/wavlm-base-plus-sv \
  --asr-model openai/whisper-large-v3 \
  --asr-language zh \
  --mode voice_clone \
  --decoding-config configs/decoding/baseline.yaml \
  --skip-existing
```

## 批量消融实验

### 1. 通用解码策略消融

```bash
CHECKPOINT=./models/MOSS-TTS-Nano bash scripts/eval_aishell3_plain_decoding_ablation.sh
```

### 2. 熵感知系列消融

```bash
CHECKPOINT=./models/MOSS-TTS-Nano bash scripts/eval_aishell3_plain_entropy_ablation.sh
```

### 3. 定向低阈值熵实验

```bash
CHECKPOINT=./models/MOSS-TTS-Nano bash scripts/eval_aishell3_plain_entropy_targeted.sh
```

## 常用配置文件

常用配置包括：

- [`configs/decoding/baseline.yaml`](./configs/decoding/baseline.yaml)
- [`configs/decoding/conservative_audio.yaml`](./configs/decoding/conservative_audio.yaml)
- [`configs/decoding/repetition_aware_logit_shaping.yaml`](./configs/decoding/repetition_aware_logit_shaping.yaml)
- [`configs/decoding/entropy_aware_decoding.yaml`](./configs/decoding/entropy_aware_decoding.yaml)
- [`configs/decoding/entropy_aware_decoding_v2.yaml`](./configs/decoding/entropy_aware_decoding_v2.yaml)
- [`configs/decoding/entropy_aware_decoding_v2_hysteresis.yaml`](./configs/decoding/entropy_aware_decoding_v2_hysteresis.yaml)
- [`configs/decoding/entropy_aware_decoding_v3_gentle.yaml`](./configs/decoding/entropy_aware_decoding_v3_gentle.yaml)
- [`configs/decoding/entropy_aware_decoding_v3_gentle_hysteresis.yaml`](./configs/decoding/entropy_aware_decoding_v3_gentle_hysteresis.yaml)
- [`configs/decoding/entropy_triggered_short_horizon_branching.yaml`](./configs/decoding/entropy_triggered_short_horizon_branching.yaml)

你也可以直接复制这些 YAML，自定义新的实验参数。

## 指标说明

当前评测主要使用两个指标：

- `CER`
  字符错误率，越小越好
- `SIM`
  说话人相似度，越大越好

## 说明

本仓库包含的很多脚本和配置，是在原始 `MOSS-TTS-Nano` 基础上为中文 AISHELL-3 评测、解码改进和实验复现而整理出的二次开发版本。如果你需要查看原始项目说明、官方模型介绍或原版部署方式，请参考原始 `MOSS-TTS-Nano` 发布页和模型仓库。
