# AISHELL-3 使用流程

这份文档汇总了当前仓库里和 AISHELL-3 相关的测试子集抽样与解码评测流程。

## 1. 目录约定

当前约定的目录如下：

- 完整 AISHELL-3 数据集：`datasets/aishell3`
- 抽样后的子集清单：`datasets/aishell3_subset`

当前子集目录中的核心文件：

- `datasets/aishell3_subset/test.jsonl`
- `datasets/aishell3_subset/summary.json`

## 2. 从 AISHELL-3 抽样子集

当前抽样逻辑是：

- 测试说话人：`15`
- 每个测试说话人的测试条数固定为 `N`

命令：

```bash
cd /inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano

python -m scripts.sample_aishell3_subset \
  --num-test-speakers 15 \
  --test-utts-per-speaker 10
```

说明：

- 默认从 `datasets/aishell3` 读取原始数据
- 默认写到 `datasets/aishell3_subset`
- 脚本自带阶段性打印和 `tqdm` 进度条

## 3. 从完整 AISHELL-3 生成全量评测 Manifest

如果你不是评测抽样子集，而是想基于完整 AISHELL-3 测试集生成评测清单，可以运行：

```bash
python -m scripts.prepare_aishell3_manifest \
  --target-root datasets/aishell3/test \
  --prompt-root datasets/aishell3/train \
  --output-manifest output/aishell3_eval_manifest.jsonl
```

## 4. 对抽样子集做测评

核心测评脚本是：

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

指标：

- `SIM`：说话人相似度，基于 speaker embedding cosine similarity
- `CER`：字符错误率，基于 Whisper 转写结果计算

### 4.1 Plain 模式测评

直接运行：

```bash
bash scripts/eval_aishell3_plain.sh
```

默认行为：

- 评测 `test.jsonl`
- 推理模式使用 `continuation`
- 合成时忽略 prompt 音频
- `SIM` 默认拿 `target_audio_path` 作为参考音频

如果 checkpoint 不在默认位置：

```bash
CHECKPOINT=./output/moss_tts_nano_sft_plain/checkpoint-last \
bash scripts/eval_aishell3_plain.sh
```

默认输出目录：

- `output/aishell3_plain_test_eval`

### 4.2 Ref 模式测评

直接运行：

```bash
bash scripts/eval_aishell3_ref.sh
```

默认行为：

- 评测 `test.jsonl`
- 推理模式使用 `voice_clone`
- 合成时使用 manifest 中的 `prompt_audio_path`

如果 checkpoint 不在默认位置：

```bash
CHECKPOINT=./output/moss_tts_nano_sft_ref/checkpoint-last \
bash scripts/eval_aishell3_ref.sh
```

默认输出目录：

- `output/aishell3_ref_test_eval`

## 5. 关键文件

评测公共代码：

- [evaluation/eval_utils.py](/inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano/evaluation/eval_utils.py:1)

主要脚本：

- [scripts/sample_aishell3_subset.py](/inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano/scripts/sample_aishell3_subset.py:1)
- [scripts/prepare_aishell3_manifest.py](/inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano/scripts/prepare_aishell3_manifest.py:1)
- [scripts/run_aishell3_eval.py](/inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano/scripts/run_aishell3_eval.py:1)
- [scripts/eval_aishell3_plain.sh](/inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano/scripts/eval_aishell3_plain.sh:1)
- [scripts/eval_aishell3_ref.sh](/inspire/hdd/global_user/gongjingjing-25039/ykzhou/workspace/MOSS-TTS-Nano/scripts/eval_aishell3_ref.sh:1)
