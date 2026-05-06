from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from tqdm.auto import tqdm

from evaluation.eval_utils import (
    choose_prompt_utterance,
    ensure_dir,
    load_aishell3_split,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a JSONL manifest for AISHELL-3 voice-cloning evaluation.")
    parser.add_argument("--target-root", required=True, help="Target split root, for example datasets/data_aishell3/test.")
    parser.add_argument(
        "--prompt-root",
        default=None,
        help="Prompt split root. Defaults to --target-root. You can point this to train/ so prompts and targets come from different splits.",
    )
    parser.add_argument("--target-transcript-file", default=None, help="Override transcript file for the target split.")
    parser.add_argument("--prompt-transcript-file", default=None, help="Override transcript file for the prompt split.")
    parser.add_argument("--output-manifest", required=True, help="Output JSONL manifest path.")
    parser.add_argument("--audio-ext", default=".wav", help="Audio suffix to index. Default: .wav")
    parser.add_argument(
        "--speaker-id-regex",
        default=r"^(.*?)(\d{4})$",
        help="Regex used to infer speaker id from utt id. The first non-empty capture group is used.",
    )
    parser.add_argument(
        "--prompt-policy",
        default="first",
        choices=("first", "random", "longest"),
        help="How to choose the prompt utterance for the same speaker.",
    )
    parser.add_argument("--min-text-chars", type=int, default=1, help="Skip target utterances shorter than this.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of items to write.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    target_root = Path(args.target_root).expanduser().resolve()
    prompt_root = Path(args.prompt_root).expanduser().resolve() if args.prompt_root else target_root
    output_manifest = Path(args.output_manifest).expanduser().resolve()
    ensure_dir(output_manifest.parent)

    print(f"[prepare] loading target split from {target_root}")
    target_items = load_aishell3_split(
        split_root=target_root,
        transcript_file=args.target_transcript_file,
        audio_ext=args.audio_ext,
        speaker_id_regex=args.speaker_id_regex,
    )
    print(f"[prepare] loaded {len(target_items)} target utterances")
    print(f"[prepare] loading prompt split from {prompt_root}")
    prompt_items = load_aishell3_split(
        split_root=prompt_root,
        transcript_file=args.prompt_transcript_file,
        audio_ext=args.audio_ext,
        speaker_id_regex=args.speaker_id_regex,
    )
    print(f"[prepare] loaded {len(prompt_items)} prompt utterances")

    prompts_by_speaker: dict[str, list] = defaultdict(list)
    for item in prompt_items:
        prompts_by_speaker[item.speaker_id].append(item)

    print(f"[prepare] writing manifest to {output_manifest}")
    written = 0
    skipped_missing_prompt = 0
    skipped_short_text = 0
    with output_manifest.open("w", encoding="utf-8") as handle:
        progress = tqdm(target_items, desc="prepare-manifest", unit="utt")
        for item in progress:
            if len(str(item.text).strip()) < args.min_text_chars:
                skipped_short_text += 1
                progress.set_postfix(written=written, skipped_missing_prompt=skipped_missing_prompt, refresh=False)
                continue
            prompt_item = choose_prompt_utterance(
                candidates=prompts_by_speaker.get(item.speaker_id, []),
                target_utt_id=item.utt_id if prompt_root == target_root else "",
                policy=args.prompt_policy,
                rng=rng,
            )
            if prompt_item is None:
                skipped_missing_prompt += 1
                progress.set_postfix(written=written, skipped_missing_prompt=skipped_missing_prompt, refresh=False)
                continue

            payload = {
                "utt_id": item.utt_id,
                "speaker_id": item.speaker_id,
                "text": item.text,
                "target_audio_path": str(item.audio_path),
                "prompt_audio_path": str(prompt_item.audio_path),
                "prompt_text": prompt_item.text,
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            written += 1
            progress.set_postfix(written=written, skipped_missing_prompt=skipped_missing_prompt, refresh=False)

            if args.limit is not None and written >= args.limit:
                break

    print("[prepare] manifest generation completed")
    print(
        json.dumps(
            {
                "output_manifest": str(output_manifest),
                "written": written,
                "target_items": len(target_items),
                "prompt_items": len(prompt_items),
                "skipped_missing_prompt": skipped_missing_prompt,
                "skipped_short_text": skipped_short_text,
                "target_root": str(target_root),
                "prompt_root": str(prompt_root),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
