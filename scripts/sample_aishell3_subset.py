from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from tqdm.auto import tqdm

from evaluation.eval_utils import ensure_dir, load_aishell3_split


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample a test-only AISHELL-3 subset for evaluation.")
    parser.add_argument(
        "--data-root",
        default=str((Path(__file__).resolve().parents[1] / "datasets" / "aishell3").resolve()),
        help="AISHELL-3 root. Default: repo-local datasets/aishell3",
    )
    parser.add_argument("--train-transcript-file", default=None)
    parser.add_argument("--test-transcript-file", default=None)
    parser.add_argument(
        "--output-dir",
        default=str((Path(__file__).resolve().parents[1] / "datasets" / "aishell3_subset").resolve()),
        help="Directory to write sampled manifests and split metadata. Default: repo-local datasets/aishell3_subset",
    )
    parser.add_argument("--audio-ext", default=".wav")
    parser.add_argument(
        "--speaker-id-regex",
        default=r"^(.*?)(\d{4})$",
        help="Regex used to infer speaker id from utt id. The first non-empty capture group is used.",
    )
    parser.add_argument("--num-test-speakers", type=int, default=15)
    parser.add_argument("--test-utts-per-speaker", type=int, default=10)
    parser.add_argument(
        "--prompt-policy",
        default="first",
        choices=("first", "random", "longest"),
        help="How to pick the prompt utterance from the same speaker's train split.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def choose_prompt_item(items: list, *, policy: str, rng: random.Random):
    if policy == "random":
        return rng.choice(items)
    if policy == "longest":
        return max(items, key=lambda item: len(item.text))
    return sorted(items, key=lambda item: item.utt_id)[0]


def sample_eval_manifest(
    *,
    split_name: str,
    speakers: list[str],
    eval_items_by_speaker: dict[str, list],
    prompt_items_by_speaker: dict[str, list],
    utts_per_speaker: int,
    prompt_policy: str,
    rng: random.Random,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for speaker_id in tqdm(speakers, desc=f"sample-{split_name}", unit="spk"):
        eval_candidates = list(eval_items_by_speaker[speaker_id])
        prompt_candidates = list(prompt_items_by_speaker[speaker_id])
        if len(eval_candidates) < utts_per_speaker:
            raise ValueError(
                f"speaker {speaker_id} has only {len(eval_candidates)} {split_name} utterances, need {utts_per_speaker}"
            )
        if not prompt_candidates:
            raise ValueError(f"speaker {speaker_id} has no prompt utterances in train split")
        selected = rng.sample(eval_candidates, utts_per_speaker)
        prompt_item = choose_prompt_item(prompt_candidates, policy=prompt_policy, rng=rng)
        for item in sorted(selected, key=lambda row: row.utt_id):
            records.append(
                {
                    "split": split_name,
                    "utt_id": item.utt_id,
                    "speaker_id": item.speaker_id,
                    "text": item.text,
                    "target_audio_path": str(item.audio_path),
                    "prompt_audio_path": str(prompt_item.audio_path),
                    "prompt_text": prompt_item.text,
                }
            )
    return records


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    train_root = data_root / "train"
    test_root = data_root / "test"
    output_dir = ensure_dir(args.output_dir)

    print(f"[sample] loading train split from {train_root}")
    train_items = load_aishell3_split(
        split_root=train_root,
        transcript_file=args.train_transcript_file,
        audio_ext=args.audio_ext,
        speaker_id_regex=args.speaker_id_regex,
    )
    print(f"[sample] loaded {len(train_items)} train utterances")
    print(f"[sample] loading test split from {test_root}")
    test_items = load_aishell3_split(
        split_root=test_root,
        transcript_file=args.test_transcript_file,
        audio_ext=args.audio_ext,
        speaker_id_regex=args.speaker_id_regex,
    )
    print(f"[sample] loaded {len(test_items)} test utterances")

    train_items_by_speaker: dict[str, list] = defaultdict(list)
    test_items_by_speaker: dict[str, list] = defaultdict(list)
    for item in train_items:
        train_items_by_speaker[item.speaker_id].append(item)
    for item in test_items:
        test_items_by_speaker[item.speaker_id].append(item)

    eligible_speakers = sorted(
        speaker_id
        for speaker_id, train_rows in train_items_by_speaker.items()
        if len(train_rows) >= 1 and len(test_items_by_speaker.get(speaker_id, [])) >= args.test_utts_per_speaker
    )
    print(f"[sample] found {len(eligible_speakers)} eligible speakers")

    if len(eligible_speakers) < args.num_test_speakers:
        raise ValueError(f"num_test_speakers={args.num_test_speakers} exceeds eligible speakers={len(eligible_speakers)}")
    test_speakers = sorted(rng.sample(eligible_speakers, args.num_test_speakers))

    print("[sample] building test manifest")
    test_manifest = sample_eval_manifest(
        split_name="test",
        speakers=test_speakers,
        eval_items_by_speaker=test_items_by_speaker,
        prompt_items_by_speaker=train_items_by_speaker,
        utts_per_speaker=args.test_utts_per_speaker,
        prompt_policy=args.prompt_policy,
        rng=rng,
    )

    print(f"[sample] writing manifests to {output_dir}")
    write_jsonl(output_dir / "test.jsonl", test_manifest)

    summary = {
        "data_root": str(data_root),
        "seed": args.seed,
        "eligible_speakers": len(eligible_speakers),
        "test_speakers": test_speakers,
        "num_test_records": len(test_manifest),
        "test_utts_per_speaker": args.test_utts_per_speaker,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
