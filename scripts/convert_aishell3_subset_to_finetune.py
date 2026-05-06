from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional, Sequence

from tqdm.auto import tqdm


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Convert sampled AISHELL-3 subset manifests into MOSS-TTS-Nano finetuning JSONL files.")
    parser.add_argument(
        "--subset-dir",
        default=str((repo_root / "datasets" / "aishell3_subset").resolve()),
        help="Subset directory containing train.jsonl / test_seen.jsonl / test_unseen.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default=str((repo_root / "datasets" / "aishell3_subset_finetune").resolve()),
        help="Directory to write finetuning-format JSONL files.",
    )
    parser.add_argument(
        "--train-ref-policy",
        default="first",
        choices=("first", "random", "longest"),
        help="How to choose same-speaker reference audio for train_ref.raw.jsonl.",
    )
    parser.add_argument("--language", default="zh", help="Language tag written into each record.")
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def choose_reference(items: list[dict[str, object]], *, exclude_utt_id: str, policy: str, rng: random.Random) -> dict[str, object]:
    candidates = [item for item in items if str(item["utt_id"]) != exclude_utt_id]
    if not candidates:
        raise ValueError(f"speaker sample {exclude_utt_id} has no alternate same-speaker reference audio")
    if policy == "random":
        return rng.choice(candidates)
    if policy == "longest":
        return max(candidates, key=lambda item: len(str(item.get("text") or "")))
    return sorted(candidates, key=lambda item: str(item["utt_id"]))[0]


def convert_train_plain(train_records: list[dict[str, object]], *, language: str) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for record in tqdm(train_records, desc="convert-train-plain", unit="utt"):
        converted.append(
            {
                "audio": str(record["audio_path"]),
                "text": str(record["text"]),
                "language": language,
                "speaker_id": str(record["speaker_id"]),
                "source_split": "train",
                "utt_id": str(record["utt_id"]),
            }
        )
    return converted


def convert_train_ref(
    train_records: list[dict[str, object]],
    *,
    language: str,
    policy: str,
    rng: random.Random,
) -> list[dict[str, object]]:
    by_speaker: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in train_records:
        by_speaker[str(record["speaker_id"])].append(record)

    converted: list[dict[str, object]] = []
    for record in tqdm(train_records, desc="convert-train-ref", unit="utt"):
        reference = choose_reference(
            by_speaker[str(record["speaker_id"])],
            exclude_utt_id=str(record["utt_id"]),
            policy=policy,
            rng=rng,
        )
        converted.append(
            {
                "audio": str(record["audio_path"]),
                "text": str(record["text"]),
                "ref_audio": str(reference["audio_path"]),
                "language": language,
                "speaker_id": str(record["speaker_id"]),
                "source_split": "train",
                "utt_id": str(record["utt_id"]),
                "ref_utt_id": str(reference["utt_id"]),
            }
        )
    return converted


def convert_eval_ref(records: list[dict[str, object]], *, split_name: str, language: str) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for record in tqdm(records, desc=f"convert-{split_name}", unit="utt"):
        converted.append(
            {
                "audio": str(record["target_audio_path"]),
                "text": str(record["text"]),
                "ref_audio": str(record["prompt_audio_path"]),
                "language": language,
                "speaker_id": str(record["speaker_id"]),
                "source_split": split_name,
                "utt_id": str(record["utt_id"]),
                "prompt_text": str(record.get("prompt_text") or ""),
            }
        )
    return converted


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rng = random.Random(args.seed)
    subset_dir = Path(args.subset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = subset_dir / "train.jsonl"
    test_seen_path = subset_dir / "test_seen.jsonl"
    test_unseen_path = subset_dir / "test_unseen.jsonl"

    print(f"[convert] loading subset manifests from {subset_dir}")
    train_records = read_jsonl(train_path)
    test_seen_records = read_jsonl(test_seen_path)
    test_unseen_records = read_jsonl(test_unseen_path)
    print(
        f"[convert] loaded train={len(train_records)} "
        f"test_seen={len(test_seen_records)} test_unseen={len(test_unseen_records)}"
    )

    print("[convert] converting train plain finetune JSONL")
    train_plain = convert_train_plain(train_records, language=args.language)
    print("[convert] converting train reference-conditioned finetune JSONL")
    train_ref = convert_train_ref(train_records, language=args.language, policy=args.train_ref_policy, rng=rng)
    print("[convert] converting seen-speaker test reference-conditioned JSONL")
    test_seen_ref = convert_eval_ref(test_seen_records, split_name="test_seen", language=args.language)
    print("[convert] converting unseen-speaker test reference-conditioned JSONL")
    test_unseen_ref = convert_eval_ref(test_unseen_records, split_name="test_unseen", language=args.language)

    output_paths = {
        "train_plain": output_dir / "train_plain.raw.jsonl",
        "train_ref": output_dir / "train_ref.raw.jsonl",
        "test_seen_ref": output_dir / "test_seen_ref.raw.jsonl",
        "test_unseen_ref": output_dir / "test_unseen_ref.raw.jsonl",
    }
    print(f"[convert] writing finetuning JSONL files to {output_dir}")
    write_jsonl(output_paths["train_plain"], train_plain)
    write_jsonl(output_paths["train_ref"], train_ref)
    write_jsonl(output_paths["test_seen_ref"], test_seen_ref)
    write_jsonl(output_paths["test_unseen_ref"], test_unseen_ref)

    summary = {
        "subset_dir": str(subset_dir),
        "output_dir": str(output_dir),
        "language": args.language,
        "train_ref_policy": args.train_ref_policy,
        "seed": args.seed,
        "num_train_plain_records": len(train_plain),
        "num_train_ref_records": len(train_ref),
        "num_test_seen_ref_records": len(test_seen_ref),
        "num_test_unseen_ref_records": len(test_unseen_ref),
        "files": {name: str(path) for name, path in output_paths.items()},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[convert] wrote summary to {output_dir / 'summary.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
