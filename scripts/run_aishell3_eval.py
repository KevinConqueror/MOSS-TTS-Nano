from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional, Sequence

import torch
from tqdm.auto import tqdm

from evaluation.eval_utils import (
    SpeakerSimilarityMetric,
    WhisperCerMetric,
    append_jsonl,
    compute_cer,
    ensure_dir,
    mean_or_none,
    read_jsonl,
    resolve_torch_device,
    resolve_torch_dtype,
    write_json,
)

from infer import load_model, resolve_sampling_kwargs, set_logging  # noqa: E402
from moss_tts_nano.defaults import DEFAULT_AUDIO_TOKENIZER_PATH, DEFAULT_CHECKPOINT_PATH  # noqa: E402
from text_normalization_pipeline import WeTextProcessingManager, prepare_tts_request_texts  # noqa: E402


MOSS_AUDIO_TOKENIZER_TYPE = "moss-audio-tokenizer-nano"
DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AISHELL-3 evaluation for MOSS-TTS-Nano with SIM and CER metrics.")
    parser.add_argument("--manifest", required=True, help="JSONL created by scripts/prepare_aishell3_manifest.py")
    parser.add_argument("--output-dir", required=True, help="Directory for generated audio and evaluation outputs.")

    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--text-tokenizer-path", default=None)
    parser.add_argument(
        "--audio-tokenizer-pretrained-name-or-path",
        default=str(DEFAULT_AUDIO_TOKENIZER_PATH),
    )
    parser.add_argument("--mode", default="voice_clone", choices=("voice_clone", "continuation"))
    parser.add_argument(
        "--ignore-prompt-audio",
        action="store_true",
        help="Ignore manifest prompt_audio_path/prompt_text and evaluate as plain TTS.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--nq", type=int, default=None)
    parser.add_argument("--max-new-frames", type=int, default=375)
    parser.add_argument("--voice-clone-max-text-tokens", type=int, default=75)
    parser.add_argument("--voice-clone-max-memory-per-sample-gb", type=float, default=1.0)
    parser.add_argument("--do-sample", type=int, nargs="?", const=1, default=None, choices=[0, 1])
    parser.add_argument("--text-do-sample", type=int, nargs="?", const=1, default=None, choices=[0, 1])
    parser.add_argument("--audio-do-sample", type=int, nargs="?", const=1, default=None, choices=[0, 1])
    parser.add_argument("--text-temperature", type=float, default=None)
    parser.add_argument("--text-top-p", type=float, default=None)
    parser.add_argument("--text-top-k", type=int, default=None)
    parser.add_argument("--audio-temperature", type=float, default=None)
    parser.add_argument("--audio-top-p", type=float, default=None)
    parser.add_argument("--audio-top-k", type=int, default=None)
    parser.add_argument("--audio-repetition-penalty", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--top-k", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--top-p", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--repetition-penalty", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--decoding-config", default=str(DEFAULT_REPO_ROOT / "configs" / "decoding" / "default.yaml"))
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--enable-wetext-processing", type=int, nargs="?", const=1, default=1, choices=[0, 1])
    parser.add_argument("--disable-wetext-processing", action="store_true")
    parser.add_argument("--enable-normalize-tts-text", action="store_true", default=True)
    parser.add_argument("--disable-normalize-tts-text", action="store_true")

    parser.add_argument("--speaker-model", default="microsoft/wavlm-base-plus-sv")
    parser.add_argument("--asr-model", default="openai/whisper-large-v3")
    parser.add_argument("--asr-language", default="zh")
    parser.add_argument("--metrics-device", default="auto")
    parser.add_argument("--metrics-dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--results-jsonl", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument(
        "--exclude-utt-id",
        action="append",
        default=None,
        help="Utterance id to exclude from summary aggregation. Can be passed multiple times.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only recompute summary from existing results.jsonl without running inference.",
    )
    return parser.parse_args(argv)


def build_summary(records: list[dict[str, object]], args: argparse.Namespace) -> dict[str, object]:
    excluded_utt_ids = {utt_id for utt_id in (args.exclude_utt_id or []) if utt_id}
    included_records = [record for record in records if str(record.get("utt_id") or "") not in excluded_utt_ids]
    excluded_records = [record for record in records if str(record.get("utt_id") or "") in excluded_utt_ids]
    successful = [record for record in included_records if record.get("status") == "ok"]
    failed = [record for record in included_records if record.get("status") != "ok"]
    return {
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "num_total_records": len(included_records),
        "num_success": len(successful),
        "num_failed": len(failed),
        "mean_sim": mean_or_none(record.get("sim") for record in successful),
        "mean_cer": mean_or_none(record.get("cer") for record in successful),
        "num_excluded_records": len(excluded_records),
        "excluded_utt_ids": sorted(excluded_utt_ids),
        "speaker_model": args.speaker_model,
        "asr_model": args.asr_model,
        "mode": args.mode,
        "ignore_prompt_audio": bool(args.ignore_prompt_audio),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    set_logging()
    args = parse_args(argv)
    output_dir = ensure_dir(args.output_dir)
    audio_output_dir = ensure_dir(output_dir / "audio")
    results_jsonl = Path(args.results_jsonl).expanduser().resolve() if args.results_jsonl else output_dir / "results.jsonl"
    summary_json = Path(args.summary_json).expanduser().resolve() if args.summary_json else output_dir / "summary.json"

    items = read_jsonl(args.manifest)
    if args.limit is not None:
        items = items[: args.limit]
    print(f"[eval] loaded {len(items)} manifest items from {Path(args.manifest).expanduser().resolve()}")

    if args.summary_only:
        if not results_jsonl.is_file():
            raise FileNotFoundError(f"results file not found for --summary-only: {results_jsonl}")
        all_records = read_jsonl(results_jsonl)
        summary = build_summary(all_records, args=args)
        write_json(summary_json, summary)
        print(f"[eval] wrote summary to {summary_json}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    existing_records_by_utt_id: dict[str, dict[str, object]] = {}
    if args.skip_existing and results_jsonl.is_file():
        print(f"[eval] loading existing results from {results_jsonl}")
        for record in read_jsonl(results_jsonl):
            utt_id = str(record.get("utt_id") or "")
            if utt_id:
                existing_records_by_utt_id[utt_id] = record

    device = resolve_torch_device(args.device)
    dtype = resolve_torch_dtype(args.dtype, device)
    metrics_device = resolve_torch_device(args.metrics_device)
    metrics_dtype = resolve_torch_dtype(args.metrics_dtype, metrics_device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print(f"[eval] loading TTS model from {args.checkpoint}")
    model = load_model(args.checkpoint, device=device, dtype=dtype)
    sampling_kwargs = resolve_sampling_kwargs(args)

    enable_wetext_processing = bool(args.enable_wetext_processing) and not bool(args.disable_wetext_processing)
    enable_normalize_tts_text = bool(args.enable_normalize_tts_text) and not bool(args.disable_normalize_tts_text)
    text_normalizer_manager = None
    if enable_wetext_processing:
        print("[eval] initializing WeTextProcessing")
        text_normalizer_manager = WeTextProcessingManager()
        snapshot = text_normalizer_manager.ensure_ready()
        if not snapshot.ready:
            raise RuntimeError(snapshot.error or snapshot.message)

    print(f"[eval] loading ASR model from {args.asr_model}")
    cer_metric = WhisperCerMetric(
        model_name_or_path=args.asr_model,
        device=metrics_device,
        dtype=metrics_dtype,
        language=args.asr_language,
    )
    print(f"[eval] loading speaker similarity model from {args.speaker_model}")
    sim_metric = SpeakerSimilarityMetric(
        model_name_or_path=args.speaker_model,
        device=metrics_device,
    )

    print(f"[eval] writing audio to {audio_output_dir}")
    print(f"[eval] writing per-utterance results to {results_jsonl}")
    newly_written_records: list[dict[str, object]] = []
    progress = tqdm(items, desc="aishell3-eval", unit="utt")
    for index, item in enumerate(progress, start=1):
        utt_id = str(item.get("utt_id") or f"sample_{index:06d}")
        if args.skip_existing and utt_id in existing_records_by_utt_id:
            logging.info("skip existing utt_id=%s", utt_id)
            progress.set_postfix(skipped="existing", utt_id=utt_id, refresh=False)
            continue

        raw_text = str(item["text"])
        raw_prompt_text = "" if args.ignore_prompt_audio else str(item.get("prompt_text") or "")
        prepared_texts = prepare_tts_request_texts(
            text=raw_text,
            prompt_text=raw_prompt_text,
            voice="",
            enable_wetext=enable_wetext_processing,
            enable_normalize_tts_text=enable_normalize_tts_text,
            text_normalizer_manager=text_normalizer_manager,
        )
        normalized_text = str(prepared_texts["text"])
        normalized_prompt_text = None if args.ignore_prompt_audio else str(prepared_texts["prompt_text"]).strip() or None
        prompt_audio_path = None if args.ignore_prompt_audio else (
            str(item["prompt_audio_path"]) if item.get("prompt_audio_path") not in (None, "") else None
        )
        if args.mode == "voice_clone" and not prompt_audio_path:
            raise ValueError("voice_clone mode requires prompt_audio_path unless --ignore-prompt-audio is not used")
        target_audio_path = str(item["target_audio_path"])
        output_audio_path = audio_output_dir / f"{utt_id}.wav"

        started_at = time.time()
        try:
            inference_kwargs = {
                "text": normalized_text,
                "output_audio_path": str(output_audio_path),
                "mode": args.mode,
                "prompt_audio_path": prompt_audio_path,
                "reference_audio_path": prompt_audio_path,
                "text_tokenizer_path": args.text_tokenizer_path,
                "audio_tokenizer_type": MOSS_AUDIO_TOKENIZER_TYPE,
                "audio_tokenizer_pretrained_name_or_path": args.audio_tokenizer_pretrained_name_or_path,
                "device": device,
                "nq": args.nq,
                "max_new_frames": args.max_new_frames,
                "voice_clone_max_text_tokens": args.voice_clone_max_text_tokens,
                "voice_clone_max_memory_per_sample_gb": args.voice_clone_max_memory_per_sample_gb,
                "use_kv_cache": True,
                **sampling_kwargs,
            }
            if args.mode != "voice_clone":
                inference_kwargs["prompt_text"] = normalized_prompt_text
            model.inference(**inference_kwargs)
            transcription = cer_metric.transcribe(output_audio_path)
            cer = compute_cer(raw_text, transcription)
            sim = sim_metric.similarity(target_audio_path, output_audio_path)
            record = {
                "utt_id": utt_id,
                "speaker_id": item.get("speaker_id"),
                "status": "ok",
                "text": raw_text,
                "normalized_text": normalized_text,
                "prompt_audio_path": prompt_audio_path,
                "target_audio_path": target_audio_path,
                "generated_audio_path": str(output_audio_path),
                "asr_transcription": transcription,
                "cer": cer,
                "sim": sim,
                "elapsed_seconds": time.time() - started_at,
            }
            logging.info(
                "done %d/%d utt_id=%s sim=%.4f cer=%.4f audio=%s",
                index,
                len(items),
                utt_id,
                sim,
                cer,
                output_audio_path,
            )
            progress.set_postfix(status="ok", utt_id=utt_id, sim=f"{sim:.4f}", cer=f"{cer:.4f}", refresh=False)
        except Exception as exc:  # noqa: BLE001
            record = {
                "utt_id": utt_id,
                "speaker_id": item.get("speaker_id"),
                "status": "failed",
                "text": raw_text,
                "prompt_audio_path": prompt_audio_path,
                "generated_audio_path": str(output_audio_path),
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": time.time() - started_at,
            }
            logging.exception("failed %d/%d utt_id=%s", index, len(items), utt_id)
            progress.set_postfix(status="failed", utt_id=utt_id, refresh=False)

        append_jsonl(results_jsonl, record)
        newly_written_records.append(record)

    all_records = read_jsonl(results_jsonl) if results_jsonl.is_file() else newly_written_records
    summary = build_summary(all_records, args=args)
    write_json(summary_json, summary)
    print(f"[eval] wrote summary to {summary_json}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
