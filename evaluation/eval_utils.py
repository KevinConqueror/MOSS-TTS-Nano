from __future__ import annotations

import json
import random
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def append_jsonl(path: str | Path, record: dict[str, object]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, payload: dict[str, object]) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def load_audio_mono(path: str | Path, target_sample_rate: int):
    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(Path(path).expanduser().resolve()))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sample_rate)
    return waveform


def resolve_torch_device(device_arg: str):
    import torch

    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_torch_dtype(dtype_arg: str, device):
    import torch

    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cuda":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def normalize_eval_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).strip().lower()
    kept_chars: list[str] = []
    for char in normalized:
        category = unicodedata.category(char)
        if category.startswith("P") or category.startswith("Z") or category.startswith("C"):
            continue
        kept_chars.append(char)
    return "".join(kept_chars)


def edit_distance(ref: str, hyp: str) -> int:
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    previous = list(range(len(hyp) + 1))
    for ref_index, ref_char in enumerate(ref, start=1):
        current = [ref_index]
        for hyp_index, hyp_char in enumerate(hyp, start=1):
            substitution_cost = 0 if ref_char == hyp_char else 1
            current.append(
                min(
                    previous[hyp_index] + 1,
                    current[hyp_index - 1] + 1,
                    previous[hyp_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def compute_cer(reference_text: str, hypothesis_text: str) -> float:
    normalized_reference = normalize_eval_text(reference_text)
    normalized_hypothesis = normalize_eval_text(hypothesis_text)
    if not normalized_reference:
        return 0.0 if not normalized_hypothesis else 1.0
    return edit_distance(normalized_reference, normalized_hypothesis) / max(len(normalized_reference), 1)


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


@dataclass(frozen=True)
class Aishell3Utterance:
    utt_id: str
    speaker_id: str
    text: str
    audio_path: Path


def iter_audio_files(root: Path, audio_ext: str) -> Iterator[Path]:
    suffix = audio_ext.lower()
    for path in sorted(root.rglob(f"*{suffix}")):
        if path.is_file():
            yield path.resolve()


def build_audio_index(root: Path, audio_ext: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in iter_audio_files(root, audio_ext=audio_ext):
        index.setdefault(path.stem, path)
    return index


def infer_speaker_id(utt_id: str, speaker_id_regex: str) -> str:
    match = re.match(speaker_id_regex, utt_id)
    if match:
        for group_index in range(1, len(match.groups()) + 1):
            value = match.group(group_index)
            if value:
                return value
    return utt_id


def parse_transcript_line(line: str) -> Optional[tuple[str, str]]:
    stripped = line.strip()
    if not stripped:
        return None

    if "|" in stripped:
        parts = [part.strip() for part in stripped.split("|")]
    elif "\t" in stripped:
        parts = [part.strip() for part in stripped.split("\t")]
    else:
        parts = stripped.split(maxsplit=1)

    if len(parts) < 2:
        return None

    utt_id = parts[0]
    nonempty_tail = [part for part in parts[1:] if part]
    if not nonempty_tail:
        return None
    raw_text = nonempty_tail[-1]
    tokens = [token for token in re.split(r"\s+", raw_text) if token]
    cjk_tokens = [token for token in tokens if re.search(r"[\u3400-\u9fff]", token)]
    if cjk_tokens:
        text = "".join(cjk_tokens)
    else:
        text = raw_text
    return utt_id, text


def find_default_transcript_file(split_root: Path) -> Path:
    candidates = [
        split_root / "content.txt",
        split_root / "label_train-set.txt",
        split_root / "label_test-set.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    matches = sorted(split_root.rglob("content.txt"))
    if matches:
        return matches[0].resolve()
    raise FileNotFoundError(f"could not find transcript file under {split_root}")


def load_aishell3_split(
    *,
    split_root: str | Path,
    transcript_file: str | Path | None,
    audio_ext: str,
    speaker_id_regex: str,
) -> list[Aishell3Utterance]:
    split_path = Path(split_root).expanduser().resolve()
    transcript_path = (
        Path(transcript_file).expanduser().resolve() if transcript_file is not None else find_default_transcript_file(split_path)
    )
    audio_index = build_audio_index(split_path, audio_ext=audio_ext)

    utterances: list[Aishell3Utterance] = []
    with transcript_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            parsed = parse_transcript_line(raw_line)
            if parsed is None:
                continue
            utt_id, text = parsed
            utt_id = Path(utt_id).stem
            audio_path = audio_index.get(utt_id)
            if audio_path is None:
                continue
            speaker_id = infer_speaker_id(utt_id, speaker_id_regex=speaker_id_regex)
            utterances.append(
                Aishell3Utterance(
                    utt_id=utt_id,
                    speaker_id=speaker_id,
                    text=text,
                    audio_path=audio_path,
                )
            )
    return utterances


def choose_prompt_utterance(
    *,
    candidates: list[Aishell3Utterance],
    target_utt_id: str,
    policy: str,
    rng: random.Random,
) -> Optional[Aishell3Utterance]:
    filtered = [candidate for candidate in candidates if candidate.utt_id != target_utt_id]
    if not filtered:
        return None
    if policy == "random":
        return rng.choice(filtered)
    if policy == "longest":
        return max(filtered, key=lambda item: len(item.text))
    return sorted(filtered, key=lambda item: item.utt_id)[0]


class WhisperCerMetric:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device,
        dtype,
        language: str,
    ) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        self.device = device
        self.dtype = dtype if device.type == "cuda" else torch.float32
        self.language = language
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name_or_path,
            dtype=self.dtype,
        )
        self.model.to(device)
        self.model.eval()

    def transcribe(self, audio_path: str | Path) -> str:
        import torch

        waveform = load_audio_mono(audio_path, target_sample_rate=16000)
        audio = waveform.squeeze(0).cpu().numpy()
        inputs = self.processor(
            audio=audio,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
        )
        inputs = {
            name: value.to(device=self.device, dtype=self.dtype) if torch.is_floating_point(value) else value.to(self.device)
            for name, value in inputs.items()
        }
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.bool)
        generate_kwargs: dict[str, object] = {
            "task": "transcribe",
            "language": self.language,
        }
        with torch.inference_mode():
            predicted_ids = self.model.generate(**inputs, **generate_kwargs)
        return self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


class SpeakerSimilarityMetric:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        device,
    ) -> None:
        from transformers import AutoFeatureExtractor, AutoModel

        try:
            from transformers import AutoModelForAudioXVector
        except ImportError:
            AutoModelForAudioXVector = None

        self.device = device
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name_or_path)
        self.model = None
        if AutoModelForAudioXVector is not None:
            try:
                self.model = AutoModelForAudioXVector.from_pretrained(model_name_or_path)
            except Exception:  # pragma: no cover - fallback depends on local transformers version/model support
                pass
        if self.model is None:
            self.model = AutoModel.from_pretrained(model_name_or_path)
        self.model.to(device)
        self.model.eval()
        self.model_name_or_path = model_name_or_path

    def embed(self, audio_path: str | Path):
        import torch
        import torch.nn.functional as F

        waveform = load_audio_mono(audio_path, target_sample_rate=16000)
        inputs = self.feature_extractor(
            waveform.squeeze(0).cpu().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        inputs = {name: value.to(self.device) for name, value in inputs.items()}
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(dtype=torch.bool)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        embeddings = getattr(outputs, "embeddings", None)
        if embeddings is None:
            embeddings = getattr(outputs, "xvector", None)
        if embeddings is None:
            model_class = type(self.model).__name__
            raise RuntimeError(
                f"speaker model output does not expose embeddings or xvector: model={self.model_name_or_path!r}, "
                f"class={model_class}"
            )
        return F.normalize(embeddings, dim=-1).detach().cpu()

    def similarity(self, reference_audio_path: str | Path, generated_audio_path: str | Path) -> float:
        import torch.nn.functional as F

        reference_embedding = self.embed(reference_audio_path)
        generated_embedding = self.embed(generated_audio_path)
        return float(F.cosine_similarity(reference_embedding, generated_embedding).mean().item())
