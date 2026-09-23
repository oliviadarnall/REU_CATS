#!/usr/bin/env python3

"""
On-the-fly BEATs inference for cat call-type events.

This script does not use HuBERT shards or precomputed features. It reads each event
waveform, runs the trained BEATs classifier, and writes one prediction row per audio file.
It can also update parquet metadata that contains an event_audio_paths column.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from train_beats_call_classifier import (
    CatCallBEATsClassifier,
    SUPPORTED_AUDIO_SUFFIXES,
    load_cat_call_beats_checkpoint,
)


def atomic_write_parquet(df: pd.DataFrame, path: str | os.PathLike[str], required: bool = True) -> bool:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
    except ImportError as exc:
        if required:
            raise
        print(f"Warning: could not write parquet because pyarrow/fastparquet is unavailable: {exc}")
        return False
    os.replace(tmp, path)
    return True


def atomic_write_tsv(df: pd.DataFrame, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, sep=",", index=False)
    os.replace(tmp, path)


def atomic_write_jsonl(rows: list[dict[str, Any]], path: str | os.PathLike[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, path)


def iter_audio_files(paths: Sequence[str]) -> list[str]:
    out: list[str] = []
    for item in paths:
        p = Path(item)
        if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES:
            out.append(str(p))
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES:
                    out.append(str(fp))
        else:
            raise FileNotFoundError(f"audio path not found or unsupported suffix: {item}")
    unique = sorted(dict.fromkeys(out))
    assert unique, "no audio files found"
    return unique


def collect_paths_from_parquets(parquet_paths: Sequence[str]) -> list[str]:
    all_paths: set[str] = set()
    for p in parquet_paths:
        assert os.path.isfile(p), f"input parquet not found: {p}"
        df = pd.read_parquet(p, columns=["event_audio_paths"])
        assert "event_audio_paths" in df.columns, f"{p} missing event_audio_paths column"
        for paths in df["event_audio_paths"]:
            if paths is None:
                continue
            for path in list(paths):
                if path:
                    all_paths.add(str(path))
    paths_sorted = sorted(all_paths)
    assert paths_sorted, "no event_audio_paths found in input parquets"
    return paths_sorted


class AudioPathDataset(Dataset):
    def __init__(
        self,
        paths: Sequence[str],
        sample_rate: int = 16000,
        min_audio_seconds: float = 1.0,
        max_audio_seconds: float | None = 10.0,
    ) -> None:
        self.paths = list(paths)
        self.sample_rate = int(sample_rate)
        self.min_audio_samples = int(round(min_audio_seconds * self.sample_rate))
        self.max_audio_samples = None if max_audio_seconds is None else int(round(max_audio_seconds * self.sample_rate))
        assert self.paths, "empty path list"
        assert self.sample_rate == 16000, "BEATs expects 16 kHz audio"
        assert self.min_audio_samples > 0, "min_audio_seconds must be positive"
        if self.max_audio_samples is not None:
            assert self.max_audio_samples >= self.min_audio_samples, "max_audio_seconds must be >= min_audio_seconds"

    def __len__(self) -> int:
        return len(self.paths)

    def _read_audio(self, path: str) -> torch.Tensor:
        waveform_np, sr = sf.read(path, always_2d=True, dtype="float32")
        assert waveform_np.ndim == 2 and waveform_np.shape[0] > 0, f"bad audio shape: {path} {waveform_np.shape}"
        waveform = torch.from_numpy(waveform_np).mean(dim=1).contiguous()
        if int(sr) != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform.unsqueeze(0), int(sr), self.sample_rate).squeeze(0).contiguous()
        waveform = torch.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
        if waveform.numel() < self.min_audio_samples:
            waveform = F.pad(waveform, (0, self.min_audio_samples - waveform.numel()))
        if self.max_audio_samples is not None and waveform.numel() > self.max_audio_samples:
            extra = waveform.numel() - self.max_audio_samples
            start = extra // 2
            waveform = waveform[start:start + self.max_audio_samples].contiguous()
        assert waveform.ndim == 1 and waveform.numel() >= self.min_audio_samples, f"bad waveform after processing: {path}"
        return waveform

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        path = self.paths[idx]
        try:
            return self._read_audio(path), path
        except Exception as exc:
            raise RuntimeError(f"failed to load {path}: {exc}") from exc


def collate_audio_paths(batch: list[tuple[torch.Tensor, str]]) -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    assert batch, "empty batch"
    waveforms, paths = zip(*batch)
    lengths = torch.tensor([int(w.numel()) for w in waveforms], dtype=torch.long)
    assert int(lengths.min().item()) > 0, "zero-length waveform in batch"
    max_len = int(lengths.max().item())
    padded = torch.stack([F.pad(w, (0, max_len - int(w.numel()))) for w in waveforms], dim=0).contiguous()
    assert padded.ndim == 2 and padded.size(0) == len(paths), tuple(padded.shape)
    return padded, lengths, tuple(paths)


def apply_thresholds(
    probs: torch.Tensor,
    class_names: list[str],
    tm_thresh: float = 0.60,
    meow_thresh: float = 0.40,
    constituent_thresh: float = 0.20,
):
    """
    probs: (batch_size, num_classes)
    returns predicted class indices
    """
    idx = {c: i for i, c in enumerate(class_names)}
    meow_i = idx["meow"]
    trill_i = idx["trill"]
    tm_i = idx["trill_meow"]
    pred = []

    for p in probs:
        #Default prediction
        winner = int(torch.argmax(p))

        tm = p[tm_i].item()
        meow = p[meow_i].item()
        trill = p[trill_i].item()

        #Trill-meow rule
        if winner == tm_i:
            if (
                tm >= tm_thresh
                and meow >= constituent_thresh
                and trill >= constituent_thresh
            ):
                pred.append(tm_i)
                continue

            #Otherwise choose whichever constituent is stronger
            if meow >= trill:
                pred.append(meow_i)
            else:
                pred.append(trill_i)
            continue

        #Encourage meow predictions
        if (meow >= meow_thresh and meow >= p[winner] - 0.10):
            pred.append(meow_i)
            continue

        pred.append(winner)

    return torch.tensor(pred, device=probs.device)



def run_predictions(args: argparse.Namespace, event_paths: Sequence[str]) -> pd.DataFrame:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.set_float32_matmul_precision("high")
    print(f"Loading classifier checkpoint: {args.checkpoint}")
    model = load_cat_call_beats_checkpoint(
        args.checkpoint,
        map_location="cpu",
        beats_checkpoint=args.beats_checkpoint,
        beats_code_dir=args.beats_code_dir,
    )
    model.eval().to(device)
    class_names = list(model.class_names)
    assert class_names and len(set(class_names)) == len(class_names), f"bad class names: {class_names}"
    print(f"Classes: {class_names}")
    print(f"Device: {device}")

    dataset = AudioPathDataset(
        event_paths,
        min_audio_seconds=args.min_audio_seconds,
        max_audio_seconds=args.max_audio_seconds,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_audio_paths,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )
    rows: list[dict[str, Any]] = []
    amp_enabled = device.type == "cuda" and args.amp_dtype in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    print(f"Running on-the-fly inference for {len(event_paths)} audio files...")
    with torch.inference_mode():
        for waveforms, lengths, paths in tqdm(loader, desc="Predicting"):
            waveforms = waveforms.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                logits = model(waveforms, lengths)

            probs = torch.softmax(logits.float(), dim=1)
            conf, pred = probs.max(dim=1)

            probs_cpu = probs.cpu().numpy()
            conf_cpu = conf.cpu().numpy()
            pred_cpu = pred.cpu().numpy()
            for path, pred_idx, confidence, prob_vec in zip(paths, pred_cpu, conf_cpu, probs_cpu):
                pred_idx = int(pred_idx)
                prob_map = {name: round(float(prob), 6) for name, prob in zip(class_names, prob_vec)}
                rows.append({
                    "event_audio_path": path,
                    "predicted_class": class_names[pred_idx],
                    "confidence": round(float(confidence), 6),
                    "call_type_probabilities": prob_map,
                })
            del waveforms, lengths, logits, probs, conf, pred
    df = pd.DataFrame(rows)
    assert len(df) == len(event_paths), f"prediction count mismatch: {len(df)} vs {len(event_paths)}"
    return df


def write_outputs(predictions_df: pd.DataFrame, output_prefix: str) -> None:
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = prefix.with_suffix(".parquet")
    jsonl_path = prefix.with_suffix(".jsonl")
    tsv_path = prefix.with_suffix(".csv")
    wrote_parquet = atomic_write_parquet(predictions_df, parquet_path, required=False)
    atomic_write_jsonl(predictions_df.to_dict(orient="records"), jsonl_path)
    flat_df = predictions_df.copy()
    flat_df["call_type_probabilities"] = flat_df["call_type_probabilities"].map(json.dumps)
    atomic_write_tsv(flat_df, tsv_path)
    parquet_line = str(parquet_path) if wrote_parquet else f"{parquet_path} (skipped: pyarrow/fastparquet missing)"
    print(f"Wrote predictions:\n  {parquet_line}\n  {jsonl_path}\n  {tsv_path}")


def update_parquet(input_path: str, output_path: str, predictions_df: pd.DataFrame) -> None:
    print(f"Updating parquet metadata: {input_path} -> {output_path}")
    df = pd.read_parquet(input_path)
    assert "event_audio_paths" in df.columns, f"{input_path} missing event_audio_paths"
    pred_by_path = predictions_df.set_index("event_audio_path").to_dict(orient="index")
    event_predicted_classes: list[np.ndarray] = []
    event_confidences: list[np.ndarray] = []
    event_call_type_probabilities: list[np.ndarray] = []
    updated_events: list[np.ndarray] = []

    has_events = "events" in df.columns
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Updating rows"):
        row_paths = list(row["event_audio_paths"])
        row_events_raw = list(row["events"]) if has_events and row["events"] is not None else [{} for _ in row_paths]
        assert len(row_events_raw) == len(row_paths), "events and event_audio_paths length mismatch"
        row_classes: list[str] = []
        row_confidences: list[float] = []
        row_probs: list[dict[str, float]] = []
        row_events: list[dict[str, Any]] = []
        for event, path in zip(row_events_raw, row_paths):
            event_dict = dict(event) if isinstance(event, Mapping) else {}
            pred = pred_by_path.get(str(path))
            if pred is None:
                pred_class = str(event_dict.get("predicted_class", "unknown"))
                conf = float(event_dict.get("confidence", 0.0))
                probs = event_dict.get("call_type_probabilities", {})
                if not isinstance(probs, dict):
                    probs = {}
            else:
                pred_class = str(pred["predicted_class"])
                conf = float(pred["confidence"])
                probs = dict(pred["call_type_probabilities"])
            event_dict["predicted_class"] = pred_class
            event_dict["confidence"] = conf
            event_dict["call_type_probabilities"] = probs
            row_classes.append(pred_class)
            row_confidences.append(conf)
            row_probs.append(probs)
            row_events.append(event_dict)
        event_predicted_classes.append(np.array(row_classes, dtype=object))
        event_confidences.append(np.array(row_confidences, dtype=np.float32))
        event_call_type_probabilities.append(np.array(row_probs, dtype=object))
        updated_events.append(np.array(row_events, dtype=object))

    df["event_predicted_classes"] = event_predicted_classes
    df["event_confidences"] = event_confidences
    df["event_call_type_probabilities"] = event_call_type_probabilities
    if has_events:
        df["events"] = updated_events
    atomic_write_parquet(df, output_path, required=True)
    print(f"Wrote updated parquet: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-paths", nargs="+", help="Audio files or directories. Directories are searched recursively.")
    source.add_argument("--input-parquets", nargs="+", help="Parquets with event_audio_paths to classify on the fly.")
    parser.add_argument("--checkpoint", required=True, help="Lightning checkpoint produced by train_beats_call_classifier.py")
    parser.add_argument("--beats-checkpoint", default="BEATs_iter3_plus_AS2M.pt")
    parser.add_argument("--beats-code-dir", default="./beats")
    parser.add_argument("--output-prefix", required=True, help="Output prefix; writes .parquet, .jsonl, .tsv")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default="bf16")
    parser.add_argument("--min-audio-seconds", type=float, default=1.0)
    parser.add_argument("--max-audio-seconds", type=float, default=10.0)
    parser.add_argument("--update-parquets", action="store_true", help="Only valid with --input-parquets. Writes updated parquet copies.")
    parser.add_argument("--updated-parquet-dir", default=None, help="Directory for updated parquet copies. Defaults to output_prefix parent / updated_parquets.")
    args = parser.parse_args()

    assert args.batch_size > 0, args.batch_size
    assert args.num_workers >= 0, args.num_workers
    assert args.min_audio_seconds > 0, args.min_audio_seconds
    if args.max_audio_seconds is not None:
        assert args.max_audio_seconds >= args.min_audio_seconds, "max_audio_seconds must be >= min_audio_seconds"

    if args.audio_paths:
        event_paths = iter_audio_files(args.audio_paths)
    else:
        event_paths = collect_paths_from_parquets(args.input_parquets)
    missing = [p for p in event_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} event audio files are missing; first missing: {missing[:5]}")

    predictions_df = run_predictions(args, event_paths)
    write_outputs(predictions_df, args.output_prefix)

    if args.update_parquets:
        assert args.input_parquets, "--update-parquets requires --input-parquets"
        update_dir = Path(args.updated_parquet_dir) if args.updated_parquet_dir else Path(args.output_prefix).parent / "updated_parquets"
        update_dir.mkdir(parents=True, exist_ok=True)
        for parquet_path in args.input_parquets:
            out_path = update_dir / Path(parquet_path).name
            update_parquet(parquet_path, str(out_path), predictions_df)


if __name__ == "__main__":
    main()
