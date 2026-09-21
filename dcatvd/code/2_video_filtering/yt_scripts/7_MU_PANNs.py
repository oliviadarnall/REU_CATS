"""
Filter extracted Meow Units using PANNs.

INPUT
meow_units/
audio/
spectrograms/
waveforms/
metadata.csv

OUTPUT
meow_units/

accepted/
audio/
spectrograms/
waveforms/

rejected/
audio/
spectrograms/
waveforms/

metadata_filtered.csv
metadata_filtered.jsonl
"""

import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import sys

# CONFIG

MU_ROOT = Path("meow_units")

AUDIO_DIR = MU_ROOT / "audio"
SPEC_DIR = MU_ROOT / "spectrograms"
WAVE_DIR = MU_ROOT / "waveforms"

METADATA_CSV = MU_ROOT / "metadata.csv"

PANNS_ROOT = Path("audioset_tagging_cnn")

CHECKPOINT = Path(
    "panns_resources/Cnn14_mAP=0.431.pth"
)

LABEL_FILE = Path(
    "panns_resources/class_labels_indices.csv"
)

ACCEPT_THRESHOLD = 0.05

#Remove near-silent audio

MIN_RMS = 0.005

TARGET_LABELS = {
    "Cat",
    "Meow",
    "Purr",
    "Hiss",
    "Animal",
    "Domestic animals, pets"
}

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

#OUTPUT FOLDERS

ACCEPTED_AUDIO = (
    MU_ROOT /
    "accepted" /
    "audio"
)

ACCEPTED_SPEC = (
    MU_ROOT /
    "accepted" /
    "spectrograms"
)

ACCEPTED_WAVE = (
    MU_ROOT /
    "accepted" /
    "waveforms"
)

REJECTED_AUDIO = (
    MU_ROOT /
    "rejected" /
    "audio"
)

REJECTED_SPEC = (
    MU_ROOT /
    "rejected" /
    "spectrograms"
)

REJECTED_WAVE = (
    MU_ROOT /
    "rejected" /
    "waveforms"
)

for folder in [
    ACCEPTED_AUDIO,
    ACCEPTED_SPEC,
    ACCEPTED_WAVE,
    REJECTED_AUDIO,
    REJECTED_SPEC,
    REJECTED_WAVE
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )

#LOAD PANNS

sys.path.append(str(PANNS_ROOT))
sys.path.append(str(PANNS_ROOT / "pytorch"))

from pytorch.models import Cnn14
from pytorch.pytorch_utils import move_data_to_device

model = Cnn14(
    sample_rate=32000,
    window_size=1024,
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14000,
    classes_num=527
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model"]
)

model.to(DEVICE)
model.eval()

#LOAD LABELS

LABELS = []

with open(LABEL_FILE) as f:
    lines = f.readlines()[1:]
    for line in lines:
        parts = line.strip().split(",")
        label = (
            parts[-1]
            .strip('"')
        )
        LABELS.append(label)

#HELPERS

def copy_file(src, dst):
    if src.exists():
        dst.parent.mkdir(
            parents=True,
            exist_ok=True
        )
        shutil.copy2(src, dst)

def classify_audio(audio_path):
    audio, sr = librosa.load(
        audio_path,
        sr=32000,
        mono=True
    )
    duration = len(audio) / sr
    MIN_PANNS_DURATION = 1.0

    if duration < MIN_PANNS_DURATION:
        target_samples = int(MIN_PANNS_DURATION * sr)
        audio = np.pad(audio, (0,target_samples - len(audio)))

    rms = float(
        np.sqrt(
            np.mean(audio ** 2)
        )
    )

    if rms < MIN_RMS:
        return {
            "accepted": False,
            "reason": "low_rms",
            "rms": rms,
            "score": 0.0,
            "target_scores": {}
        }

    audio = audio[None, :]
    audio = move_data_to_device(audio, DEVICE)

    with torch.no_grad():
        output = model(audio)

    scores = (
        output["clipwise_output"]
        .data.cpu()
        .numpy()[0]
    )

    max_target_score = 0.0
    target_scores = {}
    for idx, score in enumerate(scores):
        label = LABELS[idx]
        if label in TARGET_LABELS:
            target_scores[label] = (
                float(score)
            )
            max_target_score = max(
                max_target_score,
                float(score)
            )
    accepted = (
        max_target_score
        >= ACCEPT_THRESHOLD
    )

    return {
        "accepted": accepted,
        "reason":
            "accepted"
            if accepted
            else "low_score",

        "rms": rms,

        "score":
            max_target_score,

        "target_scores":
            target_scores
    }

#MAIN

metadata = pd.read_csv(METADATA_CSV)

filtered_rows = []

print(
    f"Processing "
    f"{len(metadata)} MUs"
)

for _, row in tqdm(
    metadata.iterrows(),
    total=len(metadata)
):

    mu_id = row["mu_id"]

    audio_path = (MU_ROOT / row["audio_path"])
    spec_path = (  MU_ROOT /row["spectrogram_path"])
    wave_path = (MU_ROOT / row["waveform_path"])

    channel_id = row["channel_id"]

    if not audio_path.exists():
        continue

    try:
        result = classify_audio(audio_path)
    except Exception as e:
        print(f"Error {mu_id}: {e}")
        continue

    accepted = result["accepted"]

    if accepted:
        audio_dst = (
            ACCEPTED_AUDIO /
            channel_id /
            audio_path.name
        )

        spec_dst = (
            ACCEPTED_SPEC /
            channel_id /
            spec_path.name
        )

        wave_dst = (
            ACCEPTED_WAVE /
            channel_id /
            wave_path.name
        )

    else:
        audio_dst = (
            REJECTED_AUDIO /
            channel_id /
            audio_path.name
        )

        spec_dst = (
            REJECTED_SPEC /
            channel_id /
            spec_path.name
        )

        wave_dst = (
            REJECTED_WAVE /
            channel_id /
            wave_path.name
        )

    copy_file(
        audio_path,
        audio_dst
    )

    copy_file(
        spec_path,
        spec_dst
    )

    copy_file(
        wave_path,
        wave_dst
    )

    row = row.to_dict()

    row["panns_score"] = (
        result["score"]
    )

    row["rms"] = (
        result["rms"]
    )

    row["accepted"] = (
        accepted
    )

    row["channel_id"] = (
        channel_id
    )

    row["output_partition"] = (
        "accepted"
        if accepted
        else "rejected"
    )

    row["filter_reason"] = (
        result["reason"]
    )

    row["target_scores"] = json.dumps(
        result["target_scores"]
    )

    filtered_rows.append(row)

#SAVE METADATA

filtered_df = pd.DataFrame(
    filtered_rows
)

filtered_df.to_csv(
    MU_ROOT /
    "metadata_filtered.csv",
    index=False
)

with open(
    MU_ROOT /
    "metadata_filtered.jsonl",
    "w"
) as f:

    for row in filtered_rows:

        f.write(
            json.dumps(row)
            + "\n"
        )

accepted_count = sum(
    row["accepted"]
    for row in filtered_rows
)

rejected_count = (
    len(filtered_rows)
    - accepted_count
)

print()
print(
    f"Accepted: {accepted_count}"
)

print(f"Rejected: {rejected_count}")