"""
Filter cat vocalization dataset using PANNs
for all channels in the dataset_wav folder

Rejects files whose inference takes too long
"""

import json
import shutil
import time
from pathlib import Path
import librosa
import numpy as np
import torch

import sys

#Import PANNs

PANNS_ROOT = Path("audioset_tagging_cnn")

sys.path.append(str(PANNS_ROOT))
sys.path.append(str(PANNS_ROOT / "pytorch"))

from pytorch.models import Cnn14
from pytorch.pytorch_utils import move_data_to_device

#Configure the Paths

WAV_ROOT = Path("downloads/dataset_wav")

MP4_ROOT = Path("downloads/dataset_mp4")

FILTERED_ROOT = Path(
    "downloads/dataset_filtered"
)

METADATA_PATH = Path(
    "metadata/panns_results.json"
)

ACCEPT_THRESHOLD = 0.4

MAX_INFERENCE_SECONDS = 120.0

TARGET_LABELS = {
    "Cat",
    "Meow",
    "Purr",
    "Hiss"
}

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

#Load the model

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
    "panns_resources/Cnn14_mAP=0.431.pth",
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model"]
)

model.to(DEVICE)
model.eval()

#Load the labels

with open(
    "panns_resources/class_labels_indices.csv"
) as f:
    lines = f.readlines()[1:]
    LABELS = []
    for line in lines:
        parts = line.strip().split(",")
        label = parts[-1].strip('"')
        LABELS.append(label)

#Classify the audio by running inference with PANNs
def classify_audio(wav_path):
    start_time = time.time()
    audio, sr = librosa.load(
        wav_path,
        sr=32000,
        mono=True
    )

    audio = audio[None, :]
    audio = move_data_to_device(
        audio,
        DEVICE
    )

    with torch.no_grad():
        output = model(audio)

    inference_time = (
        time.time() - start_time
    )

    scores = (
        output["clipwise_output"]
        .data.cpu()
        .numpy()[0]
    )

    predictions = []

    for idx, score in enumerate(scores):
        predictions.append(
            (LABELS[idx], float(score))
        )

    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    max_target_score = 0.0
    target_scores = {}

    for label, score in predictions:
        if label in TARGET_LABELS:
            target_scores[label] = score
            max_target_score = max(
                max_target_score,
                score
            )

    return {
        "max_target_score":
            max_target_score,

        "target_scores":
            target_scores,

        "top_predictions":
            predictions[:10],

        "inference_time":
            inference_time
    }

#Determine whether to accept or reject
def determine_category(score, inference_time):
    #Reject files that are too slow
    if (inference_time> MAX_INFERENCE_SECONDS):
        return "rejected_timeout"
    if score >= ACCEPT_THRESHOLD:
        return "accepted"
    return "rejected"

#Move files

def move_file(src, dst):
    dst.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    shutil.copy2(src, dst)

#Main loop

results = []

wav_files = list(WAV_ROOT.rglob("*.wav"))
print(f"Found {len(wav_files)} wav files")

for wav_path in wav_files:
    try:
        channel_id = (wav_path.parent.name)
        video_id = wav_path.stem

        mp4_path = (
            MP4_ROOT
            / channel_id
            / f"{video_id}.mp4"
        )

        print(f"\nProcessing {video_id}")

        #Classify the audio using PANNs
        output = classify_audio(wav_path)

        #Decide whether to accept or reject
        category = determine_category(
            output["max_target_score"],
            output["inference_time"]
        )

        #Destination paths
        dst_wav = (
            FILTERED_ROOT
            / category
            / "wav"
            / channel_id
            / wav_path.name
        )

        dst_mp4 = (
            FILTERED_ROOT
            / category
            / "mp4"
            / channel_id
            / mp4_path.name
        )

        #Copy the files into the respective destination
        #Copy only accepted files
        if category == "accepted":
            move_file(
                wav_path,
                dst_wav
            )

            if mp4_path.exists():
                move_file(
                    mp4_path,
                    dst_mp4
                )

        #Log the metadata
        result = {
            "video_id": video_id,
            "channel_id": channel_id,
            "classification": category,
            "max_target_score": output["max_target_score"],
            "target_scores": output["target_scores"],
            "top_predictions": output["top_predictions"]
        }

        results.append(result)

        print(
            f"{category.upper()} | "
            f"score="
            f"{output['max_target_score']:.3f} | "
            f"time="
            f"{output['inference_time']:.2f}s"
        )

    except Exception as e:
        print(f"ERROR: {wav_path}")
        print(e)

#Save the metadata

METADATA_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    METADATA_PATH,
    "w"
) as f:
    json.dump(
        results,
        f,
        indent=2
    )

print("\nDone!")