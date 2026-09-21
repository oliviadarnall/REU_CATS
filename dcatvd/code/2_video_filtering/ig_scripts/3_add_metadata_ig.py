import json
import subprocess
from pathlib import Path
from datetime import datetime

# Paths

WAV_ROOT = Path(
    "downloads/dataset_filtered/accepted/wav"
)

CHANNELS_JSON = Path(
    "channels/channels.json"
)

OUTPUT_ROOT = Path(
    "downloads/dataset_filtered/accepted/metadata"
)

# Age / life stage

def compute_age_years(birthday_str, upload_date_str):
    birthday_dt = datetime.strptime(
        birthday_str,
        "%Y-%m-%d"
    )

    upload_dt = datetime.strptime(
        upload_date_str,
        "%Y%m%d"
    )

    delta_days = (upload_dt - birthday_dt).days
    return round(delta_days / 365.25, 2)


def determine_stage(age_years):
    if age_years <= 0.58:
        return "kitten"
    elif age_years <= 3:
        return "junior"
    elif age_years <= 7:
        return "prime"
    elif age_years <= 11:
        return "mature"
    elif age_years <= 15:
        return "senior"
    else:
        return "geriatric"

# yt-dlp helper

def get_upload_date(video_id):
    url = f"https://www.instagram.com/reel/{video_id}"

    cmd = [
        "yt-dlp",
        "--cookies", "scripts/instagram_cookies.txt",
        "--dump-single-json",
        "--skip-download",
        url,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Failed: {video_id}")
        print(result.stderr)
        return None

    try:
        info = json.loads(result.stdout)
        return info.get("upload_date")
    except Exception:
        print(f"Could not parse metadata for {video_id}")
        return None

# Load channel metadata

with open(CHANNELS_JSON, "r", encoding="utf-8") as f:
    channels = json.load(f)

channel_lookup = {
    c["channel_id"]: c
    for c in channels
}

# Process every channel

for channel_dir in WAV_ROOT.iterdir():

    if not channel_dir.is_dir():
        continue

    channel_id = channel_dir.name

    if channel_id not in channel_lookup:
        print(f"Skipping {channel_id} (not found in JSON)")
        continue

    channel_info = channel_lookup[channel_id]

    breed = channel_info["breed"]
    birthday = channel_info["birthday"]

    print(f"\nProcessing {channel_id}")

    metadata = []

    wav_files = sorted(channel_dir.glob("*.wav"))

    for wav in wav_files:

        video_id = wav.stem

        upload_date = get_upload_date(video_id)

        if upload_date is None:
            continue

        age = compute_age_years(
            birthday,
            upload_date
        )

        stage = determine_stage(age)

        metadata.append(
            {
                "video_id": video_id,
                "channel_id": channel_id,
                "upload_date": upload_date,
                "birthday": birthday,
                "age": age,
                "stage": stage,
                "breed": breed,
            }
        )

        print(video_id, upload_date)

    output_dir = OUTPUT_ROOT / channel_id
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(
        output_dir / "metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    print(f"Saved {len(metadata)} entries.")