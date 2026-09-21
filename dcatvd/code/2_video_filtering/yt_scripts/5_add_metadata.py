'''
Export metadata to {channel_name}.json
'''

import json
import subprocess
from pathlib import Path
from datetime import datetime
import re

#Configure the Channel and Paths

CHANNEL_ID = "channel_id"

CHANNELS_JSON = Path("channels/channels.json")

VIDEO_METADATA_JSON = Path(
    f"metadata/{CHANNEL_ID}/video_metadata.json"
)

DOWNLOAD_DIR = Path(
    f"downloads/dataset_filtered/accepted/mp4/{CHANNEL_ID}"
)

JSON_OUTPUT = Path(
    f"downloads/dataset_filtered/accepted/metadata/{CHANNEL_ID}"
)

JSON_OUTPUT.mkdir(parents=True, exist_ok=True)

json_path = (
    JSON_OUTPUT / "metadata.json"
)


#Load the channel information from channels.json
with open(CHANNELS_JSON, "r", encoding="utf-8") as f:
    channels = json.load(f)

channel_info = next(
    c for c in channels
    if c["channel_id"] == CHANNEL_ID
)

breed = channel_info["breed"]
birthday = channel_info["birthday"]


#Load the video metadata from video_metadata.json
with open(VIDEO_METADATA_JSON, "r", encoding="utf-8") as f:
    videos = json.load(f)


#Compute the age and the lifestage of the cat
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
    

#Store exported metadata records
exported_metadata = []

#Process the videos
for video in videos:
    video_id = video["video_id"]

    input_file = DOWNLOAD_DIR / f"{video_id}.mp4"
    if not input_file.exists():
        continue

    print(f"\nProcessing {video_id}")

    #Extract the video metadata
    video_title = video.get("title", "")

    channel_name = video.get(
        "channel_name",
        "unknown_channel"
    )

    video_description = video.get(
        "video_description",
        ""
    )

    upload_date = video.get(
        "upload_date",
        ""
    )

    if not upload_date:
        print("Missing upload date")
        continue

    #Compute age and life stage
    age = compute_age_years(
        birthday,
        upload_date
    )

    stage = determine_stage(age)

    #Metadata dictionary
    metadata_record = {
        "video_id": video_id,
        "video_title": video_title,
        "channel_id": CHANNEL_ID,
        "channel_name": channel_name,
        "video_description": video_description,
        "upload_date": upload_date,
        "birthday": birthday,
        "age": age,
        "stage": stage,
        "breed": breed,
        "context": ""
    }

    #Add to export list
    exported_metadata.append(metadata_record)

#Export metadata JSON
with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        exported_metadata,
        f,
        indent=4,
        ensure_ascii=False
    )

print(
    f"\nSaved metadata JSON: "
    f"{json_path}"
)

print("\nFinished.")