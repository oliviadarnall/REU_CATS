'''
Script for extracting metadata from YouTube videos and shorts and saving:
- thumbnails
- video_metadata.json
- video_metadata.csv
Skip:
- thumbnails that already exist
- videos that already exist in video_metadata.json
'''

import yt_dlp
import requests
from pathlib import Path
import json
import pandas as pd
import time

#Configure the channel ID and paths

CHANNEL_ID = "channel_id"

BASE_URL = f"https://www.youtube.com/channel/{CHANNEL_ID}"

URLS = {
    "videos": BASE_URL + "/videos",
    "shorts": BASE_URL + "/shorts"
}

#Create directories

thumbnail_dir = Path(
    f"thumbnails/{CHANNEL_ID}"
)

thumbnail_dir.mkdir(
    parents=True,
    exist_ok=True
)

metadata_dir = Path(
    f"metadata/{CHANNEL_ID}"
)

metadata_dir.mkdir(
    parents=True,
    exist_ok=True
)

json_path = (
    metadata_dir / "video_metadata.json"
)

csv_path = (
    metadata_dir / "video_metadata.csv"
)

#Load existing metadata if it exists

existing_metadata = []
existing_video_ids = set()

if json_path.exists():
    try:
        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:
            existing_metadata = json.load(f)

        existing_video_ids = {
            v["video_id"]
            for v in existing_metadata
            if "video_id" in v
        }

        print(
            f"Loaded existing metadata for "
            f"{len(existing_video_ids)} videos"
        )

    except Exception as e:
        print("Failed to load existing metadata")
        print(e)

#Extract flat metadata

ydl_flat_opts = {
    "extract_flat": True,
    "quiet": True,
    "playlistend": 1000
}

all_videos = []

#Get video ids from YouTube videos and shorts
with yt_dlp.YoutubeDL(ydl_flat_opts) as ydl:

    for section_name, url in URLS.items():

        print(f"\nExtracting {section_name}")

        try:
            info = ydl.extract_info(
                url,
                download=False
            )

            if "entries" not in info:
                continue

            for v in info["entries"]:

                if v is None:
                    continue

                video_id = v.get("id")

                if not video_id:
                    continue

                video_data = {
                    "video_id": video_id,
                    "title": v.get("title"),
                    "url": (
                        f"https://www.youtube.com/watch?"
                        f"v={video_id}"
                    ),
                    "section": section_name,
                    "thumbnail": None
                }

                if (
                    "thumbnails" in v
                    and v["thumbnails"]
                ):
                    video_data["thumbnail"] = (
                        v["thumbnails"][-1]["url"]
                    )

                all_videos.append(video_data)

        except Exception as e:
            print(f"Failed to extract {section_name}")
            print(e)

#Remove duplicates

unique_videos = {}

for v in all_videos:
    unique_videos[v["video_id"]] = v

videos = list(unique_videos.values())

print(f"\nTotal unique videos found: {len(videos)}")

#Download thumbnails
#Skip thumbnails that already exist

for v in videos:

    if not v["thumbnail"]:
        continue

    thumbnail_path = (
        thumbnail_dir / f"{v['video_id']}.jpg"
    )

    #Skip if thumbnail already exists
    if thumbnail_path.exists():
        print(
            f"Skipping existing thumbnail: "
            f"{v['video_id']}"
        )
        continue

    try:
        img = requests.get(
            v["thumbnail"],
            timeout=10
        ).content

        with open(
            thumbnail_path,
            "wb"
        ) as f:
            f.write(img)

        print(
            f"Downloaded thumbnail: "
            f"{v['video_id']}"
        )

    except Exception as e:
        print(
            f"Failed thumbnail: "
            f"{v['video_id']}"
        )
        print(e)

#Filter videos that already have metadata

videos_to_process = []

for v in videos:

    if v["video_id"] in existing_video_ids:
        print(
            f"Skipping existing metadata: "
            f"{v['video_id']}"
        )
        continue

    videos_to_process.append(v)

print(
    f"\nNew videos needing metadata: "
    f"{len(videos_to_process)}"
)

#Get the detailed metadata

ydl_detail_opts = {
    "quiet": True,
    "javascript_runtimes": ["deno"]
}

new_detailed_videos = []

with yt_dlp.YoutubeDL(ydl_detail_opts) as ydl:

    for idx, video in enumerate(videos_to_process):

        try:
            print(
                f"\n[{idx+1}/{len(videos_to_process)}] "
                f"Detailed metadata: "
                f"{video['video_id']}"
            )

            info = ydl.extract_info(
                video["url"],
                download=False
            )

            detailed_video = {
                "video_id": info.get("id", ""),
                "title": info.get("title", ""),
                "url": video["url"],
                "channel_id": info.get(
                    "channel_id",
                    ""
                ),
                "channel_name": info.get(
                    "channel",
                    ""
                ),
                "video_description": info.get(
                    "description",
                    ""
                ),
                "upload_date": info.get(
                    "upload_date",
                    ""
                ),
                "duration": info.get(
                    "duration",
                    0
                ),
                "section": video["section"],
                "thumbnail": video["thumbnail"]
            }

            new_detailed_videos.append(
                detailed_video
            )

            # polite rate limiting
            # time.sleep(0.25)

        except Exception as e:
            print(
                f"Failed detailed extraction: "
                f"{video['video_id']}"
            )
            print(e)

#Combine old + new metadata

combined_metadata = (
    existing_metadata + new_detailed_videos
)

#Remove duplicates again just in case
combined_unique = {}

for v in combined_metadata:
    combined_unique[v["video_id"]] = v

combined_metadata = list(
    combined_unique.values()
)

print(
    f"\nTotal metadata entries saved: "
    f"{len(combined_metadata)}"
)

#Save JSON

with open(
    json_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        combined_metadata,
        f,
        indent=4,
        ensure_ascii=False
    )

print(f"\nSaved JSON: {json_path}")

#Save CSV

df = pd.DataFrame(combined_metadata)

df.to_csv(
    csv_path,
    index=False
)

print(f"Saved CSV: {csv_path}")

print("\nFinished.")