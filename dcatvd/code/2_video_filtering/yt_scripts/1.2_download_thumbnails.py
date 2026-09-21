"""
Script for extracting metadata from YouTube videos and shorts and saving:
- thumbnails
- video_metadata.json
- video_metadata.csv

- Skips thumbnails that already exist
- Skips videos already present in video_metadata.json
- Saves progress every 50 newly processed videos
- Safe to resume after interruption
"""

import yt_dlp
import requests
from pathlib import Path
import json
import pandas as pd
import time

#Configure the Channel and paths

CHANNEL_ID = "channel_id"

BASE_URL = f"https://www.youtube.com/channel/{CHANNEL_ID}"

URLS = {
    "videos": BASE_URL + "/videos",
    "shorts": BASE_URL + "/shorts",
}

SAVE_EVERY = 50

thumbnail_dir = Path(
    f"thumbnails/{CHANNEL_ID}"
)

thumbnail_dir.mkdir(
    parents=True,
    exist_ok=True,
)

metadata_dir = Path(
    f"metadata/{CHANNEL_ID}"
)

metadata_dir.mkdir(
    parents=True,
    exist_ok=True,
)

json_path = (
    metadata_dir / "video_metadata.json"
)

csv_path = (
    metadata_dir / "video_metadata.csv"
)

#Load the existing metadata

existing_metadata = []
existing_video_ids = set()

if json_path.exists():
    try:
        with open(
            json_path,
            "r",
            encoding="utf-8",
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

def save_metadata(metadata):
    unique = {}
    for v in metadata:
        video_id = v.get("video_id")
        if video_id:
            unique[video_id] = v
    metadata = list(unique.values())

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )

    pd.DataFrame(
        metadata
    ).to_csv(
        csv_path,
        index=False,
    )

    print(
        f"\nCheckpoint saved "
        f"({len(metadata)} total videos)"
    )

#Extract the video IDs

ydl_flat_opts = {
    "extract_flat": True,
    "quiet": True,
    "playlistend": 1000,
}

all_videos = []

with yt_dlp.YoutubeDL(
    ydl_flat_opts
) as ydl:
    for section_name, url in URLS.items():
        print(
            f"\nExtracting {section_name}"
        )
        try:
            info = ydl.extract_info(
                url,
                download=False,
            )
            if (
                not info
                or "entries" not in info
            ):
                continue

            for v in info["entries"]:
                if v is None:
                    continue

                video_id = v.get("id")

                if not video_id:
                    continue

                video_data = {
                    "video_id": video_id,
                    "title": v.get(
                        "title"
                    ),
                    "url": (
                        "https://www.youtube.com/watch"
                        f"?v={video_id}"
                    ),
                    "section": section_name,
                    "thumbnail": None,
                }

                if (
                    "thumbnails" in v
                    and v["thumbnails"]
                ):
                    video_data[
                        "thumbnail"
                    ] = (
                        v["thumbnails"][-1][
                            "url"
                        ]
                    )

                all_videos.append(
                    video_data
                )

        except Exception as e:

            print(
                f"Failed to extract "
                f"{section_name}"
            )

            print(e)

unique_videos = {}

for v in all_videos:
    unique_videos[
        v["video_id"]
    ] = v
videos = list(
    unique_videos.values()
)

print(
    f"\nTotal unique videos found: "
    f"{len(videos)}"
)

#Download the thumbnails

for v in videos:
    if not v["thumbnail"]:
        continue
    thumbnail_path = (
        thumbnail_dir
        / f"{v['video_id']}.jpg"
    )
    if thumbnail_path.exists():
        print(
            "Skipping existing thumbnail: "
            f"{v['video_id']}"
        )
        continue

    try:
        response = requests.get(
            v["thumbnail"],
            timeout=10,
        )
        response.raise_for_status()

        with open(
            thumbnail_path,
            "wb",
        ) as f:
            f.write(
                response.content
            )
        print(
            "Downloaded thumbnail: "
            f"{v['video_id']}"
        )

    except Exception as e:
        print(
            "Failed thumbnail: "
            f"{v['video_id']}"
        )
        print(e)

#Filter unprocessed videos

videos_to_process = []

for v in videos:
    if (
        v["video_id"]
        in existing_video_ids
    ):
        print(
            "Skipping existing metadata: "
            f"{v['video_id']}"
        )
        continue
    videos_to_process.append(
        v
    )
print(
    "\nNew videos needing metadata: "
    f"{len(videos_to_process)}"
)

#Extract the detailed metadata

ydl_detail_opts = {
    "quiet": True,
    "javascript_runtimes": [
        "deno"
    ],
}

new_detailed_videos = []

with yt_dlp.YoutubeDL(
    ydl_detail_opts
) as ydl:
    for idx, video in enumerate(
        videos_to_process
    ):
        video_id = video[
            "video_id"
        ]
        if (
            video_id
            in existing_video_ids
        ):
            print(
                "Skipping existing metadata: "
                f"{video_id}"
            )
            continue

        try:
            print(
                f"\n[{idx + 1}/"
                f"{len(videos_to_process)}] "
                f"Detailed metadata: "
                f"{video_id}"
            )
            info = ydl.extract_info(
                video["url"],
                download=False,
            )
            detailed_video = {
                "video_id": info.get(
                    "id",
                    "",
                ),
                "title": info.get(
                    "title",
                    "",
                ),
                "url": video["url"],
                "channel_id": info.get(
                    "channel_id",
                    "",
                ),
                "channel_name": info.get(
                    "channel",
                    "",
                ),
                "video_description": info.get(
                    "description",
                    "",
                ),
                "upload_date": info.get(
                    "upload_date",
                    "",
                ),
                "duration": info.get(
                    "duration",
                    0,
                ),
                "section": video[
                    "section"
                ],
                "thumbnail": video[
                    "thumbnail"
                ],
            }
            new_detailed_videos.append(
                detailed_video
            )
            existing_video_ids.add(
                detailed_video[
                    "video_id"
                ]
            )

            if (
                len(
                    new_detailed_videos
                )
                % SAVE_EVERY
                == 0
            ):

                print(
                    "\nSaving checkpoint "
                    f"after "
                    f"{len(new_detailed_videos)} "
                    "new videos..."
                )

                save_metadata(
                    existing_metadata
                    + new_detailed_videos
                )

            # Optional polite delay
            # time.sleep(0.25)

        except Exception as e:
            print(
                "Failed detailed extraction: "
                f"{video_id}"
            )
            print(e)

combined_metadata = (
    existing_metadata
    + new_detailed_videos
)

save_metadata(
    combined_metadata
)

print(
    "\nTotal metadata entries saved: "
    f"{len(combined_metadata)}"
)

print(
    f"\nSaved JSON: {json_path}"
)

print(
    f"Saved CSV: {csv_path}"
)

print("\nFinished.")