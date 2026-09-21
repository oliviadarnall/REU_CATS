'''
Download videos whose thumbnails passed the cat filter

Skip:
- MP4 videos already downloaded
- WAV files already downloaded
- Videos longer than 20 minutes
'''

import json
import pandas as pd
import yt_dlp

from pathlib import Path

#Configure the Channel and the paths

CHANNEL_ID = "UCgkOus3F_UXYcLMBIf39L3Q"

CSV_PATH = Path(
    f"metadata/{CHANNEL_ID}/thumbnail_results.csv"
)

JSON_PATH = Path(
    f"metadata/{CHANNEL_ID}/video_metadata.json"
)

DOWNLOAD_DIR = Path(
    f"downloads/dataset_mp4/{CHANNEL_ID}"
)

DOWNLOAD_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DOWNLOAD_WAV_DIR = Path(
    f"downloads/dataset_wav/{CHANNEL_ID}"
)

DOWNLOAD_WAV_DIR.mkdir(
    parents=True,
    exist_ok=True
)

#Maximum allowed duration
MAX_DURATION_SECONDS = 20 * 60

#Load the thumbnail results CSV

df = pd.read_csv(CSV_PATH)

#Keep only accepted thumbnails
df = df[
    (df["match"] == "True")
    | (df["match"] == True)
]

#Thumbnail filename is the video id
accepted_video_ids = set(
    df["thumbnail"]
    .str.replace(".jpg", "", regex=False)
)

print(
    f"Accepted videos: "
    f"{len(accepted_video_ids)}"
)

#Load the video metadata
with open(
    JSON_PATH,
    "r",
    encoding="utf-8"
) as f:

    videos = json.load(f)

#Filter the videos
videos_to_download = []

for v in videos:
    if v["video_id"] not in accepted_video_ids:
        continue

    duration = v.get("duration", 0)

    # Skip videos over 20 minutes
    if duration > MAX_DURATION_SECONDS:
        minutes = duration / 60
        print(
            f"Skipping long video "
            f"({minutes:.1f} min): "
            f"{v['video_id']}"
        )
        continue
    videos_to_download.append(v)

print(
    f"Videos matched to metadata: "
    f"{len(videos_to_download)}"
)

# MP4 DOWNLOADS

print("MP4 DOWNLOADS")

#Set the yt-dlp options
ydl_opts = {
    "format": "mp4/best",
    "outtmpl": str(
        DOWNLOAD_DIR / "%(id)s.%(ext)s"
    ),
    "quiet": False,
    "noplaylist": True,
    "cookiefile": "cookies.txt"
}



#Download the videos
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for video in videos_to_download:
        video_id = video["video_id"]

        # Check if already downloaded
        existing_mp4 = list(
            DOWNLOAD_DIR.glob(f"{video_id}.*")
        )

        if existing_mp4:
            print(
                f"\nSkipping existing MP4: "
                f"{video_id}"
            )
            continue

        try:
            print(
                f"\nDownloading MP4: "
                f"{video['title']}"
            )
            ydl.download([video["url"]])

        except Exception as e:
            print(
                f"Failed MP4 download: "
                f"{video_id}"
            )
            print(e)

print("\nFinished .mp4 downloads.")

# WAV DOWNLOADS

print("WAV DOWNLOADS")

#Set the yt-dlp options
ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": str(
        DOWNLOAD_WAV_DIR / "%(id)s.%(ext)s"
    ),
    "quiet": False,
    "noplaylist": True,
    "cookiefile": "cookies.txt",
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
        "preferredquality": "0",
    }],
}

#Download the videos
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for video in videos_to_download:
        video_id = video["video_id"]

        # Check if WAV already exists
        wav_path = (DOWNLOAD_WAV_DIR / f"{video_id}.wav")

        if wav_path.exists():
            print(f"\nSkipping existing WAV: " f"{video_id}")
            continue

        try:
            print(f"\nDownloading WAV: " f"{video['title']}")
            ydl.download([video["url"]])

        except Exception as e:
            print(f"Failed WAV download: " f"{video_id}")
            print(e)

print("\nFinished .wav downloads.")