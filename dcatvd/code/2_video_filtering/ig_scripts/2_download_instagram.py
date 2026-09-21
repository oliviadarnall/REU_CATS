'''
Download videos from specified Instagram channel

Skip:
- WAV files already downloaded
'''

from playwright.sync_api import sync_playwright
import time
import yt_dlp
import requests
from pathlib import Path

USERNAME = "thatcatbobbie"

DOWNLOAD_WAV_DIR = Path(
    f"downloads/dataset_wav/{USERNAME}"
)

DOWNLOAD_WAV_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with sync_playwright() as p:
    browser = p.chromium.launch_persistent_context(
        "playwright_profile",
        headless=False
    )

    page = browser.new_page()
    page.goto(f"https://www.instagram.com/{USERNAME}/reels/")
    time.sleep(5)
    urls = set()
    last_height = 0

    while True:
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        time.sleep(2)
        links = page.locator("a").evaluate_all(
            "(els) => els.map(e => e.href)"
        )
        for link in links:
            if "/reel/" in link:
                urls.add(link.split("?")[0])
        height = page.evaluate("document.body.scrollHeight")
        if height == last_height:
            break
        last_height = height
    print(f"Found {len(urls)} reels")
    browser.close()

# urls = ['https://www.instagram.com/ritesports/reel/C2kp-yHuYpU/',
# 'https://www.instagram.com/ritesports/reel/DRYOTVHDjbU/',
# 'https://www.instagram.com/ritesports/reel/DXN42Rjia4D/',
# 'https://www.instagram.com/ritesports/reel/DXN5GykidPE/',
# 'https://www.instagram.com/ritesports/reel/DXN5XDhiY9j/',
# 'https://www.instagram.com/ritesports/reel/DYXfVCkJ0-0/',
# 'https://www.instagram.com/ritesports/reel/DaVqAu0J-uu/'
# ]

ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": str(
        DOWNLOAD_WAV_DIR / "%(id)s.%(ext)s"
    ),
    "quiet": False,
    "noplaylist": True,
    "cookiefile": r'C:\Users\Samy\Documents\REU_2026\cat_dataset\ig_scripts\instagram_cookies.txt',
    "postprocessors": [{
        "key": "FFmpegExtractAudio",
        "preferredcodec": "wav",
        "preferredquality": "0",
    }],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    for video_id in urls:
        wav_path = (DOWNLOAD_WAV_DIR / f"{video_id}.wav")

        if wav_path.exists():
            print(f"\nSkipping existing WAV: " f"{video_id}")
            continue

        try:
            print(f"\nDownloading WAV: " f"{video_id}")
            ydl.download(video_id)

        except Exception as e:
            print(f"Failed WAV download: " f"{video_id}")
            print(e)