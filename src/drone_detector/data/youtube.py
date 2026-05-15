"""Optional helper: download a list of YouTube URLs into a label folder.

Usage:
    python -m drone_detector.data.youtube audio/shahed.urls.txt audio/shahed
"""

from __future__ import annotations

import sys
from pathlib import Path

import imageio_ffmpeg
import yt_dlp


def download_urls(url_file: Path, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = [
        line.strip()
        for line in url_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not urls:
        return []

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s__%(title).80B.%(ext)s"),
        "noplaylist": False,
        "quiet": False,
        "ignoreerrors": True,
        "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(urls)
    return urls


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: python -m drone_detector.data.youtube <urls.txt> <out_dir>")
        return 2
    url_file = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    if not url_file.exists():
        print(f"URL file not found: {url_file}")
        return 1
    downloaded = download_urls(url_file, out_dir)
    print(f"Downloaded {len(downloaded)} item(s) into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
