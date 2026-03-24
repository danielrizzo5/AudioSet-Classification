"""Download and trim AudioSet audio clips using yt-dlp and ffmpeg."""

import os
import subprocess

import pandas as pd
from loguru import logger

from audioset_classification.data.csv_loader import segment_key


def audio_filename(ytid: str, start: float, end: float) -> str:
    """Return the deterministic WAV filename for a clip."""
    return f"{segment_key(ytid, start, end)}.wav"


def audio_path(ytid: str, start: float, end: float, audio_dir: str) -> str:
    """Return the full path for a clip's WAV file."""
    return os.path.join(audio_dir, audio_filename(ytid, start, end))


def download_clip(
    ytid: str,
    start: float,
    end: float,
    audio_dir: str,
    sample_rate: int = 16000,
) -> str | None:
    """Download and trim one AudioSet clip.

    Uses yt-dlp to fetch audio and ffmpeg to trim to [start, end] and
    resample to mono WAV at sample_rate. Returns the output path on success,
    None if the clip could not be downloaded.

    Args:
        ytid: YouTube video ID.
        start: Start time in seconds.
        end: End time in seconds.
        audio_dir: Directory to write WAV files.
        sample_rate: Target sample rate in Hz.
    """
    os.makedirs(audio_dir, exist_ok=True)
    out_path = audio_path(ytid, start, end, audio_dir)

    if os.path.exists(out_path):
        return out_path

    url = f"https://www.youtube.com/watch?v={ytid}"
    duration = end - start

    # yt-dlp stderr is suppressed (2>/dev/null) to silence the broken-pipe noise
    # that occurs when ffmpeg closes the pipe after trimming while yt-dlp is
    # still streaming. ffmpeg errors remain visible.
    shell_cmd = (
        f"yt-dlp --quiet --no-warnings --format bestaudio --output - {url} 2>/dev/null"
        f" | ffmpeg -hide_banner -loglevel error -i pipe:0"
        f" -ss {start} -t {duration} -ac 1 -ar {sample_rate} -f wav {out_path}"
    )
    result = subprocess.run(shell_cmd, shell=True)

    if result.returncode != 0 or not os.path.exists(out_path):
        logger.warning(f"Failed to download {ytid} [{start:.1f}-{end:.1f}]")
        return None

    return out_path


def download_clips(
    df: pd.DataFrame,
    audio_dir: str,
    sample_rate: int = 16000,
) -> pd.DataFrame:
    """Download all clips in a segments DataFrame.

    Returns the DataFrame with an added 'audio_path' column. Rows that
    failed to download have None in that column.

    Args:
        df: Segments DataFrame with ytid, start_seconds, end_seconds columns.
        audio_dir: Directory to write WAV files.
        sample_rate: Target sample rate in Hz.
    """
    records = df.to_dict("records")
    paths = []
    for i, record in enumerate(records):
        ytid = str(record["ytid"])
        start = float(record["start_seconds"])
        end = float(record["end_seconds"])
        logger.info(f"[{i + 1}/{len(records)}] {ytid} [{start:.1f}-{end:.1f}]")
        path = download_clip(ytid, start, end, audio_dir, sample_rate)
        paths.append(path)

    return df.assign(audio_path=paths)
