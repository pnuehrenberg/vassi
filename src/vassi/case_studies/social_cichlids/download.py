import argparse
import os
from pathlib import Path

from ...logging import set_logging_level
from .._utils import download_url

API_TOKEN = os.environ.get("EDMOND_API_TOKEN")


def download_social_cichlids_dataset(
    *,
    output_directory: str | Path = "../../../datasets/social_cichlids",
    download_videos: bool = False,
):
    log = set_logging_level("info", enqueue=False)
    output_directory = Path(output_directory).absolute().resolve()
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
    urls = {
        "cichlids_annotations.csv": f"https://edmond.mpg.de/api/access/datafile/311990?key={API_TOKEN}",
        "cichlids_trajectories.h5": f"https://edmond.mpg.de/api/access/datafile/311993?key={API_TOKEN}"
    }
    log.info(f"Downloading social cichlids trajectories and annotations to {output_directory}")
    log.warning("this requires ~100 MB of disk space")
    for file_name, url in urls.items():
        if (output_path := output_directory / file_name).exists():
            log.info(f"Skipping {file_name} as it already exists ({output_path})")
            continue
        download_url(url, output_path)
    if not download_videos:
        return
    video_urls = {
        "GH010423.MP4": f"https://edmond.mpg.de/api/access/datafile/312077?key={API_TOKEN}",
        "GH010861.MP4": f"https://edmond.mpg.de/api/access/datafile/312083?key={API_TOKEN}",
        "GH013974.MP4": f"https://edmond.mpg.de/api/access/datafile/312078?key={API_TOKEN}",
        "GH019910.MP4": f"https://edmond.mpg.de/api/access/datafile/312080?key={API_TOKEN}",
        "GH030423.MP4": f"https://edmond.mpg.de/api/access/datafile/312082?key={API_TOKEN}",
        "GH030451.MP4": f"https://edmond.mpg.de/api/access/datafile/312079?key={API_TOKEN}",
        "GH030861.MP4": f"https://edmond.mpg.de/api/access/datafile/312084?key={API_TOKEN}",
        "GH039910.MP4": f"https://edmond.mpg.de/api/access/datafile/312081?key={API_TOKEN}",
        "GH039931.MP4": f"https://edmond.mpg.de/api/access/datafile/312085?key={API_TOKEN}",
    }
    video_directory = output_directory / "videos"
    if not video_directory.exists():
        video_directory.mkdir()
    log.info(f"Downloading social cichlids videos to {video_directory}")
    log.warning("this requires 5 GB of disk space")
    for video_file, url in video_urls.items():
        if (video_path := video_directory / video_file).exists():
            log.info(f"Skipping {video_file} as it already exists ({video_path})")
            continue
        download_url(url, video_path)

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory", default="datasets/social_cichlids", type=str
    )
    parser.add_argument("--download-videos", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_directory = Path(args.output_directory)
    download_social_cichlids_dataset(
        output_directory=output_directory,
        download_videos=args.download_videos,
    )


# export EDMOND_API_TOKEN=$(cat ~/.edmond_api_token) && python -m download_social_cichlids_dataset --download-videos
