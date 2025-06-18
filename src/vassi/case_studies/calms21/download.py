import argparse
import os
import shutil
import tempfile
from pathlib import Path

from ...logging import set_logging_level
from .._utils import download_url
from .convert import convert_calms21_sequences


def download_calm21_dataset(
    *,
    output_directory: str | Path,
    remove_taskprog_features: bool = True,
    download_videos: bool = False,
):
    log = set_logging_level("info", enqueue=False)
    output_directory = Path(output_directory).absolute().resolve()
    if not output_directory.exists():
        output_directory.mkdir(parents=True)
    url = "https://data.caltech.edu/records/s0vdx-0k302/files/task1_classic_classification.zip?download=1"
    temp_handle, temp_file = tempfile.mkstemp(
        suffix=".calms21.temp", dir=output_directory
    )
    log.info(f"Downloading CALMS21 trajectories and annotations to {output_directory}")
    log.warning(
        "download and extraction require 1 GB of disk space (~190 MB after conversion if original files are discarded)"
    )
    download_url(url, temp_file)
    shutil.unpack_archive(temp_file, output_directory, format="zip")
    os.close(temp_handle)
    os.remove(temp_file)
    if download_videos:
        log.info(f"Downloading CALMS21 videos to {output_directory}")
        log.warning(
            "download and extraction require 60 GB of disk space (< 30 GB after extraction)"
        )
        url = "https://data.caltech.edu/records/s0vdx-0k302/files/task1_videos_mp4.zip?download=1"
        temp_handle, temp_file = tempfile.mkstemp(
            suffix=".calms21-videos.temp", dir=output_directory
        )
        download_url(url, temp_file)
        shutil.unpack_archive(temp_file, output_directory, format="zip")
        os.close(temp_handle)
        os.remove(temp_file)
    if not remove_taskprog_features:
        return
    for json_file in [
        "taskprog_features_task1_train.json",
        "taskprog_features_task1_test.json",
    ]:
        os.remove(output_directory / "task1_classic_classification" / json_file)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_directory", default="datasets/CALMS21", type=str)
    parser.add_argument("--remove-taskprog-features", action="store_false")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--keep-original", action="store_true")
    parser.add_argument("--download-videos", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_directory = Path(args.output_directory)
    download_calm21_dataset(
        output_directory=output_directory,
        remove_taskprog_features=args.remove_taskprog_features,
        download_videos=args.download_videos,
    )
    if args.download_only:
        exit()
    convert_calms21_sequences(
        output_directory / "task1_classic_classification" / "calms21_task1_train.json",
        output_directory / "task1_classic_classification" / "calms21_task1_test.json",
        output_directory,
    )
    if args.keep_original:
        exit()
    shutil.rmtree(output_directory / "task1_classic_classification")
