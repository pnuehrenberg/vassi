import argparse
import os
import shutil
import tempfile
import urllib.request

from .convert_calms21_dataset import convert_calms21_sequences
from tqdm.auto import tqdm


class DownloadProgressBar(tqdm):
    # https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_directory: str):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(
            url, filename=output_directory, reporthook=t.update_to
        )


def download_calm21_dataset(
    *,
    output_directory: str = "../../datasets/CALMS21",
    remove_taskprog_features: bool = True,
):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    url = "https://data.caltech.edu/records/s0vdx-0k302/files/task1_classic_classification.zip?download=1"
    _, temp_file = tempfile.mkstemp(suffix=".calms21.temp", dir=output_directory)
    download_url(url, temp_file)
    shutil.unpack_archive(temp_file, output_directory, format="zip")
    os.remove(temp_file)
    if not remove_taskprog_features:
        return
    for json_file in [
        "taskprog_features_task1_train.json",
        "taskprog_features_task1_test.json",
    ]:
        os.remove(
            os.path.join(
                output_directory, "task1_classic_classification", json_file
            )
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_directory", default="../../datasets/CALMS21", type=str
    )
    parser.add_argument("--remove-taskprog-features", action="store_false")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--keep-original", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_calm21_dataset(
        output_directory=args.output_directory,
        remove_taskprog_features=args.remove_taskprog_features,
    )
    if args.download_only:
        exit()
    convert_calms21_sequences(
        os.path.join(
            args.output_directory,
            "task1_classic_classification",
            "calms21_task1_train.json",
        ),
        os.path.join(
            args.output_directory,
            "task1_classic_classification",
            "calms21_task1_test.json",
        ),
        args.output_directory,
    )
    if args.keep_original:
        exit()
    shutil.rmtree(os.path.join(args.output_directory, "task1_classic_classification"))
