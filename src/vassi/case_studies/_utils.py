import urllib.request
from pathlib import Path

from tqdm.auto import tqdm


class DownloadProgressBar(tqdm):
    # https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int | None = None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_directory: str | Path):
    with DownloadProgressBar(
        unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
    ) as t:
        urllib.request.urlretrieve(
            url, filename=output_directory, reporthook=t.update_to
        )
