"""Real datasets module."""

import requests
from frouros.datasets.base import Dataset


class Elec2(Dataset):
    """Elec2 dataset class.

    :param file_path: filepath for the downloaded file
    :type file_path: str
    """

    def __init__(self,
                 file_path: str):
        """Init method."""
        super().__init__(url=('https://www.openml.org/data/get_csv/'
                              '2419/electricity-normalized.arff'),
                         file_path=file_path,
                         url_mirrors=None)

    def download(self):
        """Download dataset.

        :raises class:`frouros.datasets.base.RequestFileError`:
        Request file exception
        :return: None
        """
        try:
            self._get_file(url=self._url)
        except (requests.exceptions.RequestException,
                IOError) as e:
            if self.url_mirrors:
                for url_mirror in self._url_mirrors:
                    self._get_file(url=url_mirror)
                    break
                else:
                    raise e
            else:
                raise e
