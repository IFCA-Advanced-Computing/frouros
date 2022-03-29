"""Real datasets module."""

from typing import Optional

import numpy as np  # type: ignore
from frouros.datasets.base import Dataset


class Elec2(Dataset):
    """Elec2 dataset class."""

    def __init__(self,
                 file_path: Optional[str] = None):
        """Init method.

        :param file_path: file path for the downloaded file
        :type file_path: str
        """
        super().__init__(url=('https://www.openml.org/data/get_csv/'
                              '2419/electricity-normalized.arff'),
                         file_path=file_path,
                         url_mirrors=None)

    def read_file(self,
                  **kwargs) -> np.ndarray:
        """Read file.

        :return: read file
        :rtype: np.ndarray
        """
        dataset = np.genfromtxt(fname=self.file_path,
                                **kwargs)
        return dataset
