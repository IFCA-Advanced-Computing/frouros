"""Real datasets module."""

from typing import Optional

import numpy as np  # type: ignore
from scipy.io import arff
from frouros.datasets.base import Dataset


class Elec2(Dataset):
    """Elec2 dataset class."""

    def __init__(self, file_path: Optional[str] = None, verbose: bool = True) -> None:
        """Init method.

        :param file_path: file path for the downloaded file
        :type file_path: Optional[str]
        :param verbose: whether more information will be provided
        during download or not
        :type verbose: bool
        """
        super().__init__(
            url=[
                "https://www.openml.org/data/download/2419/electricity-normalized.arff",
                "https://nextcloud.ifca.es/index.php/s/2coqgBEpa82boLS/download",
            ],
            file_path=file_path,
            verbose=verbose,
        )

    def read_file(self, **kwargs) -> np.ndarray:
        """Read file.

        # :param kwargs: dict of kwargs
        # :type kwargs: dict
        :return: read file
        :rtype: numpy.ndarray
        """
        index = kwargs.get("index", 0)
        dataset = arff.loadarff(f=self.file_path)[index]
        return dataset
