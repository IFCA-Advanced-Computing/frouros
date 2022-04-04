"""Real datasets module."""

from typing import Optional

import numpy as np  # type: ignore
from frouros.datasets.base import Dataset
from frouros.datasets.exceptions import AllNaNValuesError


class Elec2(Dataset):
    """Elec2 dataset class."""

    def __init__(self, file_path: Optional[str] = None, verbose: bool = True):
        """Init method.

        :param file_path: file path for the downloaded file
        :type file_path: str
        :param verbose: whether more information will be provided
        during download or not
        :type verbose: bool
        """
        super().__init__(
            url="https://nextcloud.ifca.es/index.php/s/S7xqbSnMjEgegiG/download",
            file_path=file_path,
            verbose=verbose,
        )

    def read_file(self, **kwargs) -> np.ndarray:
        """Read file.

        :return: read file
        :rtype: np.ndarray
        """
        dataset = np.genfromtxt(fname=self.file_path, **kwargs)
        if np.isnan(dataset).all():
            raise AllNaNValuesError
        return dataset
