"""Real datasets module."""

from typing import Optional

import numpy as np  # type: ignore
from scipy.io import arff  # type: ignore

from frouros.datasets.base import BaseDatasetDownload


class Elec2(BaseDatasetDownload):
    """Elec2 dataset [harries1999splice]_.

    :References:

    .. [harries1999splice] Harries, Michael.
        "Splice-2 comparative evaluation: Electricity pricing." (1999).
    """

    def __init__(self, file_path: Optional[str] = None) -> None:
        """Init method.

        :param file_path: file path for the downloaded file
        :type file_path: Optional[str]
        """
        super().__init__(
            url=[
                "https://nextcloud.ifca.es/index.php/s/2coqgBEpa82boLS/download",
                "https://www.openml.org/data/download/2419/electricity-normalized.arff",
            ],
            file_path=file_path,
        )

    def read_file(self, **kwargs) -> np.ndarray:
        """Read file.

        :param kwargs: dict of kwargs
        :type kwargs: dict
        :return: read file
        :rtype: numpy.ndarray
        """
        index = kwargs.get("index", 0)
        dataset = arff.loadarff(f=self.file_path)[index]
        return dataset
