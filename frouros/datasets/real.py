"""Real datasets module."""

from typing import Any, Optional

import numpy as np
from scipy.io import arff

from frouros.datasets.base import BaseDatasetDownload


class Elec2(BaseDatasetDownload):
    """Elec2 dataset [harries1999splice]_.

    :param file_path: file path for the downloaded file, defaults to None. If None, the file will be downloaded to a temporary file.
    :type file_path: Optional[str]

    :Note:
    Dataset can be downloaded from the following sources (in order of preference):

    - https://nextcloud.ifca.es/index.php/s/2coqgBEpa82boLS/download
    - https://www.openml.org/data/download/2419/electricity-normalized.arff

    :References:

    .. [harries1999splice] Harries, Michael.
        "Splice-2 comparative evaluation: Electricity pricing." (1999).

    :Example:

    >>> from frouros.datasets.real import Elec2
    >>> elec2 = Elec2()
    >>> elec2.download()
    INFO:frouros:Trying to download data from https://nextcloud.ifca.es/index.php/s/2coqgBEpa82boLS/download to /tmp/tmpro3ienx0
    >>> dataset = elec2.load()
    >>> dataset
    array([(0.    , b'2', 0.      , 0.056443, 0.439155, 0.003467, 0.422915, 0.414912, b'UP'),
       (0.    , b'2', 0.021277, 0.051699, 0.415055, 0.003467, 0.422915, 0.414912, b'UP'),
       (0.    , b'2', 0.042553, 0.051489, 0.385004, 0.003467, 0.422915, 0.414912, b'UP'),
       ...,
       (0.9158, b'7', 0.957447, 0.043593, 0.34097 , 0.002983, 0.247799, 0.362281, b'DOWN'),
       (0.9158, b'7', 0.978723, 0.066651, 0.329366, 0.00463 , 0.345417, 0.206579, b'UP'),
       (0.9158, b'7', 1.      , 0.050679, 0.288753, 0.003542, 0.355256, 0.23114 , b'DOWN')],
      dtype=[('date', '<f8'), ('day', 'S1'), ('period', '<f8'), ('nswprice', '<f8'), ('nswdemand', '<f8'), ('vicprice', '<f8'), ('vicdemand', '<f8'), ('transfer', '<f8'), ('class', 'S4')])
    """  # noqa: E501  # pylint: disable=line-too-long

    def __init__(  # noqa: D107
        self,
        file_path: Optional[str] = None,
    ) -> None:
        super().__init__(
            url=[
                "https://nextcloud.ifca.es/index.php/s/2coqgBEpa82boLS/download",
                "https://www.openml.org/data/download/2419/electricity-normalized.arff",
            ],
            file_path=file_path,
        )

    def read_file(self, **kwargs: Any) -> np.ndarray:
        """Read file.

        :param kwargs: additional arguments
        :type kwargs: Any
        :return: read file
        :rtype: numpy.ndarray
        """
        index = kwargs.get("index", 0)
        dataset = arff.loadarff(f=self.file_path)[index]
        return dataset
