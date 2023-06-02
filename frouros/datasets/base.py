"""Base dataset module."""

import abc
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np  # type: ignore
import requests

from frouros.datasets.exceptions import (
    DownloadError,
    InvalidURLError,
    ReadFileError,
)
from frouros.utils.logger import logger


class BaseDatasetDownload(abc.ABC):
    """Abstract class representing a downloadable dataset."""

    def __init__(
        self,
        url: Union[str, List[str]],
        file_path: Optional[str] = None,
    ) -> None:
        """Init method.

        :param url: url or url mirrors from where dataset will be downloaded
        :type url: Union[str, List[str]]
        :param file_path: file path for the downloaded file
        :type file_path: str
        """
        self.url = url  # type: ignore
        self.file_path: Optional[Path] = (
            Path(file_path)
            if file_path
            else Path(tempfile.NamedTemporaryFile(delete=False).name)
        )

    @property
    def file_path(self) -> Optional[Path]:
        """File path property.

        :return: file path for the downloaded dataset
        :rtype: Optional[Path]
        """
        return self._file_path

    @file_path.setter
    def file_path(self, value: Optional[Path]) -> None:
        """File path setter.

        :param value: value to be set
        :type value: Optional[Path]
        """
        self._file_path = value

    @property
    def url(self) -> Union[str, List[str]]:
        """URL property.

        :return: URL from where dataset will be downloaded
        :rtype: Union[str, List[str]]
        """
        return self._url

    @url.setter
    def url(self, value: Union[str, List[str]]) -> None:
        """URL setter.

        :param value: value to be set
        :type value: Union[str, List[str]]
        :raises InvalidURLError: Invalid URL exception
        """
        urls = [value] if isinstance(value, str) else value
        for url in urls:
            if not self._check_valid_url(url=url):
                raise InvalidURLError(f"{value} is not a valid URL.")
        self._url = urls

    @staticmethod
    def _check_valid_url(url: str) -> bool:
        final_url = urllib.parse.urlparse(url=urllib.parse.urljoin(base=url, url="/"))
        is_valid = (
            all([final_url.scheme, final_url.netloc, final_url.path])
            and len(final_url.netloc.split(".")) > 1
        )
        return is_valid

    def _get_file(self, url: str) -> None:
        response = self._request_file(url=url)
        self._save_file(response=response)

    def _remove_temporal_file(self) -> None:
        self.file_path.unlink()  # type: ignore
        self.file_path = None

    def _request_file(self, url: str) -> requests.models.Response:
        logger.info("Trying to download data from %s to %s", url, self._file_path)
        request_head = requests.head(url=url)
        if not request_head.ok:
            raise requests.exceptions.RequestException()
        request_response = requests.get(url=url, stream=True)
        request_response.raise_for_status()
        return request_response

    def _save_file(self, response: requests.models.Response) -> None:
        try:
            self._write_file(content=response.content)
        except IOError as e:
            raise e

    def _write_file(self, content: bytes) -> None:
        with open(file=self.file_path, mode="ab") as f:  # type: ignore
            f.write(content)

    def download(self) -> None:
        """Download dataset.

        :raises DownloadError: Download exception
        """
        for url in self.url:
            try:
                self._get_file(url=url)
                break
            except requests.exceptions.RequestException:
                logger.warning("File cannot be downloaded from %s", url)
        else:
            raise DownloadError("File cannot be downloaded from any of the urls.")

    def load(self, **kwargs) -> Any:
        """Load dataset.

        :param kwargs: dict of kwargs
        :type kwargs: dict
        :raises FileNotFoundError: File not found exception
        :raises ReadFileError: Read file exception
        :return: loaded dataset
        :rtype: Any
        """
        if not self.file_path:
            raise FileNotFoundError(
                "Missing file to be loaded. "
                "Try using download() before "
                "load () method."
            )
        try:
            dataset = self.read_file(**kwargs)
        except IndexError as e:
            raise ReadFileError(e) from e
        self._remove_temporal_file()
        return dataset

    @abc.abstractmethod
    def read_file(self, **kwargs) -> Any:
        """Read file abstract method.

        :param kwargs: dict of kwargs
        :type kwargs: dict
        :return: read file
        :rtype: Any
        """

    def __repr__(self) -> str:
        """Repr method.

        :return: repr value
        :rtype: str
        """
        return (
            f"{self.__class__.__name__}"
            f"(url={self.url}, file_path='{self.file_path}')"
        )


class BaseDatasetGenerator(abc.ABC):
    """Abstract class representing a dataset generator."""

    def __init__(self, seed: Optional[int] = None) -> None:
        """Init method.

        :param seed: seed value
        :type seed: Optional[int]
        """
        try:
            np.random.seed(seed=seed)
        except (TypeError, ValueError) as e:
            raise e

    def __repr__(self) -> str:
        """Repr method.

        :return: repr value
        :rtype: str
        """
        return f"{self.__class__.__name__}()"
