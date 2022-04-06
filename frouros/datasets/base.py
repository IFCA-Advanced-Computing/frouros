"""Dataset base module."""

import abc
import tempfile
from typing import Any, List, Optional, Union
import urllib.parse
from pathlib import Path

import requests
import tqdm  # type: ignore
from frouros.datasets.exceptions import (
    DownloadError,
    InvalidURLError,
    ReadFileError,
)
from frouros.utils.logger import logger


class Dataset(abc.ABC):
    """Abstract class representing a downloadable dataset."""

    def __init__(
        self,
        url: Union[str, List[str]],
        file_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """Init method.

        :param url: url or url mirrors from where dataset will be downloaded
        :type url: Union[str, List[str]]
        :param file_path: file path for the downloaded file
        :type file_path: str
        :param verbose: whether more information will be provided
        during download or not
        :type verbose: bool
        """
        self.url = url  # type: ignore
        self.file_path: Optional[Path] = (
            Path(file_path)
            if file_path
            else Path(tempfile.NamedTemporaryFile(delete=False).name)
        )
        self.verbose = verbose
        self.chunk_size = None

    @property
    def chunk_size(self) -> Optional[int]:
        """Chunk size property.

        :return: chunk size to use in writing the dataset
        :rtype: Optional[int]
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value: Optional[int]) -> None:
        """Chunk size setter.

        :param value: value to be set
        :type value: Optional[int]
        """
        self._chunk_size: Optional[int]
        if value:
            if not isinstance(value, int):
                raise TypeError("chunk_size must be int.")
            if value <= 0:
                raise ValueError("chunk_size must be greater than 0.")
            self._chunk_size = value
        else:
            self._chunk_size = None

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
    def url(self) -> List[str]:
        """URL property.

        :return: URL from where dataset will be downloaded
        :rtype: List[str]
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

    @property
    def verbose(self) -> bool:
        """Verbose property.

        :return: URL´s mirrors from where dataset can be downloaded
        :rtype: bool
        """
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool) -> None:
        """Verbose setter.

        :param value: value to be set
        :type value: bool
        """
        self._verbose = value
        self._chunk_size = 1024 if self.verbose else None

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
        request_response = requests.get(url=url, allow_redirects=True, stream=True)
        request_response.raise_for_status()
        return request_response

    def _save_file(self, response: requests.models.Response) -> None:
        try:
            if self.verbose:
                pbar = tqdm.tqdm(unit="B", unit_scale=True, total=len(response.content))
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        self._write_file(content=chunk)
                        pbar.update(n=len(chunk))
            else:
                self._write_file(content=response.content)
        except IOError as e:
            raise e

    def _write_file(self, content: bytes) -> None:
        with open(file=self.file_path, mode="ab") as f:  # type: ignore
            f.write(content)

    def download(self) -> None:
        """Download dataset.

        :raises RequestFileError: Request file exception
        """
        for url in self.url:
            try:
                self._get_file(url=url)
            except requests.exceptions.RequestException as e:
                raise DownloadError(e) from e
            break

    def load(self, **kwargs) -> Any:
        """Load dataset.

        :param kwargs: dict of kwargs
        :type kwargs: dict
        :raises FileNotFoundError: File not found exception
        :raises LoadError: Load file exception
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
