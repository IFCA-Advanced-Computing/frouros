"""Dataset base module."""

import abc
from typing import List, Optional
import urllib.parse
from pathlib import Path

import requests
import tqdm  # type: ignore
from frouros.utils.logger import logger


class InvalidFilePathError(Exception):
    """Invalid file path exception."""


class InvalidURLError(Exception):
    """Invalid URL exception."""


class RequestFileError(Exception):
    """Request file exception."""


class Dataset(abc.ABC):
    """Abstract class representing a downloadable dataset.

    :param url: url from where dataset will be downloaded
    :type url: str
    :param filename: filename for the downloaded file
    :type filename: str
    """

    def __init__(self,
                 url: str,
                 file_path: str,
                 url_mirrors: Optional[List[str]] = None,
                 verbose: bool = True):
        """Init method."""
        self._url = url
        self._file_path = Path(file_path)
        self._url_mirrors: Optional[List[str]] = url_mirrors
        self._verbose = verbose
        self._chunk_size: Optional[int] = None

    @property
    def chunk_size(self) -> Optional[int]:
        """Chunk size property.

        :return: chunk size to use in writing the dataset
        :rtype: Optional[int]
        """
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self,
                   value: Optional[int]) -> None:
        """Chunk size setter.

        :param value: value to be set
        :type value: Optional[int]
        """
        if not isinstance(value, int):
            raise TypeError('chunk_size must be int.')
        if value <= 0:
            raise ValueError('chunk_size must be greater than 0.')
        self._chunk_size = value

    @property
    def file_path(self) -> Path:
        """File path property.

        :return: file path for the downloaded dataset
        :rtype: Path
        """
        return self._file_path

    @file_path.setter
    def file_path(self,
                  value: Path) -> None:
        """File path setter.

        :param value: value to be set
        :type value: Path
        """
        try:
            self._file_path = value
        except FileNotFoundError as e:
            raise InvalidFilePathError(e) from e

    @property
    def url(self) -> str:
        """URL property.

        :return: URL from where dataset will be downloaded
        :rtype: str
        """
        return self._url

    @url.setter
    def url(self,
            value: str) -> None:
        """URL setter.

        :param value: value to be set
        :type value: str
        :raises InvalidURLError: Given URL is invalid
        """
        if not self._check_valid_url(url=value):
            raise InvalidURLError(f'{value} is not a valid URL.')
        self._url = value

    @property
    def url_mirrors(self) -> Optional[List[str]]:
        """URL´s mirrors property.

        :return: URL´s mirrors from where dataset can be downloaded
        :rtype: Optional[List[int]]
        """
        return self._url_mirrors

    @url_mirrors.setter
    def url_mirrors(self,
                    value: Optional[List[str]]) -> None:
        """URL´s mirrors setter.

        :param value: value to be set
        :type value: Optional[List[int]]
        :raises InvalidURLError: Given URL is invalid
        """
        if not value:
            self._url_mirrors = None
        else:
            for url in value:
                if not self._check_valid_url(url=url):
                    raise InvalidURLError(f'{url} is not a valid URL.')
            self._url_mirrors = value

    @property
    def verbose(self) -> bool:
        """Verbose property.

        :return: URL´s mirrors from where dataset can be downloaded
        :rtype: bool
        """
        return self._verbose

    @verbose.setter
    def verbose(self,
                value: bool) -> None:
        self._verbose = value
        self._chunk_size = 1024 if self.verbose else None

    @staticmethod
    def _check_valid_url(url: str) -> bool:
        final_url = urllib.parse.urlparse(url=urllib.parse.urljoin(base=url,
                                          url='/'))
        is_valid = (all([final_url.scheme,
                         final_url.netloc,
                         final_url.path]) and
                    len(final_url.netloc.split('.')) > 1)
        return is_valid

    def _get_file(self,
                  url: str) -> None:
        response = self._request_file(url=url)
        self._save_file(response=response)

    def _request_file(self,
                      url: str):
        logger.info('Trying to download data from %s to %s',
                    url,
                    self._file_path)
        try:
            request_response = requests.get(url=url,
                                            allow_redirects=True,
                                            stream=True)
        except requests.exceptions.RequestException as e:
            raise RequestFileError(e) from e
        return request_response

    def _save_file(self,
                   response):
        try:
            if self.verbose:
                pbar = tqdm.tqdm(unit='B',
                                 unit_scale=True,
                                 total=len(response.content))
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        self._write_file(content=chunk)
                        pbar.update(n=len(chunk))
            else:
                self._write_file(content=response.content)
        except IOError as e:
            raise e

    def _write_file(self,
                    content: bytes) -> None:
        with open(file=self._file_path,
                  mode='wb') as f:
            f.write(content)

    @abc.abstractmethod
    def download(self):
        """Download abstract method."""
