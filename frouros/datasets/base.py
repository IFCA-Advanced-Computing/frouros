"""Dataset base module."""

import abc
import tempfile
from typing import (Any,
                    List,
                    Optional)
import urllib.parse
from pathlib import Path

import requests
import tqdm  # type: ignore
from frouros.datasets.exceptions import (DownloadError,
                                         InvalidFilePathError,
                                         InvalidURLError,
                                         RequestFileError,
                                         LoadError)
from frouros.utils.logger import logger


class Dataset(abc.ABC):
    """Abstract class representing a downloadable dataset."""

    def __init__(self,
                 url: str,
                 file_path: Optional[str] = None,
                 url_mirrors: Optional[List[str]] = None,
                 verbose: bool = True):
        """Init method.

        :param url: url from where dataset will be downloaded
        :type url: str
        :param file_path: filename for the downloaded file
        :type file_path: str
        :param url_mirrors: url mirrors from where dataset can be downloaded
        :type url_mirrors: Optional[List[str]]
        :param verbose: whether more information will be provided
        during download or not
        :type verbose: bool
        :raises class:`frouros.datasets.exceptions.DownloadError`:
        Download file exception
        """
        self._url = url
        self._file_path = Path(file_path) \
            if file_path \
            else Path(tempfile.NamedTemporaryFile(delete=False).name)
        self._url_mirrors: Optional[List[str]] = url_mirrors
        self._verbose = verbose
        self._chunk_size: Optional[int] = None

        try:
            self.download()
        except DownloadError as e:
            raise e

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
        :raises class:`frouros.datasets.exceptions.InvalidURLError`:
        Invalid URL exception
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
        :raises class:`frouros.datasets.exceptions.InvalidURLError`:
        Invalid URL exception
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
        """Verbose setter.

        :param value: value to be set
        :type value: bool
        """
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

    def _remove_temporal_file(self) -> None:
        self.file_path.unlink()
        self.file_path = None

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
        with open(file=self.file_path,
                  mode='wb') as f:
            f.write(content)

    def download(self) -> None:
        """Download dataset.

        :raises class:`frouros.datasets.exceptions.RequestFileError`:
        Request file exception
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

    def load(self,
             **kwargs) -> Any:
        """Load dataset.

        :raises class:`FileNotFoundError`:
        File not found exception
        :raises class:`frouros.datasets.exceptions.LoadError`:
        Load file exception
        :return: loaded dataset
        :rtype: Any
        """
        if not self.file_path:
            raise FileNotFoundError('Missing file to be loaded. '
                                    'Try using download() before '
                                    'load () method.')
        try:
            dataset = self.read_file(**kwargs)
        except LoadError as e:
            raise e
        self._remove_temporal_file()
        return dataset

    @abc.abstractmethod
    def read_file(self,
                  **kwargs) -> Any:
        """Read file abstract method."""
