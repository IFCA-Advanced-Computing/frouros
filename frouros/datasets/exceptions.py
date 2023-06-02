"""BaseDatasetDownload exception module."""


class DownloadError(Exception):
    """Download exception."""


class InvalidBlockError(Exception):
    """Invalid block exception."""


class InvalidFilePathError(Exception):
    """Invalid file path exception."""


class InvalidURLError(Exception):
    """Invalid URL exception."""


class RequestFileError(Exception):
    """Request file exception."""


class ReadFileError(Exception):
    """Read file exception."""
