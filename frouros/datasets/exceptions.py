"""Dataset exception module."""


class DownloadError(Exception):
    """Download exception."""


class InvalidFilePathError(Exception):
    """Invalid file path exception."""


class InvalidURLError(Exception):
    """Invalid URL exception."""


class RequestFileError(Exception):
    """Request file exception."""


class LoadError(Exception):
    """Load exception."""
