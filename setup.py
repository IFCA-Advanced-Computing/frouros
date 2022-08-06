"""Package setup file."""

import codecs

from setuptools import find_packages, setup
import toml


def long_description():
    """Read and return README as long description."""
    with codecs.open("README.md", encoding="utf-8-sig") as f:
        return f.read()


# ground truth package metadata is loaded from pyproject.toml
# for context see:
#   - [PEP 621 -- Storing project metadata in pyproject.toml]
#     (https://www.python.org/dev/peps/pep-0621)
pyproject = toml.load("pyproject.toml")


def setup_package():
    """Set up package."""
    setup(
        author_email=pyproject["project"]["authors"][0]["email"],
        author=pyproject["project"]["authors"][0]["name"],
        description=pyproject["project"]["description"],
        extras_require=pyproject["project"]["optional-dependencies"],
        include_package_data=True,
        install_requires=pyproject["project"]["dependencies"],
        keywords=pyproject["project"]["keywords"],
        classifiers=pyproject["project"]["classifiers"],
        # FIXME: Find a way not to hardcode license
        license="BSD-3-Clause",
        long_description=long_description(),
        long_description_content_type="text/markdown",
        maintainer_email=pyproject["project"]["maintainers"][0]["email"],
        maintainer=pyproject["project"]["maintainers"][0]["name"],
        name=pyproject["project"]["name"],
        packages=find_packages(
            where=".",
            exclude=["tests", "tests.*"],
        ),
        project_urls=pyproject["project"]["urls"],
        python_requires=pyproject["project"]["requires-python"],
        setup_requires=pyproject["build-system"]["requires"],
        url=pyproject["project"]["urls"]["repository"],
        version=pyproject["project"]["version"],
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
