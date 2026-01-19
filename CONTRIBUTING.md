# How to contribute

Frouros is an open-source project. Anyone with an interest in the project can join the community and contribute to it in different ways. The following sections describe how you can contribute.

## Adding a feature or solving a bug

Recommended steps for first time contributors:

1. Fork repository on GitHub.
2. Set up develop environment (it is not mandatory, but we highly recommend the use of a [virtual environment](https://docs.python.org/3.11/library/venv.html)):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
   ```
3. Download and install development version:
    ```bash
    git clone https://github.com/<ACCOUNT>/frouros  ## Replace <ACCOUNT> with your GitHub account
    cd frouros
    git checkout main
    git pull
    pip install -e '.[dev]'
   ```
4. (Optional but recommended) Install pre-commit hooks:
    ```bash
    pip install pre-commit
    pre-commit install
    ```
5. After adding and committing your fix or feature, ensure that code coverage is at least 90% (otherwise the PR will be rejected) and that linting is successfully executed using the following command:
    ```bash
   tox
   ```
6. Create a pull request to the original repository.

## Reporting a bug

1. Check that there is not an [issue](https://github.com/IFCA-Advanced-Computing/frouros/issues) that currently highlights the bug or a [pull request](https://github.com/IFCA-Advanced-Computing/frouros/pulls) that solves it.
2. Create an [issue](https://github.com/IFCA-Advanced-Computing/frouros/issues/new) in GitHub.
