# Contributing to LeggedManip Lab

Thank you for your interest in contributing! This document outlines the process for contributing to the project.

## Getting Started

1. Fork the repository and clone it locally.
2. Install the project in development mode:

```bash
cd LeggedManip_Lab
pip install -e source/LeggedManip_Lab
```

3. Install pre-commit hooks:

```bash
pre-commit install
```

## Development Workflow

1. Create a new branch for your feature or bug fix.
2. Make your changes and ensure they pass linting (`ruff check`) and formatting (`ruff format`).
3. Write or update tests if applicable.
4. Commit your changes with a descriptive message.
5. Push your branch and open a pull request.

## Code Style

- Python code follows the `ruff` configuration defined in `pyproject.toml`.
- Line length limit is 120 characters.
- Use Google-style docstrings.
- Keep comments in English.

## Pull Request Guidelines

- Keep PRs focused on a single change.
- Ensure all pre-commit checks pass.
- Update documentation if your changes affect the public API.
- Describe what your change does and why.

## Reporting Issues

- Use the GitHub issue tracker.
- Include steps to reproduce, expected behavior, and actual behavior.
- Include your environment details (OS, Python version, Isaac Lab version).

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
