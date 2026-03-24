# Contributing to HASTE

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/<your-org>/HAST.git
cd HAST
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Follow PEP 8
- Use type hints for function signatures
- Keep modules focused: model logic in `models/`, data loading in `data/`
- Write docstrings for public classes and functions

## Pull Requests

1. Fork the repo and create a feature branch from `main`
2. Add tests for new functionality
3. Ensure `pytest tests/ -v` passes
4. Submit a pull request with a clear description of changes

## Reporting Issues

Open a GitHub issue with:
- A clear title and description
- Steps to reproduce (if applicable)
- Python/PyTorch version and OS
