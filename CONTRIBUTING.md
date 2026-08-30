# Contributing

Thank you for improving Free LLM Colab Playground.

1. Open an issue for substantial behavior or architecture changes.
2. Create a focused branch and keep generated credentials/model files out of commits.
3. Install development dependencies with `python -m pip install -e ".[dev]"`.
4. Run `ruff check .`, `pytest`, and `python scripts/validate_notebook.py`.
5. Rebuild the notebook with `python scripts/build_notebook.py` after changing its source.
6. Explain what was tested locally and whether a real Colab CUDA runtime was used.

Changes to fitting heuristics must include tests and avoid fake precision. New telemetry must identify
whether it is measured, parsed from the backend, or estimated. UI features must work—do not add inert
buttons or placeholder controls.

By contributing, you agree that your contribution is licensed under Apache-2.0.

