# Colab notebook

`Free_LLM_Colab_Playground.ipynb` is the primary entry point. It is a clean tutorial notebook with
no committed outputs or credentials. Before publishing a fork, update `REPOSITORY_URL` in
`scripts/build_notebook.py`, rebuild the notebook, and update the Colab badge in the root README.

Rebuild and validate:

```bash
python scripts/build_notebook.py
python scripts/validate_notebook.py
```

The notebook installs this package, detects hardware, builds official llama.cpp with CUDA when
available, and launches the guided UI. GPU execution cannot be validated on a CPU-only machine.

