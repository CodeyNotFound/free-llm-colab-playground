from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def validate(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook.get("nbformat") == 4, "Notebook must use nbformat 4"
    assert isinstance(notebook.get("cells"), list) and notebook["cells"], "Notebook has no cells"
    for index, cell in enumerate(notebook["cells"]):
        assert cell.get("cell_type") in {"markdown", "code"}, f"Invalid cell {index}"
        assert isinstance(cell.get("source"), list), f"Cell {index} source is not a line list"
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"{path.name}:cell-{index}")
            assert cell.get("outputs") == [], f"Cell {index} contains committed output"
            assert cell.get("execution_count") is None, f"Cell {index} execution count is set"
    text = path.read_text(encoding="utf-8")
    for required in ("Hardware", "Hugging Face", "llama.cpp", "Launch", "Security"):
        assert required.lower() in text.lower(), f"Notebook is missing section: {required}"


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "colab/Free_LLM_Colab_Playground.ipynb")
    validate(target)
    print(f"Validated {target}")
