"""Execute the submission notebook with this interpreter, never a global kernel."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    notebook_path = ROOT / "LLM_Theatre.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    with tempfile.TemporaryDirectory(prefix="theatre-kernel-") as temp:
        kernel_path = Path(temp) / "theatre"
        kernel_path.mkdir()
        (kernel_path / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                    "display_name": "Agent Theatre (.venv)",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        manager = KernelManager(
            kernel_name="theatre", kernel_spec_manager=KernelSpecManager(kernel_dirs=[temp])
        )
        client = NotebookClient(
            notebook, km=manager, timeout=600, resources={"metadata": {"path": str(ROOT)}}
        )
        client.execute(env={**os.environ, "THEATRE_MODE": "demo"})
    nbformat.write(notebook, notebook_path)
    print(f"Executed {sum(cell.cell_type == 'code' for cell in notebook.cells)} code cells: {notebook_path}")


if __name__ == "__main__":
    main()
