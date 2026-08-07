from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def test_quickstart_notebook_executes(tmp_path):
    nb = nbformat.read(ROOT / "examples" / "quickstart.ipynb", as_version=4)
    client = NotebookClient(nb, timeout=300, kernel_name="python3",
                            resources={"metadata": {"path": str(tmp_path)}})
    client.execute()  # raises CellExecutionError on any failing cell
