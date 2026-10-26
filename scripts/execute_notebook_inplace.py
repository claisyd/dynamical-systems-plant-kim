#!/usr/bin/env python3
"""Execute a notebook in place with nbclient (clears stale outputs first).

``jupyter nbconvert --execute`` refuses some older notebooks in this repo
because legacy stream outputs lack the required ``name`` field; clearing
outputs before execution avoids that validation failure.

Usage: python scripts/execute_notebook_inplace.py NOTEBOOK [KERNEL_NAME]
The kernel runs with the notebook's own directory as CWD (as in Jupyter).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    nb_path = Path(sys.argv[1]).resolve()
    kernel = sys.argv[2] if len(sys.argv) > 2 else None
    nb = nbformat.read(nb_path, as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    kwargs = dict(timeout=-1, resources={"metadata": {"path": str(nb_path.parent)}})
    if kernel:
        kwargs["kernel_name"] = kernel
    kernelspec = nb.metadata.get("kernelspec")
    t0 = time.time()
    NotebookClient(nb, **kwargs).execute()
    if kernelspec is not None:
        nb.metadata["kernelspec"] = kernelspec
    nbformat.write(nb, nb_path)
    print(f"executed {nb_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
