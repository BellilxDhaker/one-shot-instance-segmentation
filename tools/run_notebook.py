"""Run a generated notebook in order, outside Colab.

Executing the cells in one namespace is the closest local approximation of
`Runtime > Run all`, and it catches the failures that a syntax check misses: a
cell that uses a name from an earlier cell, a module missing from the embedded
bundle, a shape that only appears at runtime.

    python tools/run_notebook.py
    python tools/run_notebook.py --notebook notebooks/03_dataset_evaluation.ipynb
    python tools/run_notebook.py --notebook notebooks/02_one_shot_pipeline.ipynb --stop-after 9

Shell lines (`!pip install ...`) are skipped; everything else runs. Set
`MPLBACKEND=Agg` so the plotting cells do not try to open a window.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_NOTEBOOK = os.path.join(PROJECT_ROOT, "notebooks", "02_one_shot_pipeline.ipynb")


def load_cells(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        notebook = json.load(handle)
    return notebook.get("cells", [])


def is_shell(text: str) -> bool:
    return any(line.strip().startswith(("!", "%")) for line in text.splitlines() if line.strip())


def run(path: str, stop_after: Optional[int] = None, timeout_note: str = "") -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    workspace = os.environ.get("NOTEBOOK_WORKDIR")
    if workspace:
        os.makedirs(workspace, exist_ok=True)
        os.chdir(workspace)

    cells = load_cells(path)
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    print(f"{os.path.relpath(path, PROJECT_ROOT)}: {len(cells)} cells, {len(code_cells)} code")

    namespace: Dict[str, Any] = {"__name__": "__main__", "__file__": path}
    executed = 0
    for index, cell in enumerate(code_cells, start=1):
        text = "".join(cell.get("source", []))
        if is_shell(text):
            print(f"[{index:>2}] shell cell skipped: {text.strip().splitlines()[0][:60]}")
            continue
        if stop_after is not None and index > stop_after:
            print(f"[{index:>2}] stopped before this cell (--stop-after {stop_after})")
            break
        print(f"[{index:>2}] ---- running")
        started = time.perf_counter()
        try:
            compiled = compile(text, f"<cell {index}>", "exec")
            exec(compiled, namespace)  # noqa: S102 - running a notebook on purpose
        except Exception:
            print(f"[{index:>2}] FAILED after {time.perf_counter() - started:.2f}s")
            traceback.print_exc()
            print(f"\ncell {index} source:\n{text}")
            return 1
        executed += 1
        print(f"[{index:>2}] ok ({time.perf_counter() - started:.2f}s)")

    print(f"\n{executed} code cells executed without error{timeout_note}")
    return 0


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", default=DEFAULT_NOTEBOOK)
    parser.add_argument("--stop-after", type=int, default=None,
                        help="execute at most this many code cells")
    parser.add_argument("--workdir", default=None,
                        help="directory to run in (default: the current one)")
    arguments = parser.parse_args(argv)
    if arguments.workdir:
        os.environ["NOTEBOOK_WORKDIR"] = arguments.workdir
    # Resolved before the run chdir's into the workspace, so a relative
    # --notebook still points at the repository when --workdir is given.
    notebook = arguments.notebook if os.path.isabs(arguments.notebook) else os.path.abspath(arguments.notebook)
    return run(notebook, arguments.stop_after)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
