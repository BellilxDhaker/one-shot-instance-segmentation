"""Command line front end for the benchmark integrity checker.

The checks themselves live in `evaluation.dataset_integrity` so the Colab
notebook can run exactly the same validation before it spends GPU time. This
script is only the shell around them:

    python scripts/check_dataset.py --root data
    python scripts/check_dataset.py --root data --strict --json results/integrity.json

The exit status is `0` when the dataset is scorable and `1` when anything
critical was found, so the script can gate a benchmark run in a shell script or
a CI job.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.dataset_integrity import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
