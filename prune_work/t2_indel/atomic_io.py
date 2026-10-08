"""Atomic result writers: write to a sibling temp file, then os.replace, so a killed job never leaves a truncated file that a resume check
(os.path.isfile) would count as done (review finding, RAW 45-46)."""
import csv
import os

import numpy as np


def atomic_savez(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def atomic_csv(path, fieldnames, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
