"""Text-only clean-up helpers for report-ready copies of existing figures.

These functions only change titles, legend labels and annotation text /
placement on an already-drawn Axes. They never touch plotted data, limits
or line styles, so the cleaned figure shares its geometry with the original.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from matplotlib.text import Annotation

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = REPO_ROOT / "ReportFigs" / "cleaned"


def clean_dir() -> Path:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    return CLEAN_DIR


def relabel_legend(ax, mapping: Mapping[str, str], loc="best", fontsize=8, **legend_kwargs):
    """Rebuild the legend using ``mapping`` (old label -> new label).

    Entries appear in the order of ``mapping``; labels absent from the
    mapping are dropped. Duplicate labels keep the last handle, matching
    the ``dict(zip(labels, handles))`` pattern used in the notebooks.
    """
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    missing = [old for old in mapping if old not in by_label]
    if missing:
        raise KeyError(f"legend labels not found on axes: {missing}; available: {list(by_label)}")
    new_handles = [by_label[old] for old in mapping]
    new_labels = [mapping[old] for old in mapping]
    return ax.legend(new_handles, new_labels, loc=loc, fontsize=fontsize, **legend_kwargs)


def retypeset_annotations(
    ax,
    mapping: Mapping[str, str],
    placement_overrides: Optional[Dict[str, Tuple[Tuple[float, float], str, bool]]] = None,
):
    """Replace annotation text via ``mapping`` (old text -> new text).

    ``placement_overrides`` maps an old text to ``((dx, dy), textcoords, arrow)``
    to move a label without changing the annotated point.
    """
    placement_overrides = placement_overrides or {}
    seen = set()
    for ann in ax.texts:
        if not isinstance(ann, Annotation):
            continue
        old = ann.get_text()
        if old not in mapping:
            continue
        seen.add(old)
        ann.set_text(mapping[old])
        if old in placement_overrides:
            xytext, textcoords, arrow = placement_overrides[old]
            ann.set_anncoords(textcoords)
            ann.xyann = xytext
            if not arrow:
                ann.arrowprops = None
                ann.arrow_patch = None
    missing = [old for old in mapping if old not in seen]
    if missing:
        raise KeyError(f"annotations not found on axes: {missing}")
