"""
splitting.py — Train / validation / test split utilities (student-implementable).

``split_data`` receives the label array ``y`` and, optionally, the full
DataFrame ``df`` (for group-aware splits).  It must return a list of
``(idx_train, idx_val, idx_test)`` tuples of integer index arrays.

Contract
--------
* ``idx_train``, ``idx_val``, ``idx_test`` are 1-D NumPy arrays of integer
  indices into the full dataset.
* ``idx_val`` may be ``None`` if no separate validation fold is needed.
* All indices must be non-overlapping; together they must cover every sample.
* Return a **list** — one element for a single split, K elements for k-fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def _composite_strata(
    y: np.ndarray, df: pd.DataFrame | None
) -> np.ndarray:
    """Composite stratum label: encodes (label, length-quartile, truncation).

    Hallucinated samples are 2× longer (median 99 vs 44 tokens) and 3×
    more often truncated (27% vs 10%) than truthful ones; the test set
    has 30% truncation, well above the train average (22%).  Stratifying
    on the joint distribution makes each fold's train and test sub-sets
    length- and truncation-balanced, giving a more honest CV estimate
    of how the probe will generalise to the unlabelled competition test.
    """
    if df is None or "response" not in df.columns:
        return y.astype(int)

    response_lengths = df["response"].astype(str).str.len().to_numpy()
    prompt_lengths = df["prompt"].astype(str).str.len().to_numpy()
    # Char-level proxy for token count.  At ~3.5 chars/token combined
    # length > 1800 chars is a reasonable proxy for the 512-token cap.
    is_truncated = ((response_lengths + prompt_lengths) > 1800).astype(int)

    quartile_bins = pd.qcut(
        response_lengths, q=4, labels=False, duplicates="drop"
    ).astype(int)

    # Pack (y, quartile, trunc) into a single integer.  At most
    # 2 * 4 * 2 = 16 strata.
    composite = (
        y.astype(int) * 8 + quartile_bins.astype(int) * 2 + is_truncated
    )

    # Drop rare strata that would prevent 5-fold splitting (need >= 5).
    counts = pd.Series(composite).value_counts()
    rare = set(counts[counts < 5].index.tolist())
    if rare:
        composite = np.where(
            np.isin(composite, list(rare)), y.astype(int), composite
        )
    return composite


def split_data(
    y: np.ndarray,
    df: pd.DataFrame | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split dataset indices into train, validation, and test subsets.

    Uses Stratified 5-Fold CV with a *composite* stratum that encodes the
    label, response-length quartile, and truncation flag.  This balances
    not just class ratios but also the length / truncation distribution
    across folds — addressing the train-test length drift that limits
    CV-estimate honesty (Liu 2026; Ravichander EMNLP 2025).

    Args:
        y:            Label array of shape ``(N,)`` with values in ``{0, 1}``.
        df:           Optional DataFrame with ``prompt`` and ``response``
                      columns; used to compute the composite strata.
        test_size:    Ignored — fold size is determined by n_splits=5 (~20 %).
        val_size:     Fraction of the non-test portion reserved for validation.
        random_state: Random seed for reproducible splits.

    Returns:
        A list of 5 ``(idx_train, idx_val, idx_test)`` tuples.
    """
    strata = _composite_strata(y, df)

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    idx = np.arange(len(y))
    splits: list[tuple[np.ndarray, np.ndarray | None, np.ndarray]] = []

    for fold_train_val, fold_test in kf.split(idx, strata):
        # Inner train/val split also stratified on the composite key
        # (falls back to y if some stratum is too small).
        try:
            fold_train, fold_val = train_test_split(
                fold_train_val,
                test_size=val_size,
                random_state=random_state,
                stratify=strata[fold_train_val],
            )
        except ValueError:
            fold_train, fold_val = train_test_split(
                fold_train_val,
                test_size=val_size,
                random_state=random_state,
                stratify=y[fold_train_val],
            )
        splits.append((fold_train, fold_val, fold_test))

    return splits
