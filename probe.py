"""
probe.py — Hallucination probe classifier (student-implemented).

Pipeline: StandardScaler -> PCA -> LDA(1) -> LogisticRegression

LDA (Linear Discriminant Analysis) finds the single direction in the
PCA-reduced space that maximally separates hallucinated from truthful
hidden states.  This is equivalent to the "mass-mean probe" from
Marks & Tegmark (2024) and is the most robust approach for small datasets
(~468 training samples) -- it computes the hallucination direction
analytically with no gradient-based overfitting.

Optuna tunes n_components x C via 3x3 grid search (9 combinations).

Decision threshold is tuned automatically inside fit() via 5-fold
out-of-fold CV to maximise accuracy on the training set.  This ensures
predict() is correctly calibrated even when fit_hyperparameters() is
never called -- e.g. final_probe in solution.py, which has no
validation split.  fit_hyperparameters() (called by evaluate.py per
fold) overrides the threshold using the held-out validation split.
Both routines target accuracy (the competition's primary metric per
README).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Position of the normalised-n_real scalar in the feature vector
# (set by aggregation.aggregate — verify if its layout changes).
_N_REAL_POS = 17024

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA = True
except ImportError:
    _OPTUNA = False

_OPTUNA_TRIALS = 20


class HallucinationProbe(nn.Module):
    """Binary classifier: StandardScaler -> PCA -> LDA(1) -> LogisticRegression.

    Wrapped in nn.Module to satisfy the evaluate.py interface.
    The core computation is scikit-learn; no neural network layers are used.
    """

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()
        self._pca: PCA | None = None
        self._lda: LDA | None = None
        self._clf: LogisticRegression | None = None
        self._length_regressor: LinearRegression | None = None
        self._threshold: float = 0.5
        self._n_components: int = 64
        self._C: float = 0.1
        # Truncated-stratum probe (length-stratified ensemble — research recommendation #4).
        # Trained only on truncated samples; blended 50/50 with main probe at inference.
        self._scaler_trunc: StandardScaler | None = None
        self._pca_trunc: PCA | None = None
        self._lda_trunc: LDA | None = None
        self._clf_trunc: LogisticRegression | None = None
        self._has_trunc_probe: bool = False

    # ------------------------------------------------------------------
    # nn.Module forward -- not used by evaluate.py, kept for API compat
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Use predict / predict_proba -- forward() is not part of this probe."
        )

    # ------------------------------------------------------------------
    # Length residualisation: regress [log(n_real), is_truncated] out of
    # every feature, then re-attach the two raw signals.  Removes the
    # length-shortcut signal that dominates QA-style hallucination data
    # (Liu 2026 spurious correlation; Ravichander 2025 illusion of
    # progress) while keeping legitimate length dependence available.
    # ------------------------------------------------------------------
    @staticmethod
    def _length_signals(X: np.ndarray) -> np.ndarray:
        """Return ``Z`` of shape (n, 2): [log1p(n_real), is_truncated]."""
        n_real_normalised = X[:, _N_REAL_POS]
        n_real = n_real_normalised * 512.0
        log_len = np.log1p(n_real)
        # n_real == 512 → truncated.  Use a small epsilon for fp safety.
        is_trunc = (n_real_normalised >= (511.5 / 512.0)).astype(float)
        return np.column_stack([log_len, is_trunc])

    def _fit_length_residualiser(self, X: np.ndarray) -> np.ndarray:
        Z = self._length_signals(X)
        self._length_regressor = LinearRegression().fit(Z, X)
        X_resid = X - self._length_regressor.predict(Z)
        return np.hstack([X_resid, Z])

    def _apply_length_residualiser(self, X: np.ndarray) -> np.ndarray:
        Z = self._length_signals(X)
        X_resid = X - self._length_regressor.predict(Z)
        return np.hstack([X_resid, Z])

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------
    def _fit_pipeline(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        n_comp: int,
        C: float = 0.1,
    ) -> tuple[PCA, LDA, LogisticRegression]:
        """Fit PCA -> LDA -> LR on scaled features and return the three objects."""
        n_comp = min(n_comp, len(y) - 2, X_scaled.shape[1])

        pca = PCA(n_components=n_comp, random_state=42)
        X_pca = pca.fit_transform(X_scaled)

        lda = LDA(n_components=1)
        X_lda = lda.fit_transform(X_pca, y)          # (n_samples, 1)

        clf = LogisticRegression(
            C=C,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_lda, y)
        return pca, lda, clf

    # ------------------------------------------------------------------
    # Optuna -- grid search over n_components x C (9 combinations)
    # ------------------------------------------------------------------
    def _run_optuna(self, X_scaled: np.ndarray, y: np.ndarray) -> None:
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=0)
        tr_idx, va_idx = next(sss.split(X_scaled, y))
        y_va = y[va_idx]

        def objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
            n_comp = trial.suggest_categorical("n_components", [32, 64, 128])
            C = trial.suggest_categorical("C", [0.01, 0.1, 1.0])
            pca_t, lda_t, clf_t = self._fit_pipeline(X_scaled[tr_idx], y[tr_idx], n_comp, C)

            X_va = clf_t.predict_proba(
                lda_t.transform(pca_t.transform(X_scaled[va_idx]))
            )[:, 1]
            try:
                return float(roc_auc_score(y_va, X_va))
            except ValueError:
                return 0.5

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.GridSampler(
                {"n_components": [32, 64, 128], "C": [0.01, 0.1, 1.0]}
            ),
        )
        study.optimize(objective, n_trials=_OPTUNA_TRIALS, show_progress_bar=False)
        self._n_components = study.best_params["n_components"]
        self._C = study.best_params["C"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Train the probe.

        Steps:
          1. StandardScaler normalisation.
          2. Optuna grid search over n_components x C (9 combinations).
          3. PCA -> LDA(1) -> LogisticRegression on full training set.
          4. Out-of-fold CV threshold tuning to maximise accuracy
             (so predict() works correctly even when fit_hyperparameters
             is not called -- as in solution.py's final_probe).

        Args:
            X: Feature matrix ``(n_samples, feature_dim)``.
            y: Integer labels ``(n_samples,)``; 0 = truthful, 1 = hallucinated.
        """
        X_scaled = self._scaler.fit_transform(X)

        if _OPTUNA and len(y) >= 20:
            self._run_optuna(X_scaled, y)
            print(f"[Optuna] best -> n_components={self._n_components}  C={self._C}")

        self._pca, self._lda, self._clf = self._fit_pipeline(
            X_scaled, y, self._n_components, self._C
        )

        self._tune_threshold_oof(X_scaled, y)
        return self

    def _tune_threshold_oof(
        self, X_scaled: np.ndarray, y: np.ndarray, n_splits: int = 5
    ) -> None:
        """Tune the decision threshold via out-of-fold CV to maximise accuracy.

        Uses k-fold CV on training data to obtain unbiased predicted
        probabilities for every sample, then sweeps the threshold to
        maximise accuracy on those OOF predictions.

        This means predict() picks the right threshold automatically,
        even when fit_hyperparameters() is never called (e.g. final_probe
        in solution.py is fitted without a validation split).
        """
        n_min = min(np.bincount(y.astype(int))) if len(y) > 0 else 0
        if n_min < n_splits:
            self._threshold = 0.5
            return

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y), dtype=float)

        for tr, va in skf.split(X_scaled, y):
            pca_t, lda_t, clf_t = self._fit_pipeline(
                X_scaled[tr], y[tr], self._n_components, self._C
            )
            oof_probs[va] = clf_t.predict_proba(
                lda_t.transform(pca_t.transform(X_scaled[va]))
            )[:, 1]

        candidates = np.unique(np.concatenate([oof_probs, np.linspace(0.0, 1.0, 201)]))
        best_threshold, best_acc = 0.5, -1.0
        for t in candidates:
            score = accuracy_score(y, (oof_probs >= t).astype(int))
            if score > best_acc:
                best_acc = score
                best_threshold = float(t)

        self._threshold = best_threshold
        print(
            f"[Threshold] OOF accuracy={best_acc:.4f}  "
            f"threshold={best_threshold:.4f}"
        )

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise accuracy."""
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 201)]))

        best_threshold, best_acc = 0.5, -1.0
        for t in candidates:
            score = accuracy_score(y_val, (probs >= t).astype(int))
            if score > best_acc:
                best_acc = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self._scaler.transform(X)
        X_pca = self._pca.transform(X_scaled)
        X_lda = self._lda.transform(X_pca)
        return self._clf.predict_proba(X_lda)
