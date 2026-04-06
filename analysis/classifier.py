"""analysis/classifier.py — Parkinsonia presence probability classifier.

Wraps a StandardScaler + LogisticRegression trained on end-member pixels
(presence vs absence) from a single site.  The classifier is intentionally
simple and interpretable: the linear decision boundary maps directly onto the
mechanistic feature understanding documented in research/LONGREACH-STAGE2.md.

Usage
-----
from analysis.classifier import ParkoClassifier

clf = ParkoClassifier(features=["nir_cv", "rec_p", "re_p10"])
clf.fit(train_df, label_col="is_presence")   # label_col: bool or 0/1

# Score any pixel table that has the same feature columns:
scored = clf.score(pixel_df)
# → pixel_df + columns: prob_lr, rank

# Inspect the model:
print(clf.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class ParkoClassifier:
    """Logistic-regression Parkinsonia presence probability classifier.

    Parameters
    ----------
    features:
        Ordered list of feature column names used for training and scoring.
        Defaults to the three primary Stage 2 features.
    max_iter:
        Passed to LogisticRegression.
    random_state:
        Passed to LogisticRegression.
    """

    features: list[str] = field(
        default_factory=lambda: ["nir_cv", "rec_p", "re_p10"]
    )
    max_iter: int = 1000
    random_state: int = 42

    # Set after fit()
    _scaler: StandardScaler = field(default=None, init=False, repr=False)
    _lr: LogisticRegression = field(default=None, init=False, repr=False)
    _n_presence: int = field(default=0, init=False, repr=False)
    _n_absence: int = field(default=0, init=False, repr=False)
    _train_accuracy: float = field(default=float("nan"), init=False, repr=False)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, label_col: str = "is_presence") -> "ParkoClassifier":
        """Fit the classifier on end-member pixels.

        Parameters
        ----------
        df:
            DataFrame containing ``features`` columns and ``label_col``.
            Rows with NaN in any feature column are dropped before fitting.
        label_col:
            Boolean or 0/1 column: True / 1 = presence (Parkinsonia),
            False / 0 = absence (grassland or background).

        Returns
        -------
        self  (for chaining)
        """
        df = df.dropna(subset=self.features).copy()

        y = df[label_col].astype(int).values
        X = df[self.features].values

        self._n_presence = int(y.sum())
        self._n_absence = int((y == 0).sum())

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._lr = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._lr.fit(X_scaled, y)
        self._train_accuracy = float(self._lr.score(X_scaled, y))
        return self

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score pixels with Parkinsonia presence probability.

        Parameters
        ----------
        df:
            DataFrame containing ``features`` columns.  Rows with NaN in any
            feature column receive NaN probability.

        Returns
        -------
        Copy of ``df`` with two new columns appended:
          ``prob_lr``  — predicted probability of Parkinsonia presence [0, 1]
          ``rank``     — integer rank (1 = highest probability)
        """
        self._check_fitted()

        result = df.copy()
        valid = result[self.features].notna().all(axis=1)

        probs = np.full(len(result), np.nan)
        if valid.any():
            X = result.loc[valid, self.features].values
            X_scaled = self._scaler.transform(X)
            probs[valid.values] = self._lr.predict_proba(X_scaled)[:, 1]

        result["prob_lr"] = probs

        # Rank: 1 = highest probability; NaN rows sort last
        rank_vals = pd.Series(probs).rank(ascending=False, method="min", na_option="bottom")
        result["rank"] = rank_vals.values.astype(int)

        return result

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def coefficients(self) -> pd.Series:
        """Return feature coefficients from the fitted logistic regression."""
        self._check_fitted()
        return pd.Series(self._lr.coef_[0], index=self.features, name="coefficient")

    def summary(self) -> str:
        """One-paragraph text summary of the fitted classifier."""
        self._check_fitted()
        coef = self.coefficients()
        lines = [
            f"ParkoClassifier  features={self.features}",
            f"  Training pixels — presence: {self._n_presence}  absence: {self._n_absence}",
            f"  Training accuracy: {self._train_accuracy:.3f}",
            "  Coefficients (standardised):",
        ]
        for feat, val in coef.items():
            lines.append(f"    {feat:12s}  {val:+.4f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if self._lr is None:
            raise RuntimeError("ParkoClassifier has not been fitted. Call .fit() first.")
