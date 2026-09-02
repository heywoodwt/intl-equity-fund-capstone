"""PCA feature engineering (numpy SVD — no heavy deps).

This is an additive, interpretable orthogonalization: features are z-scored and
rotated into linearly-independent principal components. We keep ALL components
(a full-rank rotation — no information lost), so downstream can use the original
explainable features, the PCs, or both. Explained-variance ratios are returned
so the team can truncate to the top-k components if desired.
"""

import numpy as np
import pandas as pd


def run(df, feature_cols, key="month"):
    """Fit PCA on complete-case rows of `feature_cols` (z-scored).

    Returns:
      components: DataFrame [key, PC1..PCk] — the scores (one row per kept obs).
      loadings:   DataFrame [feature_cols + _explained_var_ratio + _cumulative]
                  x [PC1..PCk] — feature loadings plus variance summary rows.
    """
    # Keep rows with any feature info; drop features missing on all those rows.
    sub = df[[key] + feature_cols].dropna(subset=feature_cols, how="all").reset_index(drop=True)
    feats = [c for c in feature_cols if sub[c].notna().any()]
    X = sub[feats].to_numpy(dtype=float)

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0)  # mean-impute scattered gaps in standardized space

    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    scores = U * S
    pcs = [f"PC{i + 1}" for i in range(scores.shape[1])]

    components = pd.concat(
        [sub[[key]], pd.DataFrame(scores, columns=pcs)], axis=1)

    loadings = pd.DataFrame(Vt.T, index=feats, columns=pcs)
    evr = (S ** 2) / (S ** 2).sum()
    loadings.loc["_explained_var_ratio"] = evr
    loadings.loc["_cumulative"] = np.cumsum(evr)
    return components, loadings
