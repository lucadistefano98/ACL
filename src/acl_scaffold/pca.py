from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

MECH_VARS_ALL = ["σmax","εf","Fmax","SL","ST","EL","ET","T","εTmax"]

def run_pca(dati: pd.DataFrame,
            mech_vars: list[str] | None = None,
            group_col: str = "Scaffold",
            n_components: int = 5):
    """
    PCA sulle variabili meccaniche (Z-score).
    Ritorna: scores, loadings, explained
    """
    if mech_vars is None:
        mech_vars = [c for c in MECH_VARS_ALL if c in dati.columns]

    df = dati[[group_col] + mech_vars].dropna().copy()
    if df.empty or len(df) < 3 or len(mech_vars) < 2:
        raise ValueError("Dati insufficienti per PCA.")

    X = df[mech_vars].values
    labels = df[group_col].astype(str).values

    # Standardizzazione
    Z = StandardScaler().fit_transform(X)

    # PCA
    n_comp = min(n_components, Z.shape[0], Z.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    scores = pca.fit_transform(Z)  # shape (n_samples, n_comp)

    # Scores dataframe
    scores_cols = [f"PC{i+1}" for i in range(n_comp)]
    scores_df = pd.DataFrame(scores, columns=scores_cols)
    scores_df.insert(0, group_col, labels)

    # Loadings: feature x component
    loadings = pca.components_.T  # shape (n_features, n_comp)
    load_df = pd.DataFrame(loadings, index=mech_vars, columns=scores_cols).reset_index()
    load_df = load_df.rename(columns={"index": "feature"})

    # Explained variance (%)
    expl = (pca.explained_variance_ratio_ * 100.0).astype(float)
    explained_df = pd.DataFrame({
        "component": scores_cols,
        "explained_var_pct": expl
    })

    return scores_df, load_df, explained_df