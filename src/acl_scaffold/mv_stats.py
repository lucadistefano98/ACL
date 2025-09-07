from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

def _as_matrix_by_group(df: pd.DataFrame, features: list[str], group_col: str):
    X_by_g = {}
    for g, sub in df.groupby(group_col, observed=False):
        M = sub[features].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
        if M.shape[0] >= 2:
            X_by_g[g] = M
    return X_by_g

def hotelling_t2_two_groups(df: pd.DataFrame, features: list[str], group_col: str = "Scaffold"):
    """
    Hotelling's T² per 2 gruppi. Ritorna dict con T2, F, df1, df2, pval.
    """
    X_by_g = _as_matrix_by_group(df[[group_col] + features], features, group_col)
    if len(X_by_g) != 2:
        raise ValueError("Hotelling T² richiede esattamente 2 gruppi con almeno 2 osservazioni ciascuno.")
    (g1, X1), (g2, X2) = list(X_by_g.items())
    n1, n2 = X1.shape[0], X2.shape[0]
    p = X1.shape[1]
    mean1, mean2 = X1.mean(axis=0), X2.mean(axis=0)
    S1 = np.cov(X1, rowvar=False)
    S2 = np.cov(X2, rowvar=False)
    Sp = ((n1 - 1)*S1 + (n2 - 1)*S2) / (n1 + n2 - 2)
    diff = mean1 - mean2
    T2 = (n1 * n2) / (n1 + n2) * diff.T @ np.linalg.pinv(Sp) @ diff
    # conversione a F
    F = ( (n1 + n2 - p - 1) / (p * (n1 + n2 - 2)) ) * T2
    df1 = p
    df2 = n1 + n2 - p - 1
    pval = 1 - stats.f.cdf(F, df1, df2)
    return {
        "groups": [g1, g2],
        "p": p, "n1": n1, "n2": n2,
        "T2": float(T2), "F": float(F), "df1": int(df1), "df2": int(df2),
        "p_value": float(pval)
    }

def manova_oneway_wilks(df: pd.DataFrame, features: list[str], group_col: str = "Scaffold"):
    """
    MANOVA 1-way: calcolo di Wilks' Lambda con approssimazione F di Rao.
    Restituisce: {'wilks':..., 'F':..., 'df1':..., 'df2':..., 'p_value':...}
    """
    work = df[[group_col] + features].copy()
    for c in features:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna()
    groups = [g for g, _ in work.groupby(group_col, observed=False)]
    g = len(groups)
    if g < 2:
        raise ValueError("MANOVA richiede almeno 2 gruppi.")
    X = work[features].to_numpy(dtype=float)
    N, p = X.shape
    # grand mean
    mu = X.mean(axis=0)
    # E (within), H (between)
    E = np.zeros((p, p))
    H = np.zeros((p, p))
    for _, sub in work.groupby(group_col, observed=False):
        Xi = sub[features].to_numpy(dtype=float)
        ni = Xi.shape[0]
        mui = Xi.mean(axis=0)
        E += (Xi - mui).T @ (Xi - mui)
        H += ni * np.outer(mui - mu, mui - mu)
    # Wilks lambda
    detE = np.linalg.det(E) if np.linalg.det(E) != 0 else np.linalg.det(E + 1e-12*np.eye(p))
    detEH = np.linalg.det(E + H)
    lam = detE / detEH
    # Rao's approximation to F
    m = (abs(p - g + 1) - 1) / 2
    n = (N - p - g) / 2
    # protezione
    lam = float(np.clip(lam, 1e-12, 1.0))
    F = ((1 - lam**(1/(1+m))) / (lam**(1/(1+m)))) * (n / p)
    df1 = p * (g - 1)
    df2 = int(2 * n * (1 + m))
    pval = 1 - stats.f.cdf(F, df1, df2)
    return {"wilks": float(lam), "F": float(F), "df1": int(df1), "df2": int(df2), "p_value": float(pval),
            "p": int(p), "groups": int(g), "N": int(N)}
