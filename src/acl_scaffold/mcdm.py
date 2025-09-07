from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

# Default: tutti i criteri "benefit" (maggiore è meglio). Puoi personalizzare.
DEFAULT_BENEFIT = True

def _orient(df: pd.DataFrame, benefit_mask: dict[str, bool]):
    X = df.copy()
    for c, is_benefit in benefit_mask.items():
        if c not in X.columns: continue
        X[c] = pd.to_numeric(X[c], errors="coerce")
        if not is_benefit:
            X[c] = -X[c]   # trasformo criterio "costo" in "benefit"
    return X

def group_means(dati: pd.DataFrame, group_col: str, criteria: list[str]):
    g = (dati[[group_col] + criteria]
         .copy())
    for c in criteria:
        g[c] = pd.to_numeric(g[c], errors="coerce")
    g = g.dropna()
    return g.groupby(group_col, observed=False)[criteria].mean().reset_index()

# ---------- PROMETHEE II (linear preference) ----------
def promethee_ii(df_alt: pd.DataFrame, criteria: list[str],
                 weights: dict[str, float] | None = None,
                 benefit_mask: dict[str, bool] | None = None,
                 s_scale: dict[str, float] | None = None):
    """
    df_alt: righe=alternative (Scaffold), colonne=criteri numerici
    s_scale: soglia di scala per funzione di preferenza lineare (default = std del criterio)
    """
    if weights is None:
        weights = {c: 1.0 for c in criteria}
    if benefit_mask is None:
        benefit_mask = {c: DEFAULT_BENEFIT for c in criteria}
    X = _orient(df_alt[criteria], benefit_mask).to_numpy(dtype=float)
    names = df_alt.iloc[:, 0].astype(str).tolist() if df_alt.columns[0] not in criteria else [f"A{i}" for i in range(len(df_alt))]
    m, n = X.shape
    w = np.array([weights[c] for c in criteria], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w)/len(w)

    if s_scale is None:
        s_scale = {}
    s = np.array([s_scale.get(c, np.nan) for c in criteria], dtype=float)
    # fallback: std
    for j in range(n):
        if not np.isfinite(s[j]) or s[j] <= 0:
            col = X[:, j]
            s[j] = np.nanstd(col) if np.nanstd(col) > 0 else (np.nanmax(col) - np.nanmin(col) + 1e-12)

    # preferenze pairwise
    P = np.zeros((m, m))
    for i in range(m):
        for k in range(m):
            if i == k: continue
            d = X[i, :] - X[k, :]
            # preferenza lineare per criterio
            pj = np.clip(d / s, 0, 1)
            P[i, k] = float(np.sum(w * pj))
    phi_plus  = P.mean(axis=1)
    phi_minus = P.mean(axis=0)
    phi = phi_plus - phi_minus
    out = pd.DataFrame({"alternative": names, "phi_plus": phi_plus, "phi_minus": phi_minus, "phi": phi})
    out = out.sort_values("phi", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out)+1)
    return out

# ---------- TOPSIS (crisp) ----------
def topsis(df_alt: pd.DataFrame, criteria: list[str],
           weights: dict[str, float] | None = None,
           benefit_mask: dict[str, bool] | None = None):
    if weights is None:
        weights = {c: 1.0 for c in criteria}
    if benefit_mask is None:
        benefit_mask = {c: DEFAULT_BENEFIT for c in criteria}

    X = df_alt[criteria].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    names = df_alt.iloc[:, 0].astype(str).tolist()
    m, n = X.shape
    w = np.array([weights[c] for c in criteria], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w)/len(w)

    # normalizzazione vettoriale
    norm = np.linalg.norm(X, axis=0)
    norm[norm == 0] = 1.0
    R = X / norm
    V = R * w

    # orientazione
    benefit = np.array([benefit_mask[c] for c in criteria], dtype=bool)
    ideal_best  = np.where(benefit, V.max(axis=0), V.min(axis=0))
    ideal_worst = np.where(benefit, V.min(axis=0), V.max(axis=0))

    Dp = np.sqrt(((V - ideal_best )**2).sum(axis=1))
    Dm = np.sqrt(((V - ideal_worst)**2).sum(axis=1))
    C  = Dm / (Dp + Dm + 1e-12)
    out = pd.DataFrame({"alternative": names, "closeness": C})
    out = out.sort_values("closeness", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out)+1)
    return out

# ---------- ELECTRE III (versione semplice) ----------
def electre_iii(df_alt: pd.DataFrame, criteria: list[str],
                weights: dict[str, float] | None = None,
                benefit_mask: dict[str, bool] | None = None):
    """
    Implementazione leggera: concordanza lineare e discordanza a soglia (veto) derivata dal range.
    """
    if weights is None:
        weights = {c: 1.0 for c in criteria}
    if benefit_mask is None:
        benefit_mask = {c: DEFAULT_BENEFIT for c in criteria}

    X = _orient(df_alt[criteria], benefit_mask).to_numpy(dtype=float)
    names = df_alt.iloc[:, 0].astype(str).tolist()
    m, n = X.shape
    w = np.array([weights[c] for c in criteria], dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w)/len(w)

    rng = np.nanmax(X, axis=0) - np.nanmin(X, axis=0) + 1e-12
    q = 0.10 * rng   # indifference
    p = 0.30 * rng   # preference
    v = 0.80 * rng   # veto

    # concordanza c_ij e discordanza d_ij per ogni coppia
    C = np.zeros((m, m))
    D = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j: continue
            c_list = []
            veto = False
            for k in range(n):
                diff = X[i, k] - X[j, k]
                if diff <= q[k]:
                    c_k = 0.0
                elif diff >= p[k]:
                    c_k = 1.0
                else:
                    c_k = (diff - q[k]) / (p[k] - q[k])
                    c_k = np.clip(c_k, 0.0, 1.0)
                c_list.append(w[k] * c_k)
                # discordanza/veto
                if diff < -v[k]:  # j supera i in modo inaccettabile su qualche criterio
                    veto = True
            C[i, j] = np.sum(c_list)
            D[i, j] = 1.0 if veto else 0.0

    # credibilità (semplificata): sigma = C * (1 - max discordanza)
    Sigma = C * (1 - D)
    phi_plus = Sigma.mean(axis=1)
    phi_minus = Sigma.mean(axis=0)
    phi = phi_plus - phi_minus
    out = pd.DataFrame({"alternative": names, "credibility_plus": phi_plus, "credibility_minus": phi_minus, "net": phi})
    out = out.sort_values("net", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out)+1)
    return out

# ---------- Borda aggregation ----------
def borda_rank(*rank_tables: pd.DataFrame, name_col: str = "alternative", rank_col: str = "rank"):
    """
    Aggrega rankings (1=best) assegnando punteggi Borda e sommandoli.
    """
    # normalizza input
    Rs = []
    for T in rank_tables:
        t = T[[name_col, rank_col]].copy()
        n = len(t)
        t["borda_points"] = (n - t[rank_col]).astype(float)  # rank 1 -> n-1 punti
        Rs.append(t)
    base = Rs[0][[name_col, "borda_points"]].rename(columns={"borda_points": "score_1"})
    for i, T in enumerate(Rs[1:], start=2):
        base = base.merge(T[[name_col, "borda_points"]].rename(columns={"borda_points": f"score_{i}"}),
                          on=name_col, how="inner")
    base["borda_total"] = base.filter(like="score_").sum(axis=1)
    base = base.sort_values("borda_total", ascending=False).reset_index(drop=True)
    base["rank"] = np.arange(1, len(base)+1)
    return base
