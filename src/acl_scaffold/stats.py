from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats.multicomp as mc

# ---------- Assunzioni ANOVA (robusto a stringhe/NaN) ----------
def check_anova_assumptions(df: pd.DataFrame, y: str, group: str = 'Scaffold'):
    work = df[[group, y]].copy()
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work[group] = work[group].astype("category")
    work = work.dropna(subset=[y])

    out = {'shapiro': {}, 'levene_p': None}
    ok_groups = []
    for g, sub in work.groupby(group):
        vals = sub[y].to_numpy(dtype=float)
        if len(vals) >= 3:
            W, p = stats.shapiro(vals)
            out['shapiro'][g] = {'W': float(W), 'p': float(p), 'n': int(len(vals))}
            ok_groups.append(vals)
        else:
            out['shapiro'][g] = {'W': np.nan, 'p': np.nan, 'n': int(len(vals))}
    if len(ok_groups) >= 2 and all(len(v) >= 2 for v in ok_groups):
        _, p_levene = stats.levene(*ok_groups, center='median')
        out['levene_p'] = float(p_levene)
    else:
        out['levene_p'] = np.nan
    return out

# ---------- ANOVA + Tukey (robusto ai tipi) ----------
def oneway_anova_tukey(df: pd.DataFrame, y: str, group: str = 'Scaffold'):
    work = df[[group, y]].copy()
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work[group] = work[group].astype('category')
    work = work.dropna(subset=[y])

    # design matrix numerica (dummy) + costante
    X = pd.get_dummies(work[group], drop_first=True, dtype=float)
    X = sm.add_constant(X, has_constant='add')
    yvec = work[y].to_numpy(dtype=float)

    model = sm.OLS(yvec, X, missing='drop').fit()

    comp = mc.MultiComparison(work[y], work[group])
    tukey = comp.tukeyhsd()
    tukey_df = pd.DataFrame(tukey._results_table.data[1:], columns=tukey._results_table.data[0])

    grand = work[y].mean()
    ss_between = sum(len(g)*(g[y].mean()-grand)**2 for _, g in work.groupby(group))
    ss_total = ((work[y]-grand)**2).sum()
    eta2 = ss_between/ss_total if ss_total > 0 else np.nan
    return {'test':'ANOVA','anova': model, 'tukey': tukey_df, 'eta2': float(eta2)}

# ---------- Kruskal + Dunn (robusto ai tipi) ----------
def kruskal_dunn(df: pd.DataFrame, y: str, group: str = 'Scaffold', p_adjust='holm'):
    work = df[[group, y]].copy()
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work[group] = work[group].astype('category')
    work = work.dropna(subset=[y])

    groups = [g[y].to_numpy(dtype=float) for _, g in work.groupby(group)]
    H, p_kw = stats.kruskal(*groups)

    try:
        import scikit_posthocs as sp
        dunn = sp.posthoc_dunn(work, val_col=y, group_col=group, p_adjust=p_adjust)
    except Exception:
        dunn = None

    n = len(work); k = work[group].nunique()
    eps2 = (H - k + 1) / (n - k) if (n - k) > 0 else np.nan
    return {'test':'Kruskal','kruskal_H': float(H), 'kruskal_p': float(p_kw), 'dunn': dunn, 'epsilon2': float(eps2)}

# ---------- Scelta automatica (usa funzioni robuste) ----------
def anova_or_kruskal(df: pd.DataFrame, y: str, group: str = 'Scaffold', alpha: float = 0.05):
    checks = check_anova_assumptions(df, y, group)
    shapiros = [v for v in checks['shapiro'].values() if not np.isnan(v['p'])]
    normal_ok = all(v['p'] > alpha for v in shapiros) if shapiros else False
    homosked_ok = (checks['levene_p'] is not None and checks['levene_p'] > alpha)
    if normal_ok and homosked_ok:
        res = oneway_anova_tukey(df, y, group)
    else:
        res = kruskal_dunn(df, y, group)
    res['assumptions'] = checks
    return res

# ---------- Effect size pairwise ----------
def cohens_d(x: np.ndarray, y: np.ndarray):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    s = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    return (x.mean() - y.mean()) / s if s > 0 else np.nan

# ---------- Bootstrap CI ----------
def bootstrap_ci(x: np.ndarray, func=np.mean, n: int = 1000, alpha: float = 0.05, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    boots = [func(rng.choice(x, size=len(x), replace=True)) for _ in range(n)]
    lo = np.quantile(boots, alpha/2); hi = np.quantile(boots, 1-alpha/2)
    return float(lo), float(hi), np.asarray(boots)

# ---------- Bootstrap CI con controllo di convergenza ----------
def bootstrap_ci_convergence(x: np.ndarray, func=np.mean, start: int = 500, max_n: int = 20000,
                             alpha: float = 0.05, tol: float = 0.02, random_state: int = 42):
    rng = np.random.default_rng(random_state)
    history = []
    n = start
    prev_width = None
    while n <= max_n:
        boots = [func(rng.choice(x, size=len(x), replace=True)) for _ in range(n)]
        lo = np.quantile(boots, alpha/2); hi = np.quantile(boots, 1-alpha/2)
        width = float(hi - lo)
        rel_change = None if prev_width is None else abs(width - prev_width) / prev_width
        history.append({'n': n, 'lo': float(lo), 'hi': float(hi), 'width': width,
                        'rel_change_vs_prev': (None if rel_change is None else float(rel_change))})
        if prev_width is not None and rel_change is not None and rel_change < tol:
            break
        prev_width = width
        n *= 2
    final = history[-1]
    return {'history': pd.DataFrame(history), 'final_ci': (final['lo'], final['hi']), 'alpha': alpha, 'tol': tol}