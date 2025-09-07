from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn.inspection import permutation_importance

# XGBoost (GPU)
try:
    from xgboost import XGBRegressor
except Exception as e:
    XGBRegressor = None

# stesse liste del backend CPU
MECH_VARS = ['σmax','εf','Fmax','SL','EL']
GEO_VARS  = ['DI','DACL','LES','NSlits']

def _xgb_model(tree_method: str = "gpu_hist"):
    if XGBRegressor is None:
        raise ImportError("xgboost non disponibile. Installa con: pip install xgboost")
    # Nota: se non hai GPU/CUDA, puoi usare tree_method='hist' (CPU) mantenendo la stessa API
    return XGBRegressor(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method=tree_method,
        predictor=("gpu_predictor" if tree_method=="gpu_hist" else "auto"),
        verbosity=0,
    )

def aggregated_cv_r2(estimator, X: pd.DataFrame, y: np.ndarray, splitter):
    preds = np.zeros_like(y, dtype=float)
    for tr, te in splitter.split(X, y):
        m = clone(estimator)
        # XGBRegressor non ha clone perfetto degli attributi gpu; copiamo i params
        m.set_params(**estimator.get_params())
        m.fit(X.iloc[tr], y[tr])
        preds[te] = m.predict(X.iloc[te])
    return r2_score(y, preds)

def geometry_to_mechanics_cv_gpu(dati: pd.DataFrame, use_gpu: bool = True):
    # Controllo colonne
    assert all(v in dati.columns for v in GEO_VARS), f"Mancano variabili geometriche. Colonne: {list(dati.columns)}"
    assert 'Scaffold' in dati.columns, "Manca colonna 'Scaffold' (gruppo)."

    df = dati[GEO_VARS + MECH_VARS + ['Scaffold']].dropna()
    X = df[GEO_VARS].copy()
    groups = df['Scaffold'].values

    # Se GPU non disponibile, ricadiamo su 'hist' (CPU) ma stesso algoritmo
    tree_method = "gpu_hist" if use_gpu else "hist"
    model = _xgb_model(tree_method=tree_method)

    rows = []
    loo = LeaveOneOut()
    logo = LeaveOneGroupOut()

    for target in MECH_VARS:
        if target not in df.columns:
            continue
        y = df[target].values
        if len(y) < 5:
            continue

        r2_loo  = aggregated_cv_r2(model, X, y, loo)
        # LOGO
        preds = np.zeros_like(y, dtype=float)
        for tr, te in logo.split(X, y, groups):
            mm = _xgb_model(tree_method=tree_method)
            mm.fit(X.iloc[tr], y[tr])
            preds[te] = mm.predict(X.iloc[te])
        from sklearn.metrics import r2_score
        r2_logo = r2_score(y, preds)

        rows.append(dict(target=target, model=("XGB-GPU" if use_gpu else "XGB-CPU"),
                         R2_LOOCV=r2_loo, R2_LOGO=r2_logo))
    return pd.DataFrame(rows)

def _bootstrap_indices_by_group(df: pd.DataFrame, group_col: str, rng):
    idx = []
    for _, sub in df.groupby(group_col, observed=False):
        ids = sub.index.to_numpy()
        if len(ids) == 0:
            continue
        take = rng.choice(ids, size=len(ids), replace=True)
        idx.extend(list(take))
    return np.array(idx)

def permutation_importance_ci_gpu(dati: pd.DataFrame, target: str,
                                  n_boot: int = 2000, group_col: str = 'Scaffold',
                                  random_state: int = 42, n_repeats: int = 20,
                                  alpha: float = 0.05, tol: float = 0.02,
                                  min_boot: int = 200, check_every: int = 50,
                                  use_gpu: bool = True):
    if target not in dati.columns:
        raise ValueError(f"Target '{target}' non presente nel dataset.")

    cols = GEO_VARS + [target, group_col]
    df = dati[cols].dropna().copy()
    X_full = df[GEO_VARS]
    y_full = df[target].values

    # modello
    tree_method = "gpu_hist" if use_gpu else "hist"
    base = _xgb_model(tree_method=tree_method)

    rng = np.random.default_rng(random_state)
    imps = []
    last_width = None

    for b in range(1, n_boot + 1):
        boot_idx = _bootstrap_indices_by_group(df, group_col, rng)
        Xb = X_full.iloc[boot_idx]
        yb = y_full[boot_idx]
        m = _xgb_model(tree_method=tree_method)
        m.set_params(**base.get_params())
        m.fit(Xb, yb)
        r = permutation_importance(m, Xb, yb, n_repeats=n_repeats, n_jobs=-1,
                                   random_state=rng.integers(0, 1_000_000_000))
        imps.append(r.importances_mean)

        if b >= min_boot and b % check_every == 0:
            arr = np.vstack(imps)
            lo = np.quantile(arr, alpha/2, axis=0)
            hi = np.quantile(arr, 1 - alpha/2, axis=0)
            width = float(np.mean(hi - lo))
            if last_width is not None:
                rel_change = abs(width - last_width) / last_width if last_width > 0 else 0.0
                if rel_change < tol:
                    break
            last_width = width

    arr = np.vstack(imps)
    return (pd.DataFrame({
        'feature': GEO_VARS,
        'imp_mean': arr.mean(axis=0),
        'imp_lo':   np.quantile(arr, 0.025, axis=0),
        'imp_hi':   np.quantile(arr, 0.975, axis=0),
        'n_boot_actual': len(imps),
        'backend': ("XGB-GPU" if use_gpu else "XGB-CPU"),
    })
    .sort_values('imp_mean', ascending=False)
    .reset_index(drop=True))