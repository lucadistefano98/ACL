from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from .stats import anova_or_kruskal

# Target del paper (εTmax esclusa se non serve)
MECH_VARS = ['σmax','εf','Fmax','SL','EL']
GEO_VARS  = ['DI','DACL','LES','NSlits']

def make_models():
    rf = Pipeline([('scaler', StandardScaler()),
                   ('rf', RandomForestRegressor(n_estimators=500, random_state=42))])
    lr = Pipeline([('scaler', StandardScaler()),
                   ('lr', LinearRegression())])
    return {'RF': rf, 'LR': lr}

def aggregated_cv_r2(model, X: pd.DataFrame, y: np.ndarray, splitter):
    preds = np.zeros_like(y, dtype=float)
    for tr, te in splitter.split(X, y):
        m = clone(model).fit(X.iloc[tr], y[tr])
        preds[te] = m.predict(X.iloc[te])
    return r2_score(y, preds)

def geometry_to_mechanics_cv(dati: pd.DataFrame):
    assert all(v in dati.columns for v in GEO_VARS), f"Mancano variabili geometriche. Colonne: {list(dati.columns)}"
    assert 'Scaffold' in dati.columns, "Manca colonna 'Scaffold' (gruppo)."

    df = dati[GEO_VARS + MECH_VARS + ['Scaffold']].dropna()
    X = df[GEO_VARS].copy()
    groups = df['Scaffold'].values
    models = make_models()
    rows = []

    for target in MECH_VARS:
        if target not in df.columns:
            continue
        y = df[target].values
        if len(y) < 5:
            continue

        loo = LeaveOneOut()
        logo = LeaveOneGroupOut()

        for mname, m in models.items():
            r2_loo = aggregated_cv_r2(m, X, y, loo)

            preds = np.zeros_like(y, dtype=float)
            for tr, te in logo.split(X, y, groups):
                mm = clone(m).fit(X.iloc[tr], y[tr])
                preds[te] = mm.predict(X.iloc[te])
            r2_logo = r2_score(y, preds)

            # Test omnibus + effect size
            omnibus = anova_or_kruskal(df[['Scaffold', target]].dropna(), target, group='Scaffold', alpha=0.05)
            if omnibus['test'] == 'ANOVA':
                pval = float(getattr(omnibus['anova'], 'f_pvalue', np.nan))
                eff  = float(omnibus.get('eta2', np.nan))
            else:
                pval = float(omnibus.get('kruskal_p', np.nan))
                eff  = float(omnibus.get('epsilon2', np.nan))

            rows.append(dict(target=target, model=mname,
                             R2_LOOCV=r2_loo, R2_LOGO=r2_logo,
                             omnibus=omnibus['test'], omnibus_p=pval, effect_size=eff))

    return pd.DataFrame(rows)

# ----- Permutation Importance con CI bootstrap -----
def _bootstrap_indices_by_group(df: pd.DataFrame, group_col: str, rng):
    idx = []
    for _, sub in df.groupby(group_col):
        ids = sub.index.to_numpy()
        if len(ids) == 0:
            continue
        take = rng.choice(ids, size=len(ids), replace=True)
        idx.extend(list(take))
    return np.array(idx)

def permutation_importance_ci(dati: pd.DataFrame, target: str, model_name: str = 'RF',
                              n_boot: int = 2000, group_col: str = 'Scaffold',
                              random_state: int = 42, n_repeats: int = 50):
    if target not in dati.columns:
        raise ValueError(f"Target '{target}' non presente nel dataset.")
    models = make_models()
    if model_name not in models:
        raise ValueError(f"Modello '{model_name}' non valido. Usa: {list(models)}")
    model = models[model_name]

    cols = GEO_VARS + [target, group_col]
    df = dati[cols].dropna().copy()
    X_full = df[GEO_VARS]
    y_full = df[target].values

    rng = np.random.default_rng(random_state)
    imps = []

    for _ in range(n_boot):
        boot_idx = _bootstrap_indices_by_group(df, group_col, rng)
        Xb = X_full.iloc[boot_idx]
        yb = y_full[boot_idx]
        m = clone(model).fit(Xb, yb)
        r = permutation_importance(m, Xb, yb, n_repeats=n_repeats,
                                   random_state=rng.integers(0, 1_000_000_000))
        imps.append(r.importances_mean)

    imps = np.vstack(imps)
    return (pd.DataFrame({
        'feature': GEO_VARS,
        'imp_mean': imps.mean(axis=0),
        'imp_lo':   np.quantile(imps, 0.025, axis=0),
        'imp_hi':   np.quantile(imps, 0.975, axis=0),
    })
    .sort_values('imp_mean', ascending=False)
    .reset_index(drop=True))
