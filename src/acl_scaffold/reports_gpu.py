from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np

from .io import load_matrix
from .models import geometry_to_mechanics_cv  # CV e stats restano CPU come nel paper
from .stats import bootstrap_ci_convergence, anova_or_kruskal
from .models_gpu import permutation_importance_ci_gpu, MECH_VARS

def run_cv_report(xlsx_path: str | Path, out_csv: str | Path):
    dati, _ = load_matrix(xlsx_path)
    res = geometry_to_mechanics_cv(dati)
    out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    return res

def run_permimp_report_gpu(xlsx_path: str | Path, target: str, out_csv: str | Path,
                           n_boot: int = 2000, n_repeats: int = 20,
                           alpha: float = 0.05, tol: float = 0.02,
                           min_boot: int = 200, check_every: int = 50):
    dati, _ = load_matrix(xlsx_path)
    res = permutation_importance_ci_gpu(dati, target=target, n_boot=n_boot,
                                        n_repeats=n_repeats, alpha=alpha, tol=tol,
                                        min_boot=min_boot, check_every=check_every,
                                        random_state=42, n_jobs=-1)
    out = Path(out_csv); out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    return res

def run_stats_tables(xlsx_path: str | Path, outdir: str | Path,
                     mech_vars: list[str] | None = None,
                     group_col: str = "Scaffold") -> pd.DataFrame:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    dati, _ = load_matrix(xlsx_path)
    if mech_vars is None:
        mech_vars = ['σmax','εf','Fmax','SL','EL']

    rows = []
    for y in mech_vars:
        if y not in dati.columns: continue
        df = dati[[group_col, y]].dropna()
        if df.empty or df[group_col].nunique() < 2: continue

        res = anova_or_kruskal(df, y, group=group_col, alpha=0.05)
        if res['test'] == 'ANOVA':
            pval = float(getattr(res['anova'], 'f_pvalue', float('nan')))
            eff  = float(res.get('eta2', float('nan')))
            tuk = res.get('tukey')
            if tuk is not None:
                tuk.to_csv(outdir / f"tukey_{y}.csv", index=False)
        else:
            pval = float(res.get('kruskal_p', float('nan')))
            eff  = float(res.get('epsilon2', float('nan')))
            dunn = res.get('dunn')
            if dunn is not None:
                dunn.to_csv(outdir / f"dunn_{y}.csv", index=True)

        rows.append({"variable": y, "test": res['test'], "p_value": pval, "effect_size": eff})

    overview = pd.DataFrame(rows).sort_values("variable")
    overview.to_csv(outdir / "stats_overview.csv", index=False)
    return overview

def run_all_gpu(xlsx_path: str | Path, outdir: str | Path,
                permimp_all_targets: bool = False):
    """
    Mantiene la stessa pipeline del CPU run_all, ma calcola la Permutation Importance con XGBoost GPU.
    - Se permimp_all_targets=True: esegue anche σmax, εf, SL, EL (oltre a Fmax).
    - Altrimenti, solo Fmax (come nel paper).
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # 1) CV (come prima, CPU)
    run_cv_report(xlsx_path, outdir / "cv_results.csv")

    # 2) Permutation Importance (GPU)
    if permimp_all_targets:
        dati, _ = load_matrix(xlsx_path)
        targets = [v for v in MECH_VARS if v in dati.columns]
        for t in targets:
            run_permimp_report_gpu(xlsx_path, target=t, out_csv=outdir / f"permimp_{t}.csv",
                                   n_boot=2000, n_repeats=20, alpha=0.05, tol=0.02,
                                   min_boot=200, check_every=50)
    else:
        run_permimp_report_gpu(xlsx_path, target="Fmax", out_csv=outdir / "permimp_Fmax.csv",
                               n_boot=2000, n_repeats=20, alpha=0.05, tol=0.02,
                               min_boot=200, check_every=50)

    # 3) Bootstrap convergence (σmax) come prima
    dati, _ = load_matrix(xlsx_path)
    if "σmax" in dati.columns:
        series = dati["σmax"].dropna().to_numpy()
        if series.size >= 2:
            hist = bootstrap_ci_convergence(series, func=np.mean,
                                            start=500, max_n=20000,
                                            alpha=0.05, tol=0.02,
                                            random_state=42)["history"]
            hist.to_csv(outdir / "bootstrap_convergence_sigma_max.csv", index=False)

    # 4) Tabelle statistiche (come prima)
    run_stats_tables(xlsx_path, outdir)
