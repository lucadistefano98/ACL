from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

from .io import load_matrix
from .models import geometry_to_mechanics_cv, permutation_importance_ci
from .stats import bootstrap_ci_convergence
from .pca import run_pca, MECH_VARS_ALL
from .mv_stats import manova_oneway_wilks, hotelling_t2_two_groups
from .cluster import run_clustering_and_save
from .mcdm import group_means, promethee_ii, topsis, electre_iii, borda_rank

# GPU backend (XGBoost)
def _gpu_available() -> bool:
    try:
        import xgboost  # noqa
        return True
    except Exception:
        return False

# ---------- CV ----------
def run_cv_report_cpu(dati: pd.DataFrame, out_csv: Path):
    res = geometry_to_mechanics_cv(dati)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    return res

def run_cv_report_gpu(dati: pd.DataFrame, out_csv: Path, use_gpu: bool = True):
    from .models_gpu import geometry_to_mechanics_cv_gpu
    res = geometry_to_mechanics_cv_gpu(dati, use_gpu=use_gpu)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    return res

# ---------- Permutation importance ----------
def run_permimp_cpu(dati: pd.DataFrame, target: str, out_csv: Path, n_boot: int = 2000):
    res = permutation_importance_ci(dati, target=target, model_name='RF', n_boot=n_boot)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    return res

def run_permimp_gpu(dati: pd.DataFrame, target: str, out_csv: Path, n_boot: int = 2000, use_gpu: bool = True):
    from .models_gpu import permutation_importance_ci_gpu
    res = permutation_importance_ci_gpu(dati, target=target, n_boot=n_boot, use_gpu=use_gpu)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_csv, index=False)
    return res

# ---------- Stats overview (ANOVA/Kruskal) ----------
def run_stats_tables(xlsx_path: str | Path, outdir: str | Path,
                     mech_vars: list[str] | None = None,
                     group_col: str = "Scaffold") -> pd.DataFrame:
    from .stats import anova_or_kruskal
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    dati, _ = load_matrix(xlsx_path)
    if mech_vars is None:
        mech_vars = ['σmax','εf','Fmax','SL','EL']

    rows = []
    for y in mech_vars:
        if y not in dati.columns:
            continue
        df = dati[[group_col, y]].dropna()
        if df.empty or df[group_col].nunique() < 2:
            continue
        res = anova_or_kruskal(df, y, group=group_col, alpha=0.05)
        if res['test'] == 'ANOVA':
            pval = float(getattr(res['anova'], 'f_pvalue', float('nan')))
            eff  = float(res.get('eta2', float('nan')))
            tuk  = res.get('tukey')
            if tuk is not None:
                (outdir / f"tukey_{y}.csv").write_text(tuk.to_csv(index=False), encoding="utf-8")
        else:
            pval = float(res.get('kruskal_p', float('nan')))
            eff  = float(res.get('epsilon2', float('nan')))
            dunn = res.get('dunn')
            if dunn is not None:
                dunn.to_csv(outdir / f"dunn_{y}.csv", index=True)
        rows.append({"variable": y, "test": res['test'], "p_value": pval, "effect_size": eff})

    overview = pd.DataFrame(rows).sort_values("variable")
    (outdir / "stats_overview.csv").write_text(overview.to_csv(index=False), encoding="utf-8")
    return overview

# ---------- PCA ----------
def run_pca_block(dati: pd.DataFrame, outdir: Path, group_col="Scaffold"):
    try:
        scores_df, load_df, explained_df = run_pca(dati, mech_vars=None, group_col=group_col, n_components=5)
        scores_df.to_csv(outdir / "pca_scores.csv", index=False)
        load_df.to_csv(outdir / "pca_loadings.csv", index=False)
        explained_df.to_csv(outdir / "pca_explained.csv", index=False)
    except Exception as e:
        print("[warn] PCA non eseguita:", e)

# ---------- MANOVA / Hotelling ----------
def run_multivariate_tests(dati: pd.DataFrame, outdir: Path, group_col="Scaffold"):
    used = [c for c in MECH_VARS_ALL if c in dati.columns]
    if len(used) < 2 or dati[group_col].nunique() < 2:
        return
    sub = dati[[group_col] + used].dropna()
    outdir.mkdir(parents=True, exist_ok=True)
    # MANOVA (k>=2)
    try:
        man = manova_oneway_wilks(sub, used, group_col=group_col)
        pd.DataFrame([man]).to_csv(outdir / "manova_wilks.csv", index=False)
    except Exception as e:
        print("[warn] MANOVA fallita:", e)
    # Hotelling T² per confronti a coppie (se utile)
    try:
        pairs = []
        groups = sub[group_col].unique().tolist()
        if len(groups) == 2:
            ht = hotelling_t2_two_groups(sub, used, group_col=group_col)
            pd.DataFrame([ht]).to_csv(outdir / "hotelling_T2.csv", index=False)
        elif len(groups) > 2:
            # opzionale: solo prime due per avere un riferimento
            g1, g2 = groups[:2]
            sub2 = sub[sub[group_col].isin([g1, g2])]
            ht = hotelling_t2_two_groups(sub2, used, group_col=group_col)
            pd.DataFrame([ht]).to_csv(outdir / f"hotelling_T2_{g1}_vs_{g2}.csv", index=False)
    except Exception as e:
        print("[warn] Hotelling T² non calcolato:", e)

# ---------- Clustering ----------
def run_clustering_block(dati: pd.DataFrame, outdir: Path, group_col="Scaffold"):
    try:
        run_clustering_and_save(dati, outdir=outdir, group_col=group_col, mech_vars=None)
    except Exception as e:
        print("[warn] Clustering non eseguito:", e)

# ---------- MCDM ----------
def run_mcdm_block(dati: pd.DataFrame, outdir: Path, group_col="Scaffold"):
    crit = [c for c in ['σmax','εf','Fmax','SL','EL','ST','ET','T','εTmax'] if c in dati.columns]
    if len(crit) < 2:
        return
    gm = group_means(dati, group_col=group_col, criteria=crit)
    # Tutti benefit per default (puoi personalizzare da paper)
    weights = {c: 1.0 for c in crit}
    benefit = {c: True for c in crit}

    try:
        prom = promethee_ii(gm, [*crit], weights=weights, benefit_mask=benefit)
        prom.to_csv(outdir / "mcdm_promethee.csv", index=False)
    except Exception as e:
        print("[warn] PROMETHEE II non eseguito:", e)

    try:
        top = topsis(gm, [*crit], weights=weights, benefit_mask=benefit)
        top.to_csv(outdir / "mcdm_topsis.csv", index=False)
    except Exception as e:
        print("[warn] TOPSIS non eseguito:", e)

    try:
        ele = electre_iii(gm, [*crit], weights=weights, benefit_mask=benefit)
        ele.to_csv(outdir / "mcdm_electre.csv", index=False)
    except Exception as e:
        print("[warn] ELECTRE III non eseguito:", e)

    # Borda
    try:
        # uso i rank delle tre tabelle (se esistono)
        tabs = []
        for p in [outdir / "mcdm_promethee.csv", outdir / "mcdm_topsis.csv", outdir / "mcdm_electre.csv"]:
            if p.exists():
                t = pd.read_csv(p)
                if "rank" in t.columns and "alternative" in t.columns:
                    tabs.append(t)
        if tabs:
            borda = borda_rank(*tabs)
            borda.to_csv(outdir / "mcdm_borda.csv", index=False)
    except Exception as e:
        print("[warn] Borda non eseguito:", e)

# ---------- RUN ALL ----------
def run_all(xlsx_path: str | Path, outdir: str | Path, backend: str = "cpu", permimp_all: bool = False):
    """
    Esegue:
      1) CV (LOOCV/LOGO) con backend: CPU=RF scikit-learn, GPU=XGBoost
      2) Permutation Importance (Fmax o tutte)
      3) Bootstrap convergence su σmax
      4) Tabelle statistiche (ANOVA/Kruskal)
      5) PCA (scores, loadings, explained)
      6) MANOVA / Hotelling T²
      7) Clustering (KMeans + Ward) con metriche e linkage
      8) MCDM (PROMETHEE II, TOPSIS, ELECTRE III + Borda)
    """
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    dati, _ = load_matrix(xlsx_path)

    # 1) CV
    if backend == "gpu":
        use_gpu = _gpu_available()
        if not use_gpu:
            print("[warn] xgboost non disponibile o GPU non attiva: ricado su XGB-CPU.")
        print("[step] 1/8 CV (GPU)" if use_gpu else "[step] 1/8 CV (XGB-CPU)")
        run_cv_report_gpu(dati, outdir / "cv_results.csv", use_gpu=use_gpu)
    else:
        print("[step] 1/8 CV (CPU RF)")
        run_cv_report_cpu(dati, outdir / "cv_results.csv")

    # 2) Permutation Importance
    mech_vars = ["σmax", "εf", "Fmax", "SL", "EL"] if permimp_all else ["Fmax"]
    print(f"[step] 2/8 Permutation Importance -> {', '.join([m for m in mech_vars if m in dati.columns])}")
    for target in mech_vars:
        if target not in dati.columns:
            continue
        out_csv = outdir / f"permimp_{target}.csv"
        if backend == "gpu":
            use_gpu = _gpu_available()
            run_permimp_gpu(dati, target, out_csv, n_boot=2000, use_gpu=use_gpu)
        else:
            run_permimp_cpu(dati, target, out_csv, n_boot=2000)

    # 3) Bootstrap convergence (σmax)
    print("[step] 3/8 Bootstrap convergence (σmax)")
    if "σmax" in dati.columns:
        series = dati["σmax"].dropna().to_numpy()
        if series.size >= 2:
            hist = bootstrap_ci_convergence(
                series, func=np.mean, start=500, max_n=20000,
                alpha=0.05, tol=0.02, random_state=42
            )["history"]
            hist.to_csv(outdir / "bootstrap_convergence_sigma_max.csv", index=False)

    # 4) Tabelle statistiche
    print("[step] 4/8 Stats tables (ANOVA/Kruskal)")
    run_stats_tables(xlsx_path, outdir)

    # 5) PCA
    print("[step] 5/8 PCA (mechanical features)")
    run_pca_block(dati, outdir)

    # 6) MANOVA / Hotelling
    print("[step] 6/8 Multivariate tests (MANOVA / T²)")
    run_multivariate_tests(dati, outdir)

    # 7) Clustering
    print("[step] 7/8 Clustering (KMeans + Ward)")
    run_clustering_block(dati, outdir)

    # 8) MCDM
    print("[step] 8/8 MCDM (PROMETHEE II / TOPSIS / ELECTRE III + Borda)")
    run_mcdm_block(dati, outdir)
