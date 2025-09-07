#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import re
from scipy.cluster.hierarchy import dendrogram

def _ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save(fig: plt.Figure, out: Path, tight: bool = True, dpi: int = 300):
    if tight:
        fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def _read_csv_safe(p: Path) -> pd.DataFrame | None:
    if not p.exists():
        print(f"[warn] manca {p}", file=sys.stderr)
        return None
    try:
        return pd.read_csv(p)
    except Exception as e:
        print(f"[warn] errore nel leggere {p}: {e}", file=sys.stderr)
        return None

def _latex_float(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)

# ---------- CV ----------
def fig_cv_results(cv_csv: Path, outdir: Path):
    df = _read_csv_safe(cv_csv)
    if df is None or df.empty:
        return
    needed = {"target", "model", "R2_LOOCV", "R2_LOGO"}
    if not needed.issubset(df.columns):
        print(f"[warn] cv_results.csv non ha colonne attese {needed}", file=sys.stderr)
        return
    order_targets = ["σmax","εf","Fmax","SL","EL"]
    df["target"] = pd.Categorical(df["target"], categories=order_targets, ordered=True)
    df = df.sort_values(["target","model"])

    piv_loo = df.pivot(index="target", columns="model", values="R2_LOOCV")
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    piv_loo.plot(kind="bar", ax=ax1)
    ax1.set_ylabel(r"$R^2$ (LOOCV)")
    ax1.set_xlabel("Target")
    ax1.set_title("Predictive performance — LOOCV")
    ax1.legend(title="Model")
    _save(fig1, outdir / "cv_r2_loocv")

    piv_logo = df.pivot(index="target", columns="model", values="R2_LOGO")
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    piv_logo.plot(kind="bar", ax=ax2)
    ax2.set_ylabel(r"$R^2$ (LOGO)")
    ax2.set_xlabel("Target")
    ax2.set_title("Predictive performance — LOGO")
    ax2.legend(title="Model")
    _save(fig2, outdir / "cv_r2_logo")

# ---------- PermImp ----------
def fig_permimp_all(outdir_reports: Path, outdir_fig: Path):
    files = sorted(outdir_reports.glob("permimp_*.csv"))
    if not files:
        print("[warn] nessun permimp_*.csv trovato", file=sys.stderr)
        return
    for f in files:
        df = _read_csv_safe(f)
        if df is None or df.empty:
            continue
        needed = {"feature","imp_mean","imp_lo","imp_hi"}
        if not needed.issubset(df.columns):
            print(f"[warn] {f.name} non contiene colonne {needed}", file=sys.stderr)
            continue
        target = re.sub(r"^permimp_(.*)\.csv$", r"\1", f.name)
        order_feats = ["DI","DACL","LES","NSlits"]
        if "feature" in df.columns:
            df["feature"] = pd.Categorical(df["feature"], categories=order_feats, ordered=True)
        df = df.sort_values("feature")
        y = df["imp_mean"].values
        ylo = df["imp_lo"].values
        yhi = df["imp_hi"].values
        x = np.arange(len(df))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(x, y)
        err = np.vstack([y - ylo, yhi - y])
        ax.errorbar(x, y, yerr=err, fmt="none", capsize=4, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(df["feature"].astype(str), rotation=0)
        ax.set_ylabel("Permutation importance (Δscore)")
        title = f"Permutation importance — {target}"
        if "backend" in df.columns and df["backend"].nunique() == 1:
            title += f" [{df['backend'].iloc[0]}]"
        ax.set_title(title)
        _save(fig, outdir_fig / f"permimp_{target}")

# ---------- Bootstrap ----------
def fig_bootstrap_convergence(csv_path: Path, outdir_fig: Path):
    df = _read_csv_safe(csv_path)
    if df is None or df.empty:
        return
    need = {"n","lo","hi","width"}
    if not need.issubset(df.columns):
        print(f"[warn] bootstrap CSV non ha colonne {need}", file=sys.stderr)
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df["n"], df["width"], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Bootstrap samples (n)")
    ax.set_ylabel("CI width (95%)")
    ax.set_title("Bootstrap CI convergence — σmax (mean)")
    _save(fig, outdir_fig / "bootstrap_convergence_sigma_max")

# ---------- Stats overview ----------
def fig_stats_overview(csv_path: Path, outdir_fig: Path):
    df = _read_csv_safe(csv_path)
    if df is None or df.empty:
        return
    need = {"variable","test","p_value","effect_size"}
    if not need.issubset(df.columns):
        print(f"[warn] stats_overview.csv non ha colonne {need}", file=sys.stderr)
        return
    order_targets = ["σmax","εf","Fmax","SL","EL"]
    df["variable"] = pd.Categorical(df["variable"], categories=order_targets, ordered=True)
    df = df.sort_values("variable")
    show = df.copy()
    show["p_value"] = show["p_value"].map(_latex_float)
    show["effect_size"] = show["effect_size"].map(_latex_float)
    fig, ax = plt.subplots(figsize=(6.5, 0.6 + 0.45*len(show)))
    ax.axis("off")
    tbl = ax.table(cellText=show.values,
                   colLabels=show.columns,
                   loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.2)
    ax.set_title("Omnibus tests overview", pad=12)
    _save(fig, outdir_fig / "stats_overview_table")

# ---------- PCA biplot ----------
def fig_pca_biplot(reports_dir: Path, outdir_fig: Path):
    scores = _read_csv_safe(reports_dir / "pca_scores.csv")
    loads  = _read_csv_safe(reports_dir / "pca_loadings.csv")
    expl   = _read_csv_safe(reports_dir / "pca_explained.csv")
    if any(x is None or x.empty for x in (scores, loads, expl)):
        return
    if not {"PC1","PC2"}.issubset(scores.columns) or not {"PC1","PC2"}.issubset(loads.columns):
        print("[warn] PCA: mancano PC1/PC2", file=sys.stderr)
        return
    group_col = "Scaffold" if "Scaffold" in scores.columns else scores.columns[0]
    groups = sorted(scores[group_col].unique().tolist())
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for g in groups:
        sub = scores[scores[group_col] == g]
        ax.scatter(sub["PC1"], sub["PC2"], label=str(g))
    for _, row in loads.iterrows():
        x, y = row["PC1"], row["PC2"]
        ax.arrow(0, 0, x, y, head_width=0.03, length_includes_head=True)
        ax.text(x*1.08, y*1.08, str(row["feature"]), fontsize=9)
    try:
        v1 = float(expl.loc[expl["component"]=="PC1","explained_var_pct"].values[0])
        v2 = float(expl.loc[expl["component"]=="PC2","explained_var_pct"].values[0])
    except Exception:
        v1 = v2 = None
    ax.set_xlabel(f"PC1 ({v1:.1f}%)" if v1 is not None else "PC1")
    ax.set_ylabel(f"PC2 ({v2:.1f}%)" if v2 is not None else "PC2")
    ax.set_title("PCA biplot — mechanical features (Z-score)")
    ax.legend(title=group_col)
    ax.axhline(0, color="gray", linewidth=0.5); ax.axvline(0, color="gray", linewidth=0.5)
    _save(fig, outdir_fig / "pca_biplot")

# ---------- Clustering metrics & dendrogram ----------
def fig_cluster_metrics(reports_dir: Path, outdir_fig: Path):
    metr = _read_csv_safe(reports_dir / "cluster_metrics.csv")
    if metr is None or metr.empty:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method in ["kmeans","ward"]:
        sub = metr[metr["method"]==method]
        if sub.empty: continue
        ax.plot(sub["k"], sub["silhouette"], marker="o", label=f"{method} silhouette")
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette")
    ax.set_title("Clustering quality (higher is better)")
    ax.legend()
    _save(fig, outdir_fig / "clustering_silhouette")

    fig2, ax2 = plt.subplots(figsize=(6.5, 4))
    for method in ["kmeans","ward"]:
        sub = metr[metr["method"]==method]
        if sub.empty: continue
        ax2.plot(sub["k"], sub["ARI"], marker="o", label=f"{method} ARI vs Scaffold")
    ax2.set_xlabel("k"); ax2.set_ylabel("Adjusted Rand Index")
    ax2.set_title("Agreement with Scaffold labels")
    ax2.legend()
    _save(fig2, outdir_fig / "clustering_ari")

def fig_dendrogram(reports_dir: Path, outdir_fig: Path):
    L = _read_csv_safe(reports_dir / "linkage_ward.csv")
    if L is None or L.empty: return
    Z = L.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    dendrogram(Z, ax=ax, no_labels=True, color_threshold=None)
    ax.set_title("Hierarchical clustering — Ward linkage")
    _save(fig, outdir_fig / "dendrogram_ward")

# ---------- MCDM figures ----------
def fig_mcdm_borda(reports_dir: Path, outdir_fig: Path):
    borda = _read_csv_safe(reports_dir / "mcdm_borda.csv")
    if borda is None or borda.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(borda["alternative"].astype(str), borda["borda_total"].values)
    ax.set_xlabel("Alternative (Scaffold)")
    ax.set_ylabel("Borda score (higher is better)")
    ax.set_title("Consensus ranking (Borda aggregation)")

    # Rotazione + allineamento delle etichette sull'asse X
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    for lab in ax.get_xticklabels():
        try:
            lab.set_horizontalalignment('right')
        except Exception:
            pass

    _save(fig, outdir_fig / "mcdm_borda")


def main():
    ap = argparse.ArgumentParser(description="Generate figures from reports CSVs")
    ap.add_argument("--reports", default="reports", help="Cartella dei CSV di output della pipeline")
    ap.add_argument("--outdir", default=None, help="Cartella destinazione figure (default: reports/figures)")
    args = ap.parse_args()

    in_dir = Path(args.reports)
    out_fig = Path(args.outdir) if args.outdir else in_dir / "figures"
    _ensure_outdir(out_fig)

    print(f"[fig] input = {in_dir.resolve().as_posix()}")
    print(f"[fig] out    = {out_fig.resolve().as_posix()}")

    fig_cv_results(in_dir / "cv_results.csv", out_fig)
    fig_permimp_all(in_dir, out_fig)
    fig_bootstrap_convergence(in_dir / "bootstrap_convergence_sigma_max.csv", out_fig)
    fig_stats_overview(in_dir / "stats_overview.csv", out_fig)
    fig_pca_biplot(in_dir, out_fig)
    fig_cluster_metrics(in_dir, out_fig)
    fig_dendrogram(in_dir, out_fig)
    fig_mcdm_borda(in_dir, out_fig)

    print("[fig] done.")

if __name__ == "__main__":
    main()


#python .\scripts\make_figures.py --reports ".\reports" --outdir ".\reports\figures"
