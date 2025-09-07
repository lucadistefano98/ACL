from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster

MECH_VARS_ALL = ["σmax","εf","Fmax","SL","ST","EL","ET","T","εTmax"]

def prepare_matrix(dati: pd.DataFrame, group_col: str = "Scaffold", mech_vars: list[str] | None = None):
    if mech_vars is None:
        mech_vars = [c for c in MECH_VARS_ALL if c in dati.columns]
    df = dati[[group_col] + mech_vars].dropna().copy()
    X = df[mech_vars].apply(pd.to_numeric, errors="coerce").dropna().to_numpy(dtype=float)
    labs = df[group_col].astype(str).to_numpy()
    ids = df.index.to_numpy()
    Z = StandardScaler().fit_transform(X)
    return Z, labs, ids, mech_vars

def run_kmeans_allk(Z: np.ndarray, labels_true: np.ndarray, k_min=2, k_max=6, random_state=42):
    rows = []
    best = None
    for k in range(k_min, k_max+1):
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
        y = km.fit_predict(Z)
        sil = silhouette_score(Z, y) if len(np.unique(y)) > 1 else np.nan
        ari = adjusted_rand_score(labels_true, y)
        rows.append({"method":"kmeans","k":k,"silhouette":float(sil),"ARI":float(ari)})
        if best is None or sil > best["sil"]:
            best = {"k": k, "labels": y, "sil": sil}
    return pd.DataFrame(rows), best

def run_ward_allk(Z: np.ndarray, labels_true: np.ndarray, k_min=2, k_max=6):
    L = linkage(Z, method="ward")
    rows = []
    best = None
    for k in range(k_min, k_max+1):
        y = fcluster(L, t=k, criterion="maxclust")
        sil = silhouette_score(Z, y) if len(np.unique(y)) > 1 else np.nan
        ari = adjusted_rand_score(labels_true, y)
        rows.append({"method":"ward","k":k,"silhouette":float(sil),"ARI":float(ari)})
        if best is None or sil > best["sil"]:
            best = {"k": k, "labels": y, "sil": sil}
    # ritorno anche la matrice di linkage per il dendrogramma
    return pd.DataFrame(rows), best, L

def save_linkage_csv(L: np.ndarray, out_csv: Path):
    # scipy linkage: (n-1) x 4 (idx1, idx2, dist, sample_count)
    df = pd.DataFrame(L, columns=["idx1","idx2","dist","count"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

def run_clustering_and_save(dati: pd.DataFrame, outdir: Path, group_col="Scaffold", mech_vars=None):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    Z, labs, ids, used_vars = prepare_matrix(dati, group_col, mech_vars)

    # KMeans
    k_tab, k_best = run_kmeans_allk(Z, labs)
    # Ward
    w_tab, w_best, L = run_ward_allk(Z, labs)

    metrics = pd.concat([k_tab, w_tab], ignore_index=True)
    metrics.to_csv(outdir / "cluster_metrics.csv", index=False)
    save_linkage_csv(L, outdir / "linkage_ward.csv")

    # assegnazioni
    assign_k = pd.DataFrame({"sample_id": ids, group_col: labs, f"kmeans_k{k_best['k']}": k_best["labels"]})
    assign_w = pd.DataFrame({"sample_id": ids, group_col: labs, f"ward_k{w_best['k']}": w_best["labels"]})
    assign_k.to_csv(outdir / "cluster_assignments_kmeans.csv", index=False)
    assign_w.to_csv(outdir / "cluster_assignments_ward.csv", index=False)

    return metrics, assign_k, assign_w, used_vars
