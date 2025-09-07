#!/usr/bin/env python3
from __future__ import annotations
import argparse, sys, traceback
from pathlib import Path

def main():
    try:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

        src_path = (Path(__file__).resolve().parents[1] / "src").as_posix()
        sys.path.insert(0, src_path)
        print(f"[run] starting...  (sys.path[0]={src_path})")

        from acl_scaffold import report

        ap = argparse.ArgumentParser(description="Run full analysis pipeline")
        ap.add_argument("--xlsx", required=True, help="Percorso al file Excel dei dati")
        ap.add_argument("--outdir", default="reports", help="Cartella output")
        ap.add_argument("--backend", choices=["cpu", "gpu"], default="cpu",
                        help="Motore ML: cpu=scikit-learn RF, gpu=XGBoost")
        ap.add_argument("--permimp-all", action="store_true",
                        help="Calcola permutation importance per tutte le variabili meccaniche (non solo Fmax)")
        args = ap.parse_args()

        print(f"[run] xlsx={args.xlsx}")
        print(f"[run] outdir={args.outdir}")
        print(f"[run] backend={args.backend}")
        print(f"[run] permimp-all={args.permimp_all}")

        report.run_all(args.xlsx, args.outdir, backend=args.backend, permimp_all=args.permimp_all)
        print("[run] done. Outputs ->", Path(args.outdir).resolve().as_posix())

    except SystemExit:
        raise
    except Exception as e:
        print("[run] ERROR:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()


#python .\scripts\analysis.py --backend gpu --permimp-all --xlsx ".\data\Matrice_risultati.xlsx" --outdir ".\reports"
