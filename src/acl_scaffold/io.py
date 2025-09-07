from __future__ import annotations
from pathlib import Path
import pandas as pd

# Mappa: nomi originali Excel -> nomenclatura del paper
RENAME_MAP = {
    # gruppo
    'scaffold': 'Scaffold',
    'Scaffold': 'Scaffold',

    # geometriche
    'D_int': 'DI',
    'D_ext': 'DACL',
    'H_elettr': 'LES',
    'N_int': 'NSlits',

    # meccaniche
    'Stress_failure': 'σmax',
    'Strain_failure': 'εf',
    'F_max': 'Fmax',
    'Stiffness_linear': 'SL',
    'E_linear': 'EL',
    'Strain_max_toe': 'εTmax',
}

def load_matrix(xlsx_path: str | Path):
    xlsx_path = str(xlsx_path)
    xls = pd.ExcelFile(xlsx_path)

    # scegli sheet esperimenti (case-insensitive) o primo
    candidates = {s.lower(): s for s in xls.sheet_names}
    sheet_name = candidates.get('esperimenti') or candidates.get('experiments') or xls.sheet_names[0]

    dati = pd.read_excel(xls, sheet_name=sheet_name)
    dati = dati.rename(columns=RENAME_MAP)

    # normalizza nome colonna di gruppo in 'Scaffold'
    if 'Scaffold' not in dati.columns:
        for cand in ['scaffold', 'group', 'Gruppo', 'Tipo', 'Geometry']:
            if cand in dati.columns:
                dati = dati.rename(columns={cand: 'Scaffold'})
                break

    ref = pd.read_excel(xls, sheet_name='riferimentoLCA') if 'riferimentoLCA' in xls.sheet_names else None
    return dati, ref
