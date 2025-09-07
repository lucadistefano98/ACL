
# Anterior Cruciate Ligmanent (ACL) Scaffold - Analysis and Modeling

## Description
This project provides a complete pipeline for statistical analysis, predictive modeling, and ranking of ACL scaffolds, starting from experimental data in Excel format.

## Main Features
- **Data loading** from Excel files
- **Statistical analysis** (ANOVA, Kruskal, MANOVA, Hotelling T²)
- **PCA** and clustering (KMeans, Ward)
- **Predictive modeling** (Random Forest, Linear Regression, XGBoost)
- **Permutation Importance** (with bootstrap)
- **MCDM ranking** (PROMETHEE II, TOPSIS, ELECTRE III, Borda)
- **Automatic reports** and result saving

## Project Structure
- `src/acl_scaffold/`: Main Python modules
- `scripts/`: Analysis and figure generation scripts
- `data/`: Input data (Excel)
- `reports/`: Output and results (CSV, PNG, PDF)

## Requirements
See `requirements.txt`:
- pandas, numpy, scipy, scikit-learn, statsmodels, scikit-posthocs, openpyxl, xgboost

## Example usage
```python
from acl_scaffold import report
report.run_all("data/Matrice_risultati.xlsx", "reports/")
```

## Author
Simone Micalizzi, Luca Di Stefano
