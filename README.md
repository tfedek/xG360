# xG360 - Expected Goals with StatsBomb 360 Positional Data

Statistical and machine learning analysis investigating whether StatsBomb 360 positional data (player positions at the moment of a shot) improve Expected Goals (xG) estimation compared to classical shot-level attributes alone.

## Key Findings

- Model B (with 360 spatial data) consistently outperforms Model A (classical) across all metrics
- Validated through Stratified K-Fold and Leave-One-Tournament-Out (3 tournaments)
- Likelihood Ratio test confirms significance (LR=70.03, p=1.46x10^-12)
- Open goal angle ratio (a 360-derived feature) ranks as the 2nd most impactful predictor

## Data

- StatsBomb open-data: World Cup 2022, Euro 2020, Euro 2024
- 3,968 shots (penalties excluded), 372 goals
- Full 360 freeze-frame coverage across all matches

## Methodology

1. Exploratory data analysis
2. Hypothesis testing (chi-square, Mann-Whitney, Kolmogorov-Smirnov)
3. Preprocessing and assumption checks (VIF, Box-Tidwell, class imbalance)
4. Two feature sets: Model A (classical) vs Model B (classical + 360 spatial)
5. Logistic Regression and XGBoost with hyperparameter tuning
6. Validation: Stratified K-Fold + Leave-One-Tournament-Out
7. Evaluation: ROC AUC, PR AUC, Brier Score, Calibration Curves
8. Interpretation: Odds Ratio, SHAP analysis
9. Discussion and practical application

## Structure

```
src/               - Python pipeline (10 scripts, run in order)
web/               - Interactive web demo (HTML/JS/CSS)
docx_generator/    - Word document generator (Node.js)
tables/            - Result tables (CSV)
figures/           - Generated plots (PNG)
```

## Web Demo

Live: [pa-ft.com/xg360](https://pa-ft.com/xg360/)

## Running the Pipeline

```bash
pip install pandas numpy scipy scikit-learn xgboost shap statsmodels joblib matplotlib seaborn
cd src
python data_loader.py
python shot_extraction.py
python preprocessing.py
python feature_transforms.py
python assumption_checks.py
python hypothesis_testing.py
python train_models.py
python interpretation.py
python evaluation_plots.py
python export_for_web.py
```

## References

- Singh S (2025). Improving expected Goals (xG) models. *Journal of High School Science*, 9(3).
- Iapteff L et al. (2025). Toward interpretable expected goals modeling. *Frontiers in Sports and Active Living*, 7.
- van der Wurp H et al. (2020). Generalised joint regression for count data. *Statistics and Computing*, 30(5).
- Degrenne O, Carling C (2024). Comparison of goalscoring patterns. *Frontiers in Sports and Active Living*, 6.
