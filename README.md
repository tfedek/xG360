# xG360

Istraživanje da li StatsBomb 360 podaci o pozicijama igrača na terenu u trenutku šuta poboljšavaju xG procenu u odnosu na klasične atribute (udaljenost, ugao, deo tela, tip akcije).

## Podaci

- StatsBomb open-data: SP 2022 (64 meča), EP 2020 (51), EP 2024 (51)
- 3.968 šuteva, 372 gola (penali isključeni)
- Puno 360 freeze-frame pokrivanje

## Rezultati

- Model B (sa 360 atributima) LOTO AUC = 0,772 vs Model A = 0,761
- LR test: 67,97 (df=11, p=2,97e-10)
- Cluster bootstrap CI: [-0,033, 0,025] (sadrži nulu)
- Validacija: StratifiedGroupKFold po meču + Leave-One-Tournament-Out

## Struktura

```
xg_project_v2/           - Python pipeline
  football_xg/           - Config, modeling utilities
  scripts/               - 02-10 (run in order)
web/                     - Interaktivna web aplikacija
docx_generator/          - Word dokument generator (Node.js)
```

## Pokretanje

```bash
cd xg_project_v2
pip install -r requirements.txt  # ili rucno: pandas numpy scipy scikit-learn xgboost shap statsmodels
PYTHONPATH=. python scripts/02_build_dataset.py
PYTHONPATH=. python scripts/03_train_models.py
PYTHONPATH=. python scripts/07_ablation_360.py
PYTHONPATH=. python scripts/08_paired_bootstrap_ci.py
PYTHONPATH=. python scripts/09_scoring_sensitivity.py
PYTHONPATH=. python scripts/10_freeze_frame_visibility.py
```

## Web demo

https://pa-ft.com/xg360/

## Reference

- Singh S (2025). Improving expected Goals (xG) models. Journal of High School Science, 9(3).
- Iapteff L et al. (2025). Toward interpretable expected goals modeling. Frontiers in Sports and Active Living, 7.
- van der Wurp H et al. (2020). Generalised joint regression for count data. Statistics and Computing, 30(5).
- Degrenne O, Carling C (2024). Comparison of goalscoring patterns. Frontiers in Sports and Active Living, 6.
- StatsBomb (2023). StatsBomb open data. https://github.com/statsbomb/open-data
