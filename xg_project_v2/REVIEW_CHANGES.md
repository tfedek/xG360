# Review Changes – v2 (Professor feedback)

## Summary

All changes address the reviewer's methodological concerns.
New/modified files are clearly marked. Original files are preserved.

---

## [P1] In-fold isotonic calibration — `03_train_models.py` (rewritten)

**Problem:** Isotonic calibration was fitted on the full dataset,
so the calibrated Brier score on the test fold was optimistically biased
(the calibrator had seen test labels during training).

**Fix:** `CalibratedClassifierCV(method="isotonic", cv=3)` is now fitted
**only on the training split** of each outer fold. The test fold receives
predictions from a calibrator that has never seen its labels.

Both `brier_uncalibrated` and `brier_calibrated` are now reported per fold,
so the reader can see the calibration benefit without confusion about leakage.

---

## [P2] class_weight interaction with calibration — `03_train_models.py`

`class_weight` remains part of the `GridSearchCV` hyperparameter grid
(as before), but now the calibration step is applied *after* the best
class_weight is selected, within the same fold. This ensures that
the class_weight and calibrator are jointly optimised on the training
data only.

---

## [P3] goalkeeper_anomaly / feature leakage — `03_train_models.py`

The current feature set (`config.py`) does **not** use `goalkeeper_anomaly`
(a residual from a global regression fit in earlier pipeline versions).
The 360 goalkeeper feature used is `goalkeeper_distance_360`, computed
directly from the freeze frame without any global model fitting.
No leakage applies to the current feature set. This is documented
in the script header.

---

## [P4] Ablation analysis — `07_ablation_360.py` (new)

Tests the marginal contribution of each 360 feature group
(GK, PRESSURE, CONE, SHOT_LINE) by training Model A + one group
at a time and comparing LOTO AUC to the baseline (Model A alone)
and the full Model B.

Output: `data/outputs/ablation/ablation_360_loto_auc.csv`

---

## [P5] Paired bootstrap CI on OOF predictions — `08_paired_bootstrap_ci.py` (new)

Pools out-of-fold (OOF) predictions from all LOTO folds for Model A
and Model B logistic regression, then runs a paired bootstrap
(2000 iterations) on the paired (y, p_A, p_B) rows.

This is methodologically stronger than a single 80/20 split because:
- All data contributes to evaluation
- Bootstrap is on genuinely out-of-sample predictions
- Pairing preserves the shot-level correlation structure

Output: `data/outputs/model_training/paired_bootstrap_ci_oof_v2.csv`

---

## [P6] Unpenalized LR test — `03_train_models.py`

A separate, unpenalized, unweighted `statsmodels.Logit` is fitted on
the full dataset for the sole purpose of computing AIC, BIC, and the
Likelihood Ratio test. This is methodologically correct because:
- Regularised (sklearn, C<∞) log-likelihoods are not comparable via LR test
- class_weight changes the effective likelihood and breaks AIC/BIC

The sklearn pipeline (with regularisation + class_weight) remains the
primary model for Brier/AUC reporting. The statsmodels model is used
only for formal inference (LR test, AIC, BIC).

Output: `data/outputs/model_training/lr_test_results_v2.csv`

---

## [P7] Cluster-robust standard errors — `05_model_interpretation_v2.py` (new)

The unpenalized statsmodels Logit for Odds Ratio reporting now also
computes cluster-robust standard errors (clustered by `match_id`),
as a sensitivity check for within-match shot dependence.

Both standard and cluster-robust p-values are reported side by side.

Output: `data/outputs/interpretation/logistic_odds_ratios_model_b_v2.csv`

---

## [P8] Spatial feature documentation — `09_spatial_feature_documentation.py` (new)

- Pseudocode for `open_goal_angle_ratio`, `pressure_score`, and
  `nearest_defender_to_shot_line` (printable for paper appendix)
- Sensitivity analysis for `pressure_score` radius (5–15 m) and
  epsilon (0.1–2.0)
- Freeze-frame example visualization (synthetic, geometrically
  representative positions)

Output: `data/outputs/spatial_documentation/`

---

## [P9] Language — `05_model_interpretation_v2.py`

Plot titles and print statements now use "predictive association" and
"predictive contribution" instead of language implying causation.
The corresponding sections in `xg_rad_nacrt.docx` (Word document)
should be updated accordingly — see `WORD_CHANGES.md`.

---

## Execution order

```
python scripts/02_build_dataset.py       # unchanged
python scripts/03_train_models.py        # rewritten (P1, P2, P3, P6)
python scripts/04_evaluate_models.py     # unchanged (reads new CSV)
python scripts/05_model_interpretation_v2.py  # new (P7, P9)
python scripts/06_export_figures_and_predictions.py  # unchanged
python scripts/07_ablation_360.py        # new (P4)
python scripts/08_paired_bootstrap_ci.py # new (P5)
python scripts/09_spatial_feature_documentation.py  # new (P8)
```
