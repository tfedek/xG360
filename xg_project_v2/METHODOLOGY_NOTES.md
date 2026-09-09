# Metodološke napomene - pipeline v2

## Sažetak

Ovaj dokument opisuje metodološke odluke i skripte u v2 pipeline-u.
Nove i izmenjene skripte su jasno označene.

---

## Kalibracija unutar folda - `03_train_models.py`

Izotona kalibracija (`CalibratedClassifierCV(method="isotonic", cv=3)`)
fituje se isključivo na trening delu svakog spoljašnjeg folda. Test fold
dobija predikcije od kalibratora koji nikada nije video njegove oznake,
čime se izbegava optimistička pristrasnost kalibrisanog Brier skora.

I `brier_uncalibrated` i `brier_calibrated` izveštavaju se po foldu,
tako da se korist od kalibracije vidi bez zabune oko curenja informacija.

---

## class_weight i kalibracija - `03_train_models.py`

`class_weight` je deo `GridSearchCV` mreže hiperparametara. Korak
kalibracije primenjuje se posle izbora najboljeg `class_weight`-a, unutar
istog folda, tako da su `class_weight` i kalibrator zajednički optimizovani
samo na trening podacima.

---

## Curenje informacija (feature leakage) - `03_train_models.py`

Trenutni feature set (`config.py`) NE koristi `goalkeeper_anomaly`
(rezidual iz globalnog regresionog fita u ranijim verzijama pipeline-a).
360 golmanski atribut je `goalkeeper_distance_360`, izračunat direktno iz
freeze-frame kadra bez ikakvog globalnog fitovanja modela. Nema curenja
informacija u trenutnom feature setu.

---

## Ablation analiza - `07_ablation_360.py`

Testira marginalni doprinos svake grupe 360 atributa (GK, PRESSURE, CONE,
SHOT_LINE) treniranjem Modela A uz jednu dodatu grupu i poređenjem LOTO AUC
sa baseline-om (samo Model A) i punim Modelom B.

Izlaz: `data/outputs/ablation/ablation_360_loto_auc.csv`

---

## Parni bootstrap CI na OOF predikcijama - `08_paired_bootstrap_ci.py`

Spaja out-of-fold (OOF) predikcije iz svih LOTO foldova za Model A i
Model B (logistička regresija), pa pokreće parni klaster bootstrap
(2.000 iteracija) sa uzorkovanjem po utakmicama. Uparivanje čuva
korelacionu strukturu na nivou šuta.

Izlaz: `data/outputs/model_training/paired_bootstrap_ci_oof_v2.csv`

---

## Usklađivanje agregacije AUC - `11_auc_aggregation_reconciliation.py`

Poredi macro (prosek po turniru) i pooled (svi OOF spojeni) ROC AUC, i za
kalibrisane i za nekalibrisane predikcije, na istim opservacijama. Pokazuje
da izotona kalibracija unutar turnira ne menja rang (monotona je), dok
pooled preko turnira poravnava različite bazne stope golova.

Izlaz: `data/outputs/model_training/auc_aggregation_reconciliation.csv`

---

## Nepenalizovani LR test - `03_train_models.py`

Zaseban, nepenalizovan i neponderisan `statsmodels.Logit` fituje se na
celom skupu isključivo radi računanja AIC, BIC i Likelihood Ratio testa.
Regularizovane (sklearn, C<beskonačno) log-verodostojnosti nisu uporedive
LR testom, a `class_weight` menja efektivnu funkciju verodostojnosti i
narušava AIC/BIC. sklearn pipeline ostaje primarni model za Brier/AUC
izveštavanje; statsmodels model služi samo za formalnu inferenciju.

Izlaz: `data/outputs/model_training/lr_test_results_v2.csv`

---

## Klaster-robusne standardne greške - `05_model_interpretation_v2.py`

Nepenalizovani statsmodels Logit za Odds Ratio izveštavanje računa i
klaster-robusne standardne greške (klasteri po `match_id`), kao proveru
osetljivosti na zavisnost šuteva unutar iste utakmice. Prikazuju se i
standardne i klaster-robusne p-vrednosti.

Izlaz: `data/outputs/interpretation/logistic_odds_ratios_model_b_v2.csv`

---

## Dokumentacija prostornih atributa - `09_spatial_feature_documentation.py`

- Pseudokod za `open_goal_angle_ratio`, `pressure_score` i
  `nearest_defender_to_shot_line`
- Analiza osetljivosti za `pressure_score` radijus (5-15 m) i epsilon
  (0,1-2,0)
- Vizualizacija primera freeze-frame kadra (sintetičke, geometrijski
  reprezentativne pozicije)

Izlaz: `data/outputs/spatial_documentation/`

---

## Redosled izvršavanja

```
python scripts/02_build_dataset.py
python scripts/03_train_models.py
python scripts/04_evaluate_models.py
python scripts/05_model_interpretation_v2.py
python scripts/06_export_figures_and_predictions.py
python scripts/07_ablation_360.py
python scripts/08_paired_bootstrap_ci.py
python scripts/09_spatial_feature_documentation.py
python scripts/11_auc_aggregation_reconciliation.py
```
