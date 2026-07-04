xG360 - kompletan projekat
=============================

Doprinos StatsBomb 360 prostornih podataka u proceni verovatnoće gola (xG):
komparativna analiza klasičnih i prostorno-svesnih xG modela.


SADRŽAJ PAKETA
---------------

xg_rad_nacrt.docx     -> nacrt rada (Word), sa formulama, tabelama i nalazima
src/                  -> Python analitički kod (kompletan pipeline)
data/processed/       -> obrađeni podaci, gotov za modelovanje (CSV)
                         + fitted_models/ sa već istreniranim modelima (.joblib)
tables/               -> sve generisane tabele (CSV) - K-Fold, LOTO, SHAP, itd.
figures/              -> svi generisani grafici (PNG) - ROC, kalibracija, SHAP
web/                  -> funkcionalna web aplikacija (demonstracija uživo)
docx_generator/       -> Node.js skripte koje generišu xg_rad_nacrt.docx


NAPOMENA O SIROVIM PODACIMA
-----------------------------
Sirovi StatsBomb JSON podaci (events + 360 freeze-frame za 166 mečeva,
~1.4 GB) NISU uključeni u ovaj paket zbog veličine. Skripta
src/data_loader.py ih sama preuzima sa javnog StatsBomb open-data
GitHub repozitorijuma (github.com/statsbomb/open-data) i čuva lokalno
u data/raw/ pri prvom pokretanju. Internet konekcija je potrebna samo
za taj prvi korak.

Svi naknadni koraci (feature engineering, treniranje, evaluacija) rade
isključivo sa već obrađenim podacima iz data/processed/, koji SU
uključeni u ovaj paket - ne moraš ponovo da vučeš sirove podatke ako
samo želiš da pregledaš/reprodukuješ analizu od tog koraka nadalje.


REDOSLED POKRETANJA PYTHON ANALIZE (od nule)
-----------------------------------------------
Potrebno: Python 3.10+, i sledeći paketi:

    pip install pandas numpy scipy statsmodels scikit-learn xgboost \
                joblib shap matplotlib seaborn --break-system-packages

Iz foldera src/, redom:

    python3 data_loader.py           # preuzima sirove podatke (treba internet)
    python3 shot_extraction.py       # ekstrahuje šuteve i feature-e
    python3 preprocessing.py         # EDA, čišćenje, Model A/B feature setovi
    python3 feature_transforms.py    # VIF/log transformacije
    python3 assumption_checks.py     # provera pretpostavki (VIF, linearnost)
    python3 hypothesis_testing.py    # χ², Mann-Whitney, KS testovi
    python3 train_models.py          # treniranje + K-Fold + LOTO validacija
    python3 interpretation.py        # Odds Ratio + SHAP analiza
    python3 evaluation_plots.py      # ROC/PR/kalibracija grafici
    python3 export_for_web.py        # priprema JSON podataka za web app

Ako samo želiš da PREGLEDAŠ rezultate bez ponovnog pokretanja celog
pipeline-a, sve je već izračunato u data/processed/, tables/ i figures/.


WEB APLIKACIJA (DEMONSTRACIJA UŽIVO)
---------------------------------------
1. Otvori terminal u folderu web/
2. Pokreni: python3 -m http.server 8000
3. Otvori browser na: http://localhost:8000
4. Izaberi turnir, igrača i konkretan gol, klikni "Analiziraj"

Model A i Model B (logistička regresija) računaju se UŽIVO u browseru
(JavaScript), na osnovu koeficijenata izvezenih u web/data/model_coefs.json.
Ne treba server ni internet konekcija osim za učitavanje fonta (opciono).


GENERISANJE WORD DOKUMENTA (NACRTA RADA)
-------------------------------------------
Fajl xg_rad_nacrt.docx je već generisan i priložen. Ako želiš da ga
izmeniš i ponovo generišeš:

1. cd docx_generator
2. npm install
3. node generate.js
4. Novi fajl: docx_generator/xg_rad_nacrt.docx

Struktura generatora:
    helpers.js                - pomoćne funkcije (naslovi, tabele, paragrafi)
    formulas.js                - sve matematičke formule (OMML format za Word)
    part1_title.js             - naslovna strana
    part2_intro.js             - sažetak, uvod, literatura, podaci
    part3_methodology.js       - metodologija (9 koraka, sa formulama)
    part4_results.js           - rezultati (tabele sa stvarnim brojevima)
    part5_discussion.js        - diskusija, zaključak, literatura
    generate.js                 - glavna skripta koja sve spaja

NAPOMENA: matematičke formule (OMML) se ne prikazuju u LibreOffice-u
(poznato ograničenje), ali se ispravno prikazuju u Microsoft Word-u.
XML struktura je validirana prema OOXML schema.


GLAVNI NALAZI (KRATAK PODSETNIK)
------------------------------------
- 3.968 šuteva, 372 gola (9,4%), tri turnira: SP 2022, Euro 2020, Euro 2024
- Model B (sa 360 podacima) konzistentno nadmašuje Model A na svim
  metrikama, i u Stratified K-Fold i u Leave-One-Tournament-Out validaciji
- Otvorenost ugla ka golu je 2. najvažniji atribut po SHAP analizi
- Geometrija šuta je statistički slična kroz sva tri turnira (KS test),
  ali se stopa konverzije značajno razlikuje po turniru (χ² test)
- Post-hoc izotona kalibracija drastično poboljšava Brier skor

Detaljno objašnjenje svega navedenog: xg_rad_nacrt.docx
