const { h1, h2, h3, p, caption, bullet, numbered, makeTable, CONTENT_DXA } = require('./helpers');
const F = require('./formulas');

function buildPart3() {
  return [
    caption('Tabela 1. Pregled skupa podataka po turniru'),
    makeTable(
      ['Turnir', 'Broj mečeva', 'Broj šuteva (bez penala)', 'Broj golova', 'Stopa konverzije'],
      [
        ['Svetsko prvenstvo 2022', '64', '1.430', '152', '10,6%'],
        ['Evropsko prvenstvo 2020', '51', '1.234', '122', '9,9%'],
        ['Evropsko prvenstvo 2024', '51', '1.304', '98', '7,5%'],
        ['Ukupno', '166', '3.968', '372', '9,4%'],
      ],
      [2400, 1800, 2200, 1700, 1260]
    ),

    h1('Metodologija', true, '3'),
    p('Metodologija je sprovedena kroz devet uzastopnih koraka, gde svaki korak odgovara na konkretno pitanje vezano za centralnu temu rada, doprinos StatsBomb 360 prostornih podataka proceni verovatnoće gola. Namerno je izbegnuto nagomilavanje statističkih metoda bez jasne svrhe; svaka primenjena tehnika ima eksplicitno objašnjenje zašto je izabrana baš ona, a ne neka srodna alternativa.'),

    h2('3.1. Eksplorativna analiza podataka'),
    p('Prvi korak obuhvatio je deskriptivnu statistiku numeričkih atributa (srednja vrednost, standardna devijacija, kvartili), proveru nedostajućih vrednosti, detekciju outliera metodom interkvartilnog raspona (IQR, 1,5 puta pravilo) i korelacionu matricu (Pearson koeficijent) između numeričkih atributa.'),
    p('Korelaciona analiza je otkrila da su udaljenost golmana od šutera i udaljenost šutera od gola izrazito korelisani (r = 0,97), što je bilo logično očekivano jer golman po pravilu stoji blizu sopstvenog gola. Ovaj nalaz je dalje formalno ispitan u koraku provere pretpostavki (4.3) i metodološki rešen pre treniranja modela.'),

    h2('3.2. Testiranje statističkih hipoteza'),
    p('Drugi korak primenio je inferencijalnu statistiku isključivo na jasno definisana istraživačka pitanja, a ne kao paket testova bez konkretne svrhe:'),
    bullet('H1: Da li je tip akcije koja prethodi šutu (play pattern) povezan sa verovatnoćom gola? Test: hi-kvadrat test nezavisnosti.'),
    bullet('H2: Da li se udaljenost od gola razlikuje između golova i promašaja? Test: Mann-Whitney U test (neparametarski, jer distribucija udaljenosti nije normalna).'),
    bullet('H3: Da li se ugao šuta razlikuje između golova i promašaja? Test: Mann-Whitney U test.'),
    bullet('H4: Da li se stopa konverzije šuteva u gol razlikuje između tri turnira? Test: hi-kvadrat test nezavisnosti.'),
    p('Hi-kvadrat statistika za testove H1 i H4 računa se kao zbir kvadrata razlika opaženih (O) i očekivanih (E) frekvencija po ćeliji kontingencijske tabele, normalizovan očekivanom frekvencijom:'),
    F.eqChiSquare(),
    p('gde je i indeks koji se kreće kroz sve ćelije kontingencijske tabele, a n ukupan broj ćelija. Mann-Whitney U statistika za testove H2 i H3 izračunava se na osnovu rangova opservacija u dve grupe (gol, ne-gol):'),
    F.eqMannWhitney(),
    p('gde su n₁ i n₂ veličine dve grupe, a R₁ zbir rangova prve grupe u zajedničkom uređenju oba uzorka.'),
    p('Dodatno, sprovedena je formalna provera distribucijskog pomaka Kolmogorov-Smirnov testom za ključne numeričke atribute (udaljenost, ugao) između svaka dva turnira:'),
    F.eqKS(),
    p('gde su F₁ i F₂ empirijske kumulativne funkcije raspodele numeričkog atributa za dva turnira, a D je maksimalna apsolutna razlika između njih (statistika Kolmogorov-Smirnov testa). Ovaj test je direktno relevantan za tumačenje validacionih rezultata u koraku 3.6.'),

    h2('3.3. Priprema i pretprocesiranje podataka'),
    p('Treći korak obuhvatio je čišćenje podataka, kodiranje kategorijskih promenljivih (one-hot encoding), grupisanje retkih kategorija i formalnu proveru pretpostavki logističke regresije:'),
    numbered('Multikolinearnost (VIF, Variance Inflation Factor): utvrđeno je da udaljenost golmana od šutera ima VIF preko 25 zbog korelacije sa udaljenosti šutera od gola. VIF za i-ti atribut se računa kao:'),
    F.eqVIF(),
    p('gde je Rᵢ² koeficijent determinacije iz regresije i-tog atributa na sve ostale atribute u modelu. Vrednost VIF preko 5 do 10 se uobičajeno smatra znakom problematične multikolinearnosti. Problem je rešen zamenom apsolutne udaljenosti golmana sa rezidualom (anomalijom) u odnosu na očekivanu poziciju golmana za datu udaljenost šuta:'),
    F.eqGkAnomaly(),
    p('gde su a i b koeficijenti proste linearne regresije udaljenosti golmana (d_gk) na udaljenost šuta (d_shot), procenjeni na celom skupu podataka. Ovom transformacijom VIF je spušten ispod 2.'),
    p('Nakon ove korekcije, preostale promenljive sa VIF iznad 5 odnose se isključivo na dummy varijable tehnike šuta, naročito na dominantnu kategoriju (technique_Normal, VIF približno 51). Ovo je strukturni artefakt one-hot kodiranja u kombinaciji sa izrazito neravnomernom zastupljenošću kategorija (kategorija Normal čini približno 75% svih šuteva), a ne znak problematične multikolinearnosti između suštinski nezavisnih prediktora. Interpretacija koeficijenata ostaje validna, jer se svaki koeficijent dummy varijable tumači u odnosu na izostavljenu referentnu kategoriju, ne u odnosu na ostale dummy varijable.'),
    numbered('Linearnost logita (Box-Tidwell test): utvrđeno je da udaljenost od gola i ugao šuta narušavaju pretpostavku linearnosti logita. Box-Tidwell test se sprovodi dodavanjem interakcionog člana x·ln(x) u logistički model:'),
    F.eqBoxTidwell(),
    p('Statistički značajan koeficijent β₂ (p < 0,05) ukazuje na narušenu linearnost logita za atribut x. Problem je rešen log-transformacijom udaljenosti i ugla za varijantu modela namenjenu logističkoj regresiji, dok je varijanta za XGBoost ostala na sirovim vrednostima, jer stabla odluke prirodno modeluju nelinearne odnose.'),
    numbered('Neuravnoteženost klasa: odnos ne-gol prema gol je približno 9,7 prema 1, što je ocenjeno kao umerena, ne ekstremna neuravnoteženost. Rešeno je ponderisanjem klasa (class_weight u logističkoj regresiji, scale_pos_weight u XGBoost-u) izvedenim direktno iz stvarnog odnosa klasa u podacima, bez veštačkog resampling-a.'),
    numbered('Retke kategorije: tipovi šuta Corner (6 opservacija, 0 golova) i Free Kick (130 opservacija, 3 gola) uzrokovali su kvazi-savršenu separaciju u logističkoj regresiji. Rešeno je agregacijom u jedinstvenu kategoriju Set Piece pre kodiranja.'),

    h2('3.4. Definisanje dva skupa atributa'),
    p('Centralna teza rada zahteva eksplicitno poređenje dva modela, ne samo apsolutnu procenu jednog modela. Model A koristi isključivo klasične event atribute: udaljenost od gola, ugao šuta, minut meča, deo tela, tehniku, tip akcije, tip šuta, stanje rezultata, da li je šut izveden pod pritiskom i da li je prvim dodirom. Model B sadrži sve atribute Modela A, uz dodatak StatsBomb 360 prostornih atributa: broj branilaca u liniji šuta, broj saigrača u istoj zoni, broj protivnika u krugu od 5 metara, otvorenost ugla ka golu (udeo gol-okvira koji nije zaklonjen), anomaliju pozicije golmana, najbližu udaljenost branioca od linije šuta i kontinuiranu meru defanzivnog pritiska.'),
    p('Razlika u performansu i interpretaciji između ova dva modela predstavlja glavni rezultat rada, jer direktno odgovara na pitanje koliko prostorni podaci doprinose proceni.'),

    h2('3.5. Razvoj prediktivnih modela'),
    p('Za oba skupa atributa trenirana su dva modela: logistička regresija, koja pruža interpretabilne koeficijente u obliku Odds Ratio, i XGBoost, gradijentno pojačavanje stabala odluke, koje bolje hvata nelinearne interakcije između atributa. Logistička regresija modeluje logaritam šansi (log-odds) ishoda kao linearnu kombinaciju atributa:'),
    F.eqLogit(),
    p('gde je p verovatnoća gola, a β₀, β₁, ..., βₖ koeficijenti modela procenjeni metodom maksimalne verodostojnosti. Predviđena verovatnoća se zatim dobija primenom sigmoid (logističke) funkcije na linearnu kombinaciju z:'),
    F.eqSigmoidFixed(),
    p('Hiperparametri su podešavani metodom Randomized Search uz unutrašnju petokratnu Stratified K-Fold unakrsnu validaciju, čime je obezbeđeno da test podaci nikada nisu učestvovali u procesu podešavanja.'),

    h2('3.6. Validacija modela'),
    p('Validacija je sprovedena u dva nivoa koji odgovaraju na različita pitanja:'),
    bullet('Stratified K-Fold (5 foldova): koliko model radi na podacima iz iste distribucije kao i podaci za treniranje.'),
    bullet('Leave-One-Tournament-Out: koliko model generalizuje na potpuno nov turnir koji nije učestvovao u treningu. Za svaki od tri turnira, model je treniran na preostala dva turnira (uključujući ponovno podešavanje hiperparametara isključivo na tom trening skupu) i testiran na izostavljenom turniru.'),
    p('Ovakav dvostepeni pristup namerno je odabran jer K-Fold i Leave-One-Tournament-Out odgovaraju na različita pitanja: prvi procenjuje performans unutar poznate distribucije, a drugi procenjuje robusnost modela na potpuno nov kontekst. Treba napomenuti da Leave-One-Tournament-Out validacija sa samo tri turnira ima ograničenu statističku snagu za procenu varijanse generalizacije, te da su Evropsko prvenstvo 2020 i Evropsko prvenstvo 2024 strukturno sličniji jedan drugom (isto takmičenje, različite edicije) nego Svetskom prvenstvu, što treba imati u vidu pri tumačenju rezultata.'),

    h2('3.7. Evaluacija diskriminacije i kalibracije'),
    p('Modeli su evaluirani po dve nezavisne dimenzije kvaliteta. Diskriminacija, odnosno da li model ispravno rangira šuteve po riziku, merena je preko ROC AUC, Precision-Recall AUC (PR AUC, posebno informativan kod neuravnoteženih klasa), preciznosti, odziva, F1 mere i matrice konfuzije. Kalibracija, odnosno da li predviđena verovatnoća brojčano odgovara stvarnoj frekvenciji ishoda, merena je preko Brier skora, koji predstavlja srednju kvadratnu grešku između predviđene verovatnoće pᵢ i stvarnog binarnog ishoda oᵢ za svih N opservacija:'),
    F.eqBrier(),
    p('i preko kalibracione krive (reliability diagram), koja vizuelno prikazuje odnos predviđene verovatnoće i stvarne frekvencije ishoda po binovima.'),
    p('Vizuelna provera kalibracione krive otkrila je da sirovi modeli sistematski precenjuju verovatnoću gola u višem opsegu predviđenih vrednosti, što je očekivana posledica ponderisanja klasa radi poboljšanja diskriminacije manjinske klase. Stoga je primenjena post-hoc kalibracija metodom izotone regresije, koja je značajno poboljšala Brier skor (na primer, za Model B XGBoost sa 0,175 na 0,070).'),

    h2('3.8. Interpretacija modela'),
    p('Za logističku regresiju izračunati su Odds Ratio sa Wald 95% intervalima poverenja i p-vrednostima za svaki atribut. Odds Ratio za atribut i predstavlja faktor za koji se šansa (odds) gola množi kada se vrednost tog atributa poveća za jedinicu, uz fiksirane vrednosti svih ostalih atributa, i izračunava se kao exponencijal odgovarajućeg koeficijenta logističke regresije:'),
    F.eqOddsRatio(),
    p('Vrednost OR veća od 1 ukazuje na pozitivan efekat atributa na šansu gola, a vrednost manja od 1 na negativan efekat.'),
    p('Za XGBoost je primenjena SHAP (SHapley Additive exPlanations) analiza, koja predikciju modela za konkretnu opservaciju f(x) rastavlja na zbir bazne vrednosti φ₀ (prosečna predikcija na celom skupu) i doprinosa svakog atributa φᵢ:'),
    F.eqShap(),
    p('SHAP vrednosti su izvedene iz teorije kooperativnih igara (Shapley vrednosti) i imaju svojstvo da se zbir svih doprinosa φᵢ, uz baznu vrednost φ₀, egzaktno svodi na predviđenu vrednost modela za tu konkretnu opservaciju. Analiza je sprovedena bez unapred određenih očekivanja o tome koji će atributi biti najuticajniji, kako bi se izbeglo da metodologija unapred pretpostavi rezultat. Posebna pažnja posvećena je poređenju važnosti atributa između Modela A i Modela B, jer to poređenje direktno pokazuje kako se uticaj klasičnih atributa menja kada se dodaju prostorni podaci.'),

    h2('3.9. Diskusija'),
    p('Završni korak metodologije objedinjuje nalaze iz prethodnih koraka u celovitu diskusiju o doprinosu StatsBomb 360 atributa, generalizaciji modela kroz turnire, praktičnoj primeni nalaza i ograničenjima istraživanja, što je detaljno razrađeno u poglavlju 5.'),
  ];
}

module.exports = { buildPart3 };
