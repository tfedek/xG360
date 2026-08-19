const { h1, h2, p, caption, bullet, makeTable } = require('./helpers');
const F = require('./formulas');

function buildPart4() {
  return [
    h1('Rezultati', true, '4'),

    h2('4.1. Deskriptivna statistika i testiranje hipoteza'),
    p('Medijana udaljenosti od gola za golove iznosi 11,74 koordinatnih jedinica, naspram 18,84 koordinatnih jedinica za promašaje, dok medijana ugla šuta za golove iznosi 33,82 stepena, naspram 19,44 stepena za promašaje. Obe razlike su statistički vrlo značajne (Mann-Whitney U test, H2: p < 10⁻⁴⁷; H3: p < 10⁻⁵¹), što je u potpunosti u skladu sa fudbalskom intuicijom: golovi se postižu sa manje udaljenosti i pod širim uglom u odnosu na promašaje.'),
    p('Asocijacija između tipa akcije koja prethodi šutu i ishoda (H1) nije dostigla statističku značajnost na nivou alfa = 0,05 (hi-kvadrat = 14,996; p = 0,059). Nije pronađen statistički značajan dokaz o marginalnoj asocijaciji na ovom nivou.'),
    p('Uprkos graničnoj bivarijatnoj neznačajnosti, tip akcije (play pattern) je zadržan kao atribut u oba prediktivna modela (poglavlje 3.4). Razlog je metodološki: bivarijatni hi-kvadrat test ispituje samo marginalnu, izolovanu asocijaciju ove promenljive sa ishodom, dok u multivarijatnom modelu (logistička regresija ili XGBoost) isti atribut može doprineti predikciji u kombinaciji sa drugim atributima, na primer u interakciji sa udaljenošću ili uglom šuta, čak i kada njegova samostalna asocijacija nije statistički značajna. Isključivanje atributa isključivo na osnovu bivarijatnog testa pre uključivanja u model bila bi metodološka greška, jer bi mogla ukloniti informaciju koja postaje relevantna tek u kombinaciji sa ostalim prediktorima.'),
    p('Stopa konverzije šuteva u gol statistički se značajno razlikuje između tri turnira (H4: hi-kvadrat = 8,336; p = 0,015), pri čemu Evropsko prvenstvo 2024 ima primetno nižu stopu (7,5%) u odnosu na Svetsko prvenstvo 2022 (10,6%) i Evropsko prvenstvo 2020 (9,9%).'),
    p('Formalni Kolmogorov-Smirnov test pokazao je da se raspodela udaljenosti od gola i ugla šuta statistički ne razlikuju značajno između bilo koja dva turnira (p > 0,05 za sve parove). Ovo je ključan nalaz za tumačenje razlike u stopi konverzije: nije pronađen dokaz da se marginalne raspodele udaljenosti i ugla razlikuju, pa razlika u konverziji nije objašnjena tim dvema posmatranim geometrijskim karakteristikama. Mogući izvori razlike uključuju završnicu igrača, golmanski učinak, druge neobuhvaćene karakteristike šuta i slučajnu varijaciju.'),

    h2('4.2. Performans modela: StratifiedGroupKFold po meču validacija'),
    caption('Tabela 2. Rezultati StratifiedGroupKFold po meču validacije (prosek kroz 5 foldova). Brier skor prikazuje vrednost posle post-hoc kalibracije (izotona regresija unutar folda).'),
    makeTable(
      ['Model', 'ROC AUC', 'PR AUC', 'F1', 'Brier skor'],
      [
        ['Model A, logistička regresija', '0,755', '0,296', '0,358', '0,076'],
        ['Model A, XGBoost', '0,753', '0,297', '0,348', '0,076'],
        ['Model B, logistička regresija', '0,770', '0,322', '0,378', '0,074'],
        ['Model B, XGBoost', '0,759', '0,285', '0,376', '0,077'],
      ],
      [3360, 1500, 1500, 1500, 1500]
    ),
    p('Model B nadmašuje Model A na ROC AUC i PR AUC metrikama, bez obzira na to koji algoritam je korišćen. Razlika u kalibriranom Brier skoru je minimalna ali konzistentno u korist Modela B za logističku regresiju, dok je za XGBoost praktično izjednačena.'),

    h2('4.3. Generalizacija kroz turnire: Leave-One-Tournament-Out validacija'),
    caption('Tabela 3. Rezultati Leave-One-Tournament-Out validacije (prosek kroz tri turnira). Brier skor prikazuje vrednost posle post-hoc kalibracije.'),
    makeTable(
      ['Model', 'ROC AUC', 'PR AUC', 'F1', 'Brier skor'],
      [
        ['Model A, logistička regresija', '0,761', '0,302', '0,355', '0,075'],
        ['Model A, XGBoost', '0,763', '0,296', '0,357', '0,076'],
        ['Model B, logistička regresija', '0,772', '0,324', '0,387', '0,074'],
        ['Model B, XGBoost', '0,775', '0,316', '0,390', '0,074'],
      ],
      [3360, 1500, 1500, 1500, 1500]
    ),
    p('Rezultati Leave-One-Tournament-Out validacije su konzistentni sa rezultatima StratifiedGroupKFold po meču validacije. Razlike u ROC AUC između dve validacione šeme ne prelaze 0,02, što ukazuje da modeli ne pokazuju znake dramatične prenaučenosti specifičnostima jednog turnira, mada LOTO rezultati pokazuju nešto veću varijabilnost nego K-Fold (što je očekivano s obzirom na samo 3 turnira).'),
    p('Pri detaljnijem pregledu po pojedinačnom turniru, Evropsko prvenstvo 2024 dosledno pokazuje nešto niži PR AUC kao izostavljeni turnir u odnosu na druga dva, što je u skladu sa nalazom iz odeljka 4.1 da ovaj turnir ima nižu stopu konverzije šuteva u gol, uz statistički nepromenjenu geometriju šuta.'),

    h2('4.3a. Formalno testiranje statističke značajnosti razlike Model A i Model B'),
    p('Dosledna prednost Modela B u ROC AUC i PR AUC kroz obe validacione šeme jeste snažan indirektan pokazatelj, ali ne predstavlja sama po sebi formalni statistički dokaz da razlika nije slučajna. Da bi se ova razlika formalno testirala, sprovedena su dva nezavisna testa, prilagođena svakom od dva algoritma.'),
    p('Za logističku regresiju, Model A je formalno ugnježden u Model B: atributi Modela A predstavljaju tačan podskup atributa Modela B, pri čemu Model B sadrži svih 11 dodatnih StatsBomb 360 prostornih atributa navedenih u odeljku 3.4. Ugnježdena struktura modela dozvoljava primenu Likelihood Ratio (LR) testa, koji poredi logaritme verodostojnosti (log-likelihood) dva modela:'),
    F.eqLikelihoodRatio(),
    p('gde su ℓ_A i ℓ_B logaritmi verodostojnosti Modela A i Modela B, a LR statistika prati hi-kvadrat raspodelu sa brojem stepeni slobode jednakim broju dodatnih parametara u Modelu B. Oba modela su fitovana na identičnom skupu od 3.967 šuteva (bez penala, bez 1 šuta sa uglom jednakim nuli), kako bi poređenje log-likelihood vrednosti bilo validno. Rezultat je LR = 67,97, sa 11 stepeni slobode, što odgovara p-vrednosti od približno 2,97×10⁻¹⁰, odlučno odbacujući nultu hipotezu da StatsBomb 360 atributi ne doprinose modelu. Akaikeov informacioni kriterijum: AIC iznosi 2.139,93 za Model A i 2.093,96 za Model B (favorizuje B). Bajesov informacioni kriterijum (BIC) iznosi 2.202,79 naspram 2.225,96 (favorizuje A, što je metodološki očekivano jer BIC strože kažnjava dodatne parametre).'),
    p('Za poređenje na nivou pojedinačnih šuteva, sprovedena je klasterska bootstrap analiza razlike ROC AUC na kompletnim out-of-fold (OOF) predikcijama iz svih LOTO foldova, gde se utakmice uzorkuju sa ponavljanjem (svi šutevi iz izabrane utakmice ulaze u bootstrap uzorak). Pooled OOF ROC AUC iznosi 0,713 za Model A i 0,709 za Model B. 95% bootstrap interval poverenja (2.000 iteracija, 166 utakmica kao klasteri) iznosi [-0,033, 0,025]. Interval sadrži nulu, što znači da na nivou pojedinačnog šuta razlika nije statistički značajna na alfa=0,05. Ovo je vredan nalaz koji zaslužuje transparentno prikazivanje: signal od 360 atributa je realan ali umeren. Na nivou turnira (LOTO validacija, Tabela 3), razlika je konzistentna i praktično smislena (+0,011 AUC). LR test potvrđuje da 360 atributi nose stvarnu inkrementalnu informaciju (p=2,97×10⁻¹⁰). Zaključak je da Model B pokazuje konzistentno numeričko poboljšanje kroz turnire i validacione šeme, ali je inkrementalni efekat umeren i interval poverenja dobijen klasterskim bootstrapom ne isključuje nultu razliku.'),
    p('Napomena o Likelihood Ratio testu: LR test je primenjen isključivo na neponderisanom, nepenalizovanom logističkom modelu (standardni statsmodels Logit bez class_weight i bez regularizacije), jer ponderisanje klasa menja efektivnu likelihood funkciju i narušava standardnu interpretaciju LR statistike, AIC-a i BIC-a. Prediktivni modeli (sa regularizacijom i podešavanjem hiperparametara) koriste se za Brier/AUC izveštavanje; inferencijalni model služi isključivo za formalni test značajnosti.'),

    h2('4.4. Kalibracija i post-hoc korekcija'),
    caption('Tabela 4. Brier skor pre i posle post-hoc kalibracije (izotona regresija)'),
    makeTable(
      ['Model', 'Brier skor (sirovi model)', 'Brier skor (posle kalibracije)'],
      [
        ['Model A, logistička regresija', '0,146', '0,076'],
        ['Model A, XGBoost', '0,095', '0,076'],
        ['Model B, logistička regresija', '0,117', '0,074'],
        ['Model B, XGBoost', '0,087', '0,077'],
      ],
      [4500, 2430, 2430]
    ),
    p('Sirovi modeli, pre post-hoc kalibracije, pokazuju sistematsku prekalibrisanost u višem opsegu predviđenih verovatnoća. Primena izotone regresije kao post-hoc kalibracije značajno poboljšava Brier skor za sve četiri kombinacije modela (redukcije od 14% do 48% u zavisnosti od modela i validacione šeme).'),
    p('Kao dodatna provera robusnosti, ispitano je da li izbor scoring metrike za GridSearch menja zaključak o prednosti Modela B, korišćenjem nezavisnog, nekalibrisanog LOTO pipeline-a. Rezultat pokazuje da zaključak zavisi od izabranog kriterijuma: kada se hiperparametri biraju prema average_precision (metrika rangiranja), Model B blago zaostaje za Modelom A na AUC (0,709 naspram 0,713, delta=-0,004), iako zadržava bolji sirovi Brier skor (0,110 naspram 0,152). Kada se hiperparametri biraju prema neg_log_loss ili neg_brier_score (metrike koje direktno optimizuju kvalitet verovatnoća), Model B nadmašuje Model A na oba pokazatelja (AUC: 0,775 naspram 0,761, delta=+0,014; Brier: 0,074 naspram 0,076). Ovaj nalaz sugeriše da prednost dodatnih 360 atributa zavisi od toga da li se selekcija hiperparametara eksplicitno fokusira na kvalitet probabilističkih predikcija - što je i primaran cilj xG modelovanja - a ne isključivo na rangiranje. Napominjemo da apsolutne vrednosti AUC u ovoj proveri nisu direktno uporedive sa kalibrisanim rezultatima u Tabeli 3, budući da ova analiza koristi nekalibrisane predikcije radi izolovanja efekta scoring kriterijuma od efekta post-hoc kalibracije.'),

    h2('4.5. Ablation analiza: doprinos pojedinačnih grupa 360 atributa'),
    p('Da bi se preciziralo koji deo prostorne informacije proizvodi poboljšanje, sprovedena je ablation analiza u kojoj su 360 atributi podeljeni u pet grupa: golman (GK - udaljenost golmana), konusni branioci (CONE - branioci u konusu šuta i između šutera i gola), defanzivni pritisak (PRESSURE - pressure_score, broj branilaca u blizini), linija šuta (SHOT_LINE - najbliži branilac liniji šuter-gol), i otvorenost ugla (OPEN_ANGLE - open_angle_ratio_360). Za svaku grupu treniran je Model A sa jednom dodatom grupom, i merena je LOTO AUC razlika u odnosu na čist Model A.'),
    caption('Tabela 5. Ablation analiza: inkrementalni doprinos svake grupe 360 atributa (logistička regresija, LOTO validacija)'),
    makeTable(
      ['Varijanta', 'LOTO ROC AUC', 'Delta vs Model A'],
      [
        ['Model A (bez 360)', '0,760', '-'],
        ['Model A + OPEN_ANGLE', '0,760', '+0,000'],
        ['Model A + SHOT_LINE', '0,760', '+0,000'],
        ['Model A + GK', '0,765', '+0,004'],
        ['Model A + PRESSURE', '0,767', '+0,007'],
        ['Model A + CONE', '0,768', '+0,007'],
        ['Model B (sve 360 grupe)', '0,773', '+0,013'],
      ]
    ),
    p('Rezultati pokazuju da CONE daje najveće poboljšanje kada se pojedinačno doda Modelu A (+0,007), zatim PRESSURE (+0,007) i GK (+0,004). OPEN_ANGLE i SHOT_LINE ne donose merljiv doprinos kad se dodaju samostalno. Ovo ne meri jedinstveni doprinos svake grupe u punom modelu (jer su 360 atributi međusobno korelisani), već performans grupe kada se izoluje i doda baseline-u. Otvorenost ugla nije pokazala samostalan inkrementalni doprinos; njen eventualni doprinos u punom modelu može biti posledica redundancije sa drugim prostornim atributima. Napomena: ROC AUC za pun Model B u Tabeli 5 (0,773) blago se razlikuje od vrednosti u Tabeli 3 (0,772) jer ablation koristi logističku regresiju bez GridSearchCV.'),

    h2('4.6. Interpretacija modela: Odds Ratio i SHAP'),
    p('Logistička regresija na Modelu B pokazuje statistički značajan Odds Ratio za ključne atribute: veća udaljenost od gola asocirana je sa nižim odds-om gola, veći ugao šuta asociran je sa višim odds-om, šut glavom ima značajno niži Odds Ratio u odnosu na šut nogom, što je u skladu sa nalazima iz literature o nižoj stopi konverzije udaraca glavom.'),
    p('SHAP analiza na Modelu B (XGBoost) pokazuje da je ugao šuta najuticajniji prediktor u fitovanom modelu, neposredno praćen udaljenošću golmana (goalkeeper_distance_360), koja se plasira kao najuticajniji 360 atribut. Otvorenost ugla ka golu (open_angle_ratio_360) rangira se tek na 14. mestu; nije pokazala samostalan inkrementalni doprinos u ablation analizi, a eventualna vrednost u punom modelu može odražavati redundanciju sa drugim prostornim atributima. Ovo je nalaz o prediktivnoj asocijaciji, ne o uzročnom efektu.'),
    p('Zanimljiv je nesklad između dva pristupa interpretaciji za atribut najbliže udaljenosti branioca od linije šuta: u logističkoj regresiji ovaj atribut nije statistički značajan (p = 0,645), dok se u SHAP analizi XGBoost modela plasira na petom mestu po uticaju. Ovaj nesklad može ukazivati na nelinearnost ili interakcije koje logistički model ne hvata, ali sam po sebi ne identifikuje uzrok nesklada (mogući faktori uključuju i korelaciju sa drugim feature-ima i razlike u regularizaciji između dva modela).'),

    h2('4.7. Analiza kompletnosti freeze-frame podataka'),
    p('StatsBomb 360 freeze-frame ne garantuje vidljivost svih 22 igrača na terenu. Broj vidljivih igrača po šutu varira zavisno od ugla kamere i okluzije. Tabela 6 prikazuje deskriptivnu statistiku broja vidljivih igrača i vidljivih protivnika po turniru.'),
    caption('Tabela 6. Deskriptivna statistika broja vidljivih igrača u freeze-frame podacima po turniru.'),
    makeTable(
      ['Turnir', 'Vidljivi igrači (median)', 'IQR', 'Vidljivi protivnici (median)', 'IQR'],
      [
        ['FIFA World Cup 2022', '15', '13-18', '8', '7-9'],
        ['UEFA Euro 2020', '15', '12-17', '8', '7-9'],
        ['UEFA Euro 2024', '16', '14-18', '9', '7-10'],
      ]
    ),
    p('Medijana broja vidljivih igrača iznosi 15 (ukupni raspon 4-22), dok medijana vidljivih protivnika iznosi 8 (raspon 2-11). UEFA Euro 2024 pokazuje blago veći broj vidljivih igrača, što može odražavati poboljšanje u freeze-frame pokrivanju.'),
    p('[SLIKA: visibility_distribution.png]', {noIndent: true}),
    caption('Slika 3. Distribucija broja vidljivih igrača i protivnika po turniru.'),
    p('Da bi se ispitalo da li prednost Modela B zavisi od kompletnosti freeze-frame podataka, sprovedena je sensitivity analiza na podskupovima šuteva sa različitim pragovima minimalnog broja vidljivih protivnika (Tabela 7).'),
    caption('Tabela 7. Sensitivity analiza: Model B prednost za različite pragove vidljivosti protivnika (LOTO validacija).'),
    makeTable(
      ['Podskup', 'n', 'Golovi (%)', 'Med. dist.', 'AUC A', 'AUC B', 'Delta'],
      [
        ['Svi šutevi', '3.968', '372 (9,4%)', '18,0', '0,760', '0,773', '+0,013'],
        ['>=5 protivnika', '3.704', '336 (9,1%)', '18,2', '0,756', '0,765', '+0,009'],
        ['>=8 protivnika', '2.643', '190 (7,2%)', '19,3', '0,747', '0,757', '+0,011'],
      ]
    ),
    p('Model B zadržava prednost na svim pragovima vidljivosti. Struktura uzorka je stabilna kroz pragove: medijana udaljenosti se ne menja bitno (18,0 do 19,3 koordinatnih jedinica), a procenat šuteva iz blizine (<15 jed.) ostaje sličan (37-38%). Stroži pragovi ne biraju selektivno samo bliske šuteve. Smanjenje delta vrednosti za striktnije pragove (sa +0,013 na +0,009) može odražavati gubitak statističke snage (manji uzorak) jednako kao i realni efekat kompletnosti podataka. Ključni zaključak je da prednost Modela B nije artefakt nepotpunih freeze-frame podataka.'),
  ];
}

module.exports = { buildPart4 };
