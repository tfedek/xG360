const { h1, h2, p, caption, bullet, makeTable } = require('./helpers');
const F = require('./formulas');

function buildPart4() {
  return [
    h1('Rezultati', true, '4'),

    h2('4.1. Deskriptivna statistika i testiranje hipoteza'),
    p('Medijana udaljenosti od gola za golove iznosi 11,74 metara, naspram 18,84 metara za promašaje, dok medijana ugla šuta za golove iznosi 33,82 stepena, naspram 19,44 stepena za promašaje. Obe razlike su statistički vrlo značajne (Mann-Whitney U test, H2: p < 10⁻⁴⁷; H3: p < 10⁻⁵¹), što je u potpunosti u skladu sa fudbalskom intuicijom: golovi se postižu sa manje udaljenosti i pod širim uglom u odnosu na promašaje.'),
    p('Asocijacija između tipa akcije koja prethodi šutu i ishoda (H1) nije dostigla statističku značajnost na nivou α = 0,05 (hi-kvadrat = 14,996; p = 0,059), mada je rezultat granično blizu praga, što sugeriše mogući slab efekat koji bi veći uzorak mogao razjasniti.'),
    p('Uprkos graničnoj bivarijatnoj neznačajnosti, tip akcije (play pattern) je zadržan kao atribut u oba prediktivna modela (poglavlje 3.4). Razlog je metodološki: bivarijatni hi-kvadrat test ispituje samo marginalnu, izolovanu asocijaciju ove promenljive sa ishodom, dok u multivarijatnom modelu (logistička regresija ili XGBoost) isti atribut može doprineti predikciji u kombinaciji sa drugim atributima, na primer u interakciji sa udaljenošću ili uglom šuta, čak i kada njegova samostalna asocijacija nije statistički značajna. Isključivanje atributa isključivo na osnovu bivarijatnog testa pre uključivanja u model bila bi metodološka greška, jer bi mogla ukloniti informaciju koja postaje relevantna tek u kombinaciji sa ostalim prediktorima.'),
    p('Stopa konverzije šuteva u gol statistički se značajno razlikuje između tri turnira (H4: hi-kvadrat = 8,336; p = 0,015), pri čemu Evropsko prvenstvo 2024 ima primetno nižu stopu (7,5%) u odnosu na Svetsko prvenstvo 2022 (10,6%) i Evropsko prvenstvo 2020 (9,9%).'),
    p('Formalni Kolmogorov-Smirnov test pokazao je da se raspodela udaljenosti od gola i ugla šuta statistički ne razlikuju značajno između bilo koja dva turnira (p > 0,05 za sve parove). Ovo je ključan nalaz za tumačenje razlike u stopi konverzije: ako geometrija šuta, odnosno gde se šutira, nije različita na Evropskom prvenstvu 2024, a stopa konverzije jeste niža, razlika potiče iz finalizacije šanse (kvalitet izvedbe, slučajnost, golmani), a ne iz kvaliteta ili lokacije stvorenih šansi.'),

    h2('4.2. Performans modela: Stratified K-Fold validacija'),
    caption('Tabela 2. Rezultati Stratified K-Fold validacije (prosek kroz 5 foldova). Brier skor prikazuje vrednost posle post-hoc kalibracije (izotona regresija unutar folda).'),
    makeTable(
      ['Model', 'ROC AUC', 'PR AUC', 'F1', 'Brier skor'],
      [
        ['Model A, logistička regresija', '0,754', '0,301', '0,354', '0,076'],
        ['Model A, XGBoost', '0,744', '0,289', '0,345', '0,076'],
        ['Model B, logistička regresija', '0,765', '0,309', '0,383', '0,075'],
        ['Model B, XGBoost', '0,758', '0,294', '0,376', '0,076'],
      ],
      [3360, 1500, 1500, 1500, 1500]
    ),
    p('Model B nadmašuje Model A na ROC AUC i PR AUC metrikama, bez obzira na to koji algoritam je korišćen. Razlika u kalibriranom Brier skoru je minimalna ali konzistentno u korist Modela B za logističku regresiju, dok je za XGBoost praktično izjednačena.'),

    h2('4.3. Generalizacija kroz turnire: Leave-One-Tournament-Out validacija'),
    caption('Tabela 3. Rezultati Leave-One-Tournament-Out validacije (prosek kroz tri turnira). Brier skor prikazuje vrednost posle post-hoc kalibracije.'),
    makeTable(
      ['Model', 'ROC AUC', 'PR AUC', 'F1', 'Brier skor'],
      [
        ['Model A, logistička regresija', '0,760', '0,301', '0,350', '0,076'],
        ['Model A, XGBoost', '0,760', '0,293', '0,354', '0,076'],
        ['Model B, logistička regresija', '0,769', '0,320', '0,372', '0,074'],
        ['Model B, XGBoost', '0,770', '0,310', '0,389', '0,075'],
      ],
      [3360, 1500, 1500, 1500, 1500]
    ),
    p('Rezultati Leave-One-Tournament-Out validacije su konzistentni sa rezultatima Stratified K-Fold validacije. Razlike u ROC AUC između dve validacione šeme ne prelaze 0,02, što ukazuje da modeli ne pokazuju znake dramatične prenaučenosti specifičnostima jednog turnira, mada LOTO rezultati pokazuju nešto veću varijabilnost nego K-Fold (što je očekivano s obzirom na samo 3 turnira).'),
    p('Pri detaljnijem pregledu po pojedinačnom turniru, Evropsko prvenstvo 2024 dosledno pokazuje nešto niži PR AUC kao izostavljeni turnir u odnosu na druga dva, što je u skladu sa nalazom iz odeljka 4.1 da ovaj turnir ima nižu stopu konverzije šuteva u gol, uz statistički nepromenjenu geometriju šuta.'),

    h2('4.3a. Formalno testiranje statističke značajnosti razlike Model A i Model B'),
    p('Dosledna prednost Modela B kroz sve metrike i obe validacione šeme jeste snažan indirektan pokazatelj, ali ne predstavlja sama po sebi formalni statistički dokaz da razlika nije slučajna, odnosno posledica varijanse uzorka. Da bi se ova razlika formalno testirala, sprovedena su dva nezavisna testa, prilagođena svakom od dva algoritma.'),
    p('Za logističku regresiju, Model A je formalno ugnježden u Model B: atributi Modela A predstavljaju tačan podskup atributa Modela B, pri čemu su dodatni atributi StatsBomb 360 prostorni atributi (udaljenost golmana, broj branilaca u konusu šuta, protivnici između šutera i gola, protivnici u kaznenom prostoru, najbliža udaljenost branioca od linije šuta, branioci unutar 1m i 2m od linije šuta, i defanzivni pritisak). Ugnježdena struktura modela dozvoljava primenu Likelihood Ratio (LR) testa, koji poredi logaritme verodostojnosti (log-likelihood) dva modela:'),
    F.eqLikelihoodRatio(),
    p('gde su ℓ_A i ℓ_B logaritmi verodostojnosti Modela A i Modela B, a LR statistika prati hi-kvadrat raspodelu sa brojem stepeni slobode jednakim broju dodatnih parametara u Modelu B. Oba modela su fitovana na identičnom skupu od 3.968 šuteva (bez penala), kako bi poređenje log-likelihood vrednosti bilo validno. Rezultat je LR = 64,80, sa 10 stepeni slobode, što odgovara p-vrednosti od približno 4,42×10⁻¹⁰, odlučno odbacujući nultu hipotezu da StatsBomb 360 atributi ne doprinose modelu. Ovaj nalaz je dodatno potkrepljen i informacionim kriterijumima: Akaikeov informacioni kriterijum (AIC) iznosi 2.141,44 za Model A i 2.096,63 za Model B (favorizuje B), a Bajesov informacioni kriterijum (BIC) 2.204,30 naspram 2.222,35 (favorizuje A, što je metodološki očekivano jer BIC strože kažnjava dodatne parametre na uzorcima ove veličine).'),
    p('Za XGBoost, modeli nisu ugnježdeni na isti parametarski način, pa Likelihood Ratio test nije primenjiv. Umesto toga, sprovedena je bootstrap analiza razlike ROC AUC na kompletnim out-of-fold (OOF) predikcijama iz svih LOTO foldova. Za svaki šut u datasetu postoji tačno jedna predikcija dobijena iz modela koji ga nikada nije video tokom treninga. Na ovim parnim (shot-level) predikcijama sprovedeno je 2.000 bootstrap iteracija. Opažena razlika ROC AUC (Model B minus Model A) iznosi 0,061, a 95% interval poverenja iznosi [0,043, 0,079]. Pošto interval ne sadrži nulu, razlika se smatra statistički značajnom na nivou alfa = 0,05. Napomena: ova razlika (0,061) je veća od razlike u Tabeli 3 (0,010) jer bootstrap koristi pooled OOF AUC (svi šutevi zajedno kao jedan skup), dok Tabela 3 prikazuje neponderisani prosek AUC-ova izračunatih odvojeno po turniru; pooled pristup daje veći raspon jer ne prosečava preko različito teških turnira.'),
    p('Napomena o Likelihood Ratio testu: LR test je primenjen isključivo na neponderisanom, nepenalizovanom logističkom modelu (standardni statsmodels Logit bez class_weight i bez regularizacije), jer ponderisanje klasa menja efektivnu likelihood funkciju i narušava standardnu interpretaciju LR statistike, AIC-a i BIC-a. Prediktivni modeli (sa regularizacijom i podešavanjem hiperparametara) koriste se za Brier/AUC izveštavanje; inferencijalni model služi isključivo za formalni test značajnosti.'),
    p('U pripremi ovih testova, iz analize su izbačena dva šuta (0,05% od 3.968) zbog rezidualne nedostajuće vrednosti u atributu otvorenosti ugla, nastale u ekstremnom geometrijskom slučaju gde je ugao šuta jednak nuli. Ovo je odvojeno i znatno manje od izdvajanja penala (135 šuteva) opisanog u poglavlju 2, i odnosi se isključivo na Model B varijante.'),

    h2('4.4. Kalibracija i post-hoc korekcija'),
    caption('Tabela 4. Brier skor pre i posle post-hoc kalibracije (izotona regresija)'),
    makeTable(
      ['Model', 'Brier skor (sirovi model)', 'Brier skor (posle kalibracije)'],
      [
        ['Model A, logistička regresija', '0,155', '0,076'],
        ['Model A, XGBoost', '0,107', '0,076'],
        ['Model B, logistička regresija', '0,147', '0,074'],
        ['Model B, XGBoost', '0,121', '0,075'],
      ],
      [4500, 2430, 2430]
    ),
    p('Sirovi modeli, pre post-hoc kalibracije, pokazuju sistematsku prekalibrisanost u višem opsegu predviđenih verovatnoća. Primena izotone regresije kao post-hoc kalibracije značajno poboljšava Brier skor za sve četiri kombinacije modela (redukcije od 29% do 51% u zavisnosti od modela i validacione šeme).'),

    h2('4.5. Ablation analiza: doprinos pojedinačnih grupa 360 atributa'),
    p('Da bi se preciziralo koji deo prostorne informacije proizvodi poboljšanje, sprovedena je ablation analiza u kojoj su 360 atributi podeljeni u četiri grupe: golman (GK - udaljenost golmana), konusni branioci (CONE - branioci u konusu šuta i između šutera i gola), defanzivni pritisak (PRESSURE - pressure_score, broj branilaca u blizini), i linija šuta (SHOT_LINE - najbliži branilac liniji šuter-gol). Za svaku grupu treniran je Model A sa jednom dodatom grupom, i merena je LOTO AUC razlika u odnosu na čist Model A.'),
    caption('Tabela 5. Ablation analiza: inkrementalni doprinos svake grupe 360 atributa (logistička regresija, LOTO validacija)'),
    makeTable(
      ['Varijanta', 'LOTO ROC AUC', 'Delta vs Model A'],
      [
        ['Model A (bez 360)', '0,760', '-'],
        ['Model A + SHOT_LINE', '0,760', '+0,000'],
        ['Model A + GK', '0,764', '+0,004'],
        ['Model A + PRESSURE', '0,767', '+0,007'],
        ['Model A + CONE', '0,768', '+0,008'],
        ['Model B (sve 360 grupe)', '0,773', '+0,013'],
      ]
    ),
    p('Rezultati pokazuju da najveći deo inkrementalne vrednosti dolazi iz informacije o broju i rasporedu protivnika u konusu šuta (CONE, +0,008), zatim iz defanzivnog pritiska (PRESSURE, +0,007), dok pozicija golmana donosi manji, ali merljiv doprinos (+0,004). Udaljenost branioca od linije šuta (SHOT_LINE) ne donosi praktično nikakav inkrementalni doprinos kad se doda sama, bez ostalih 360 atributa. Zaključak je da nije ceo 360 sloj podjednako vredan: najveći deo poboljšanja dolazi iz geometrijske informacije o zagušenosti prostora između šutera i gola. Napomena: ROC AUC za pun Model B u Tabeli 5 (0,773) blago se razlikuje od vrednosti u Tabeli 3 (0,769) jer ablation analiza koristi logističku regresiju bez hiperparametarskog podešavanja (GridSearchCV), za razliku od Tabele 3 gde je model optimizovan unutar svakog folda.'),

    h2('4.6. Interpretacija modela: Odds Ratio i SHAP'),
    p('Logistička regresija na Modelu B pokazuje statistički značajan i fudbalski smislen Odds Ratio za ključne atribute: veća udaljenost od gola smanjuje šansu za gol, veći ugao šuta povećava šansu, šut glavom ima značajno niži Odds Ratio u odnosu na šut nogom, a veći udeo otvorenog (nezaklonjenog) ugla ka golu značajno povećava šansu (Odds Ratio približno 3,1; p < 0,001), što je u skladu sa nalazima iz literature o nižoj stopi konverzije udaraca glavom u odnosu na udarce nogom.'),
    p('SHAP analiza na Modelu B (XGBoost) pokazuje da je ugao šuta najuticajniji prediktor u fitovanom modelu, neposredno praćen otvorenošću ugla ka golu, izvedenom iz StatsBomb 360 podataka. Ovo je konkretan i merljiv nalaz o prediktivnoj asocijaciji prostornih atributa u modelu, ne o uzročnom efektu promene tih atributa na ishod.'),
    p('Zanimljiv je nesklad između dva pristupa interpretaciji za atribut najbliže udaljenosti branioca od linije šuta: u logističkoj regresiji ovaj atribut nije statistički značajan (p = 0,645), dok se u SHAP analizi XGBoost modela plasira na petom mestu po uticaju, odmah iza udaljenosti golmana. Ovaj nesklad sugeriše da efekat ovog atributa nije linearan, odnosno da ga logistička regresija ne uspeva u potpunosti da uhvati, dok XGBoost prirodno modeluje takav nelinearni ili interakcijski efekat. Ovo je dodatni argument u prilog korišćenju oba modela paralelno, jer se njihove interpretacije međusobno nadopunjuju.'),
  ];
}

module.exports = { buildPart4 };
