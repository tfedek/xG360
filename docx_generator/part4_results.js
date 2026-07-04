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
    caption('Tabela 2. Rezultati Stratified K-Fold validacije (prosek kroz 5 foldova)'),
    makeTable(
      ['Model', 'ROC AUC', 'PR AUC', 'F1', 'Brier skor'],
      [
        ['Model A, logistička regresija', '0,773', '0,339', '0,326', '0,185'],
        ['Model A, XGBoost', '0,757', '0,318', '0,313', '0,184'],
        ['Model B, logistička regresija', '0,794', '0,362', '0,358', '0,175'],
        ['Model B, XGBoost', '0,783', '0,350', '0,355', '0,159'],
      ],
      [3360, 1500, 1500, 1500, 1500]
    ),
    p('Model B nadmašuje Model A na svim metrikama, bez obzira na to koji algoritam je korišćen. Logistička regresija ostvaruje viši ROC AUC i PR AUC od XGBoost-a u oba slučaja, dok XGBoost ostvaruje niži, odnosno bolji Brier skor, što ukazuje da XGBoost ima blago slabiju diskriminaciju ali bolju kalibraciju u odnosu na logističku regresiju na ovom skupu podataka.'),

    h2('4.3. Generalizacija kroz turnire: Leave-One-Tournament-Out validacija'),
    caption('Tabela 3. Rezultati Leave-One-Tournament-Out validacije (prosek kroz tri turnira)'),
    makeTable(
      ['Model', 'ROC AUC', 'PR AUC', 'F1', 'Brier skor'],
      [
        ['Model A, logistička regresija', '0,772', '0,315', '0,324', '0,185'],
        ['Model A, XGBoost', '0,751', '0,297', '0,310', '0,178'],
        ['Model B, logistička regresija', '0,792', '0,330', '0,347', '0,173'],
        ['Model B, XGBoost', '0,780', '0,347', '0,356', '0,149'],
      ],
      [3360, 1500, 1500, 1500, 1500]
    ),
    p('Rezultati Leave-One-Tournament-Out validacije su gotovo identični rezultatima Stratified K-Fold validacije za sve četiri kombinacije modela i algoritma. Razlika u ROC AUC između dve validacione šeme ne prelazi 0,002 za bilo koju kombinaciju. Ovo je značajan nalaz: model ne pokazuje znake prenaučenosti specifičnostima jednog turnira, već generalizuje na turnir koji nikada nije vidio tokom treninga skoro identično dobro kao na podacima iz iste distribucije.'),
    p('Pri detaljnijem pregledu po pojedinačnom turniru, Evropsko prvenstvo 2024 dosledno pokazuje nešto niži PR AUC kao izostavljeni turnir u odnosu na druga dva, što je u skladu sa nalazom iz odeljka 4.1 da ovaj turnir ima nižu stopu konverzije šuteva u gol, uz statistički nepromenjenu geometriju šuta.'),

    h2('4.3a. Formalno testiranje statističke značajnosti razlike Model A i Model B'),
    p('Dosledna prednost Modela B kroz sve metrike i obe validacione šeme jeste snažan indirektan pokazatelj, ali ne predstavlja sama po sebi formalni statistički dokaz da razlika nije slučajna, odnosno posledica varijanse uzorka. Da bi se ova razlika formalno testirala, sprovedena su dva nezavisna testa, prilagođena svakom od dva algoritma.'),
    p('Za logističku regresiju, Model A je formalno ugnježden u Model B: svih 24 atributa Modela A predstavljaju tačan podskup 31 atributa Modela B, pri čemu su preostala 7 atributa StatsBomb 360 prostorni atributi (broj branilaca u liniji šuta, broj saigrača u istoj zoni, broj protivnika u krugu od 5 metara, otvorenost ugla ka golu, najbliža udaljenost branioca od linije šuta, defanzivni pritisak i anomalija pozicije golmana). Ugnježdena struktura modela dozvoljava primenu Likelihood Ratio (LR) testa, koji poredi logaritme verodostojnosti (log-likelihood) dva modela:'),
    F.eqLikelihoodRatio(),
    p('gde su ℓ_A i ℓ_B logaritmi verodostojnosti Modela A i Modela B, a LR statistika prati hi-kvadrat raspodelu sa brojem stepeni slobode jednakim broju dodatnih parametara u Modelu B. Oba modela su fitovana na identičnom skupu od 3.966 šuteva (zajednički podskup, nakon uklanjanja redova s nedostajućim vrednostima opisanih u nastavku), kako bi poređenje log-likelihood vrednosti bilo validno. Rezultat je LR = 70,03, sa 7 stepeni slobode, što odgovara p-vrednosti od približno 1,46×10⁻¹², odlučno odbacujući nultu hipotezu da StatsBomb 360 atributi ne doprinose modelu. Ovaj nalaz je dodatno potkrepljen i informacionim kriterijumima: Akaikeov informacioni kriterijum (AIC) iznosi 2.088,75 za Model A i 2.032,72 za Model B, a Bajesov informacioni kriterijum (BIC) 2.245,89 naspram 2.233,86, oba favorizujući Model B uprkos kazni za veći broj parametara.'),
    p('Za XGBoost, modeli nisu ugnježdeni na isti parametarski način, pa Likelihood Ratio test nije primenjiv. Umesto toga, sprovedena je bootstrap analiza razlike ROC AUC na test skupu jednog reprezentativnog 80/20 train-test splita (isti protokol kao u koraku evaluacije diskriminacije i kalibracije). Test skup je resempliran sa ponavljanjem 2.000 puta, pri čemu su za svaki uzorak istovremeno (parno) preuzete predikcije oba modela na istim šutevima, da bi se očuvala zavisnost između parova predikcija. Opažena razlika ROC AUC (Model B minus Model A) iznosi 0,037, a 95% interval poverenja dobijen iz bootstrap raspodele iznosi [0,0068, 0,0675]. Pošto interval ne sadrži nulu, razlika se smatra statistički značajnom, sa približnom dvostranom p-vrednošću od 0,014.'),
    p('Vredno je naglasiti da se bootstrap analiza odnosi na jedan train-test split, a ne na Leave-One-Tournament-Out validaciju opisanu u prethodnom odeljku; sprovođenje bootstrap testiranja unutar svakog LOTO folda bi zahtevalo hiljade ponovnih treniranja XGBoost modela sa unutrašnjim podešavanjem hiperparametara, što je računarski znatno zahtevnije, a ovaj pristup već pruža teorijski utemeljenu i robusnu procenu značajnosti razlike na fiksnom test skupu, bez rizika od curenja podataka. Niža, mada i dalje značajna, p-vrednost bootstrap testa u odnosu na izrazito malu p-vrednost LR testa proizlazi iz prirode samih testova: LR test procenjuje promenu log-verodostojnosti na celom skupu podataka, dok bootstrap procenjuje varijabilnost jedne metrike (ROC AUC) na znatno manjem, izdvojenom test skupu od 794 šuta, što unosi veću varijansu u procenu.'),
    p('U pripremi ovih testova, iz analize su izbačena dva šuta (0,05% od 3.968) zbog rezidualne nedostajuće vrednosti u atributu otvorenosti ugla, nastale u ekstremnom geometrijskom slučaju gde je ugao šuta jednak nuli. Ovo je odvojeno i znatno manje od izdvajanja penala (135 šuteva) opisanog u poglavlju 2, i odnosi se isključivo na Model B varijante.'),

    h2('4.4. Kalibracija i post-hoc korekcija'),
    caption('Tabela 4. Brier skor pre i posle post-hoc kalibracije (izotona regresija)'),
    makeTable(
      ['Model', 'Brier skor (sirovi model)', 'Brier skor (posle kalibracije)'],
      [
        ['Model A, logistička regresija', '0,173', '0,069'],
        ['Model A, XGBoost', '0,170', '0,070'],
        ['Model B, logistička regresija', '0,183', '0,071'],
        ['Model B, XGBoost', '0,175', '0,070'],
      ],
      [4500, 2430, 2430]
    ),
    p('Sirovi modeli, pre post-hoc kalibracije, pokazuju sistematsku prekalibrisanost u višem opsegu predviđenih verovatnoća: model predviđa visoke verovatnoće gola (na primer, preko 0,6) za situacije koje se u stvarnosti dešavaju znatno reže. Ovo je direktna posledica ponderisanja klasa primenjenog radi poboljšanja diskriminacije manjinske klase. Primena izotone regresije kao post-hoc kalibracije drastično poboljšava Brier skor za sve četiri kombinacije modela, sa redukcijom od preko 50%.'),

    h2('4.5. Interpretacija modela: Odds Ratio i SHAP'),
    p('Logistička regresija na Modelu B pokazuje statistički značajan i fudbalski smislen Odds Ratio za ključne atribute: veća udaljenost od gola smanjuje šansu za gol, veći ugao šuta povećava šansu, šut glavom ima značajno niži Odds Ratio u odnosu na šut nogom, a veći udeo otvorenog (nezaklonjenog) ugla ka golu značajno povećava šansu (Odds Ratio približno 3,1; p < 0,001), što je u skladu sa nalazima iz literature o nižoj stopi konverzije udaraca glavom u odnosu na udarce nogom.'),
    p('SHAP analiza na Modelu B (XGBoost) pokazuje da je ugao šuta najuticajniji atribut, neposredno praćen otvorenošću ugla ka golu, izvedenom iz StatsBomb 360 podataka, koja se po prosečnoj apsolutnoj SHAP vrednosti plasira odmah iza ugla šuta i ispred udaljenosti od gola. Anomalija pozicije golmana, najbliža udaljenost branioca od linije šuta i defanzivni pritisak takođe se nalaze u prvih šest najuticajnijih atributa, ispred broja branilaca u liniji šuta. Ovo je konkretan i merljiv dokaz da prostorni atributi, a ne samo klasična geometrija šuta, nose suštinsku prediktivnu informaciju u modelu.'),
    p('Zanimljiv je nesklad između dva pristupa interpretaciji za atribut najbliže udaljenosti branioca od linije šuta: u logističkoj regresiji ovaj atribut nije statistički značajan (p = 0,645), dok se u SHAP analizi XGBoost modela plasira na petom mestu po uticaju, odmah iza anomalije pozicije golmana. Ovaj nesklad sugeriše da efekat ovog atributa na verovatnoću gola nije linearan, odnosno da ga logistička regresija, koja pretpostavlja linearan odnos sa logitom, ne uspeva u potpunosti da uhvati, dok XGBoost, kroz stabla odluke, prirodno modeluje takav nelinearni ili interakcijski efekat, na primer da udaljenost branioca od linije šuta ima značajan utisak samo u kombinaciji sa malom udaljenošću od gola ili velikim otvorenim uglom. Ovo je dodatni argument u prilog korišćenju oba modela paralelno, jer se njihove interpretacije međusobno nadopunjuju, ne samo potvrđuju.'),
  ];
}

module.exports = { buildPart4 };
