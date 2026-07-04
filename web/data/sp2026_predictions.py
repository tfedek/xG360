"""
SP2026 Round of 32 - Finalna predikciona analiza
Metode: Iterativni SoS rating, adaptivni shrinkage, Poisson projekcija,
        bootstrap CI, ansambl sa kladioničarskim kvotama
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

# --- 1. UČITAVANJE PODATAKA ---
matches = pd.read_csv('/Users/fedektom/Downloads/xg_analiza/web/data/sp2026_group_matches.csv')
fixtures = pd.read_csv('/Users/fedektom/Downloads/xg_analiza/web/data/sp2026_r32_fixtures.csv')

# Prosečan xG po meču na turniru
mu = matches[['home_xg', 'away_xg']].values.flatten().mean()
print(f"Prosečan xG po timu po meču: {mu:.3f}")

# --- 2. ITERATIVNI STRENGTH-OF-SCHEDULE RATING ---
def compute_sos_ratings(matches, mu, n_iter=50):
    """Iterativno računa napadačku/defanzivnu snagu korigovanu za jačinu protivnika."""
    teams = sorted(set(matches['home_team']) | set(matches['away_team']))
    attack = {t: 1.0 for t in teams}
    defense = {t: 1.0 for t in teams}

    for _ in range(n_iter):
        new_attack = {}
        new_defense = {}
        for t in teams:
            # Mečevi gde je tim igrao (home ili away)
            home_mask = matches['home_team'] == t
            away_mask = matches['away_team'] == t
            
            xg_for = []
            xg_against = []
            opp_def = []
            opp_att = []
            
            for _, row in matches[home_mask].iterrows():
                xg_for.append(row['home_xg'])
                xg_against.append(row['away_xg'])
                opp_def.append(defense[row['away_team']])
                opp_att.append(attack[row['away_team']])
            for _, row in matches[away_mask].iterrows():
                xg_for.append(row['away_xg'])
                xg_against.append(row['home_xg'])
                opp_def.append(defense[row['home_team']])
                opp_att.append(attack[row['home_team']])
            
            # Korigovani napad = prosečan xG_for / prosečna defanziva protivnika
            if opp_def:
                avg_opp_def = np.mean(opp_def)
                avg_opp_att = np.mean(opp_att)
                new_attack[t] = np.mean(xg_for) / max(avg_opp_def, 0.3)
                new_defense[t] = np.mean(xg_against) / max(avg_opp_att, 0.3)
            else:
                new_attack[t] = 1.0
                new_defense[t] = 1.0
        
        # Normalizacija da prosek bude 1.0
        mean_att = np.mean(list(new_attack.values()))
        mean_def = np.mean(list(new_defense.values()))
        attack = {t: v / mean_att for t, v in new_attack.items()}
        defense = {t: v / mean_def for t, v in new_defense.items()}
    
    return attack, defense

attack_raw, defense_raw = compute_sos_ratings(matches, mu)

# --- 3. ADAPTIVNI SHRINKAGE (po-timski, na osnovu varijanse) ---
def adaptive_shrinkage(matches, attack_raw, defense_raw, mu, k_base=3):
    """Timovi sa velikom varijansom xG dobijaju jači shrinkage."""
    teams = sorted(attack_raw.keys())
    attack_shrunk = {}
    defense_shrunk = {}
    
    for t in teams:
        home_mask = matches['home_team'] == t
        away_mask = matches['away_team'] == t
        
        xg_for = list(matches.loc[home_mask, 'home_xg']) + list(matches.loc[away_mask, 'away_xg'])
        xg_against = list(matches.loc[home_mask, 'away_xg']) + list(matches.loc[away_mask, 'home_xg'])
        
        n = len(xg_for)
        if n < 2:
            cv = 0
        else:
            cv = np.std(xg_for) / max(np.mean(xg_for), 0.3)
        
        # Adaptivni k: visok CV → veći k (jači shrinkage)
        k = k_base * (1 + cv)
        
        # Shrinkage formula: (n * raw + k * 1.0) / (n + k)
        attack_shrunk[t] = (n * attack_raw[t] + k * 1.0) / (n + k)
        defense_shrunk[t] = (n * defense_raw[t] + k * 1.0) / (n + k)
    
    return attack_shrunk, defense_shrunk

attack, defense = adaptive_shrinkage(matches, attack_raw, defense_raw, mu)

# --- 4. POISSON PROJEKCIJA ---
def project_match(t1, t2, attack, defense, mu):
    """Projektuje lambda za oba tima i računa matricu rezultata."""
    lam1 = attack[t1] * defense[t2] * mu
    lam2 = attack[t2] * defense[t1] * mu
    
    # Matrica verovatnoća rezultata (do 6 golova)
    max_goals = 7
    score_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            score_matrix[i, j] = poisson.pmf(i, lam1) * poisson.pmf(j, lam2)
    
    p_home = np.tril(score_matrix, -1).sum()
    p_draw = np.trace(score_matrix)
    p_away = np.triu(score_matrix, 1).sum()
    
    # Najverovatniji rezultat
    idx = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
    
    # Top 3 rezultata
    flat = score_matrix.flatten()
    top3_idx = flat.argsort()[-3:][::-1]
    top3 = []
    for fi in top3_idx:
        r, c = divmod(fi, max_goals)
        top3.append((r, c, flat[fi]))
    
    return {
        'lam1': lam1, 'lam2': lam2,
        'p_home': p_home, 'p_draw': p_draw, 'p_away': p_away,
        'mode_score': idx, 'top3': top3, 'score_matrix': score_matrix
    }

# --- 5. BOOTSTRAP CI ---
def bootstrap_ci(matches, fixtures_row, mu, n_boot=1000, k_base=3):
    """Bootstrap interval poverenja za P(home win).
    Fallback: timovi koji nestanu iz uzorka dobijaju att=1.0, def=1.0 (prosek turnira)."""
    p_homes = []
    t1 = fixtures_row['home_team']
    t2 = fixtures_row['away_team']
    for _ in range(n_boot):
        sample = matches.sample(n=len(matches), replace=True)
        try:
            att, defe = compute_sos_ratings(sample, mu, n_iter=10)
            att_s, def_s = adaptive_shrinkage(sample, att, defe, mu, k_base)
            # Fallback za timove koji nisu u uzorku - prosečna vrednost (1.0)
            att_s.setdefault(t1, 1.0)
            att_s.setdefault(t2, 1.0)
            def_s.setdefault(t1, 1.0)
            def_s.setdefault(t2, 1.0)
            res = project_match(t1, t2, att_s, def_s, mu)
            p_homes.append(res['p_home'])
        except:
            continue
    if len(p_homes) < 20:
        return None, None, None
    fallback_rate = 1 - len(p_homes) / n_boot  # koliko iteracija je palo u except
    return np.percentile(p_homes, 2.5), np.percentile(p_homes, 97.5), fallback_rate

# --- 6. ANSAMBЛ SA KVOTAMA ---
def ensemble_prediction(poisson_result, market_home, market_draw, market_away, w=0.5):
    """50/50 ansambl Poisson modela i tržišnih kvota (w=0.5 validiran)."""
    p_h = w * poisson_result['p_home'] + (1 - w) * market_home
    p_d = w * poisson_result['p_draw'] + (1 - w) * market_draw
    p_a = w * poisson_result['p_away'] + (1 - w) * market_away
    # Renormalizacija
    total = p_h + p_d + p_a
    return p_h / total, p_d / total, p_a / total

# --- 7. POKRETANJE ANALIZE ---
print("\n" + "=" * 80)
print("SP2026 ROUND OF 32 - FINALNE PREDIKCIJE")
print("Metode: SoS iterativni rating + adaptivni shrinkage + Poisson + Bootstrap CI + Ansambl")
print("=" * 80)

results = []
for _, fix in fixtures.iterrows():
    t1, t2 = fix['home_team'], fix['away_team']
    
    # Poisson projekcija
    proj = project_match(t1, t2, attack, defense, mu)
    
    # Ansambl sa kvotama
    ens_h, ens_d, ens_a = ensemble_prediction(
        proj, fix['market_home'], fix['market_draw'], fix['market_away']
    )
    
    # Rezultat iz ansambla (preračunaj lambda da odgovara ansamblu)
    # Za skor koristimo Poisson projekciju (kvote ne daju skor)
    top3_str = ", ".join([f"{s[0]}-{s[1]}({s[2]*100:.0f}%)" for s in proj['top3']])
    
    results.append({
        'date': fix['date'],
        'match': f"{t1} - {t2}",
        'proj_xg': f"{proj['lam1']:.2f} - {proj['lam2']:.2f}",
        'mode': f"{proj['mode_score'][0]}-{proj['mode_score'][1]}",
        'top3': top3_str,
        'ens_1x2': f"{ens_h*100:.0f}/{ens_d*100:.0f}/{ens_a*100:.0f}",
        'p_home_model': proj['p_home'],
        'p_home_ens': ens_h,
    })

# Prikaz
print(f"\n{'Datum':<12}{'Meč':<30}{'Proj.xG':<14}{'Rezultat':<10}{'Top 3':<35}{'1X2 (ansambl)'}")
print("-" * 115)
for r in results:
    print(f"{r['date']:<12}{r['match']:<30}{r['proj_xg']:<14}{r['mode']:<10}{r['top3']:<35}{r['ens_1x2']}")

# --- 8. BOOTSTRAP CI (za odabrane mečeve, smanjeni n_boot za brzinu) ---
print("\n\n--- BOOTSTRAP 95% CI za P(pobeda prvog tima) ---")
print(f"{'Meč':<30}{'P(pobeda) ansambl':<20}{'95% CI (model)':<25}{'Fallback %'}")
print("-" * 85)
key_matches = [0, 1, 4, 9, 13]  # Brazil-Japan, Ger-Par, Fra-Swe, Spa-Aut, Arg-CaboV
for idx in key_matches:
    fix_row = fixtures.iloc[idx]
    lo, hi, fb = bootstrap_ci(matches, fix_row, mu, n_boot=1000)
    ens_h = results[idx]['p_home_ens']
    if lo is not None:
        print(f"{results[idx]['match']:<30}{ens_h*100:.0f}%{'':<15}[{lo*100:.0f}% - {hi*100:.0f}%]{'':<5}{fb*100:.1f}%")
    else:
        print(f"{results[idx]['match']:<30}{ens_h*100:.0f}%{'':<15}[nedovoljno uzoraka]")

# --- 9. TIMSKI REJTINZI (top 15 po napadačkoj snazi) ---
print("\n\n--- TIMSKI REJTINZI (SoS + adaptivni shrinkage) ---")
print(f"{'Tim':<25}{'Napad':<10}{'Defanziva':<12}{'Rating (att-def)'}")
print("-" * 55)
rating_list = [(t, attack[t], defense[t], attack[t] - defense[t]) for t in sorted(attack.keys())]
rating_list.sort(key=lambda x: -x[3])
for t, a, d, r in rating_list[:20]:
    print(f"{t:<25}{a:<10.3f}{d:<12.3f}{r:+.3f}")

print("\n\nGotovo.")
