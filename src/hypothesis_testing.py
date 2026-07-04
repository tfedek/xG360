"""
hypothesis_testing.py
========================
Korak 2 (inferencijalna statistička analiza) metodologije - testovi
vezani za KONKRETNA istraživačka pitanja (ne "paket" testova bez svrhe):

  H1: Da li je tip akcije (play_pattern) povezan sa verovatnošću gola?
      -> Chi-square test nezavisnosti
  H2: Da li se udaljenost od gola razlikuje između golova i promašaja?
      -> Mann-Whitney U (neparametarski, distribucija udaljenosti nije
         normalna - desno zakošena, potvrđeno u EDA koraku)
  H3: Da li se ugao šuta razlikuje između golova i promašaja?
      -> Mann-Whitney U
  H4: Da li se stopa golova razlikuje između tri turnira?
      -> Chi-square test nezavisnosti (turnir x ishod)

Dodatno (vezano za korak 6/9 - generalizacija kroz turnire):
  KS test: da li se RASPODELA ključnih numeričkih feature-a (distance,
  angle) razlikuje između turnira - formalna provera distribucijskog
  pomaka, objašnjava zašto LOTO performans varira (npr. EURO2024 ima
  primetno nižu stopu golova - da li je to praćeno i drugačijom
  raspodelom geometrije šuta, ili je razlika samo u finalizaciji?).

Pokretanje: python3 hypothesis_testing.py
"""

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
TABLES_DIR = Path(__file__).resolve().parent.parent / "tables"


def load_modeling_set() -> pd.DataFrame:
    """Učitava shots_raw.csv i filtrira na isti modeling set kao Model A/B
    (bez penala) - koristi se sirovi shots_raw.csv jer sadrži i izvorni
    (necodean) play_pattern, pogodniji za čitljivu kontingencijsku tabelu."""
    df = pd.read_csv(PROCESSED_DIR / "shots_raw.csv")
    return df[df["shot_type"] != "Penalty"].copy()


def h1_play_pattern_vs_goal(df: pd.DataFrame) -> dict:
    """H1: tip akcije (play_pattern) i ishod (gol/nije gol) - Chi-square."""
    contingency = pd.crosstab(df["play_pattern"], df["is_goal"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    # Cochran-ovo pravilo: udeo ćelija sa expected < 5
    low_expected_pct = (expected < 5).mean() * 100
    test_used = "Chi-square"
    if low_expected_pct > 20:
        test_used = ("Chi-square (NAPOMENA: >20% ćelija ima expected<5 - "
                      "rezultat treba tumačiti uz oprez, razmotriti Fisher/agregaciju)")

    result = {
        "hypothesis": "H1: play_pattern povezan sa is_goal",
        "test": test_used, "chi2": round(chi2, 3), "dof": dof,
        "p_value": p, "n": len(df), "pct_cells_expected_lt_5": round(low_expected_pct, 1),
    }
    print("\n-- H1: Tip akcije (play_pattern) ↔ gol --")
    print(f"   χ²={chi2:.3f}, dof={dof}, p={p:.4g}, n={len(df)}")
    print(f"   Udeo ćelija sa expected<5: {low_expected_pct:.1f}%")
    print(f"   Zaključak: {'značajna asocijacija' if p < 0.05 else 'nema značajne asocijacije'} (α=0.05)")
    return result


def mann_whitney_by_outcome(df: pd.DataFrame, numeric_col: str, label: str) -> dict:
    """Mann-Whitney U test: numeric_col između golova (is_goal=1) i
    promašaja (is_goal=0). Neparametarski test biran zato što distribucije
    udaljenosti/ugla nisu normalne (desno zakošene - potvrđeno u EDA
    koraku kroz vizuelnu i deskriptivnu proveru asimetrije)."""
    goals = df.loc[df["is_goal"] == 1, numeric_col].dropna()
    misses = df.loc[df["is_goal"] == 0, numeric_col].dropna()

    u_stat, p = stats.mannwhitneyu(goals, misses, alternative="two-sided")

    result = {
        "hypothesis": label, "test": "Mann-Whitney U",
        "median_goal": round(goals.median(), 3), "median_miss": round(misses.median(), 3),
        "u_statistic": u_stat, "p_value": p,
        "n_goals": len(goals), "n_misses": len(misses),
    }
    print(f"\n-- {label} --")
    print(f"   Medijana (gol)={goals.median():.2f}, Medijana (promašaj)={misses.median():.2f}")
    print(f"   U={u_stat:.1f}, p={p:.4g}")
    print(f"   Zaključak: {'značajna razlika' if p < 0.05 else 'nema značajne razlike'} (α=0.05)")
    return result


def h4_goal_rate_by_tournament(df: pd.DataFrame) -> dict:
    """H4: da li se stopa golova razlikuje između turnira - Chi-square
    na kontingencijskoj tabeli turnir x ishod."""
    contingency = pd.crosstab(df["tournament"], df["is_goal"])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    print("\n-- H4: Stopa golova po turniru --")
    print(contingency)
    print(f"   χ²={chi2:.3f}, dof={dof}, p={p:.4g}")
    print(f"   Zaključak: {'značajna razlika među turnirima' if p < 0.05 else 'nema značajne razlike'} (α=0.05)")
    return {
        "hypothesis": "H4: stopa golova razlikuje se po turniru",
        "test": "Chi-square", "chi2": round(chi2, 3), "dof": dof, "p_value": p,
    }


def ks_distributional_shift(df: pd.DataFrame, numeric_col: str) -> pd.DataFrame:
    """
    Kolmogorov-Smirnov test - poredi raspodelu numeric_col između SVAKOG
    para turnira (3 turnira -> 3 para). Formalna provera distribucijskog
    pomaka koja podržava/objašnjava LOTO rezultate (korak 6/9): ako je
    raspodela geometrije šuta slična kroz turnire, model bi trebalo da
    generalizuje dobro (što LOTO rezultati i pokazuju - razlike su umerene).
    """
    tournaments = sorted(df["tournament"].unique())
    rows = []
    for t1, t2 in combinations(tournaments, 2):
        x1 = df.loc[df["tournament"] == t1, numeric_col].dropna()
        x2 = df.loc[df["tournament"] == t2, numeric_col].dropna()
        stat, p = stats.ks_2samp(x1, x2)
        rows.append({
            "variable": numeric_col, "tournament_1": t1, "tournament_2": t2,
            "ks_statistic": round(stat, 4), "p_value": round(p, 4),
            "distributions_differ": p < 0.05,
        })
    result_df = pd.DataFrame(rows)
    print(f"\n-- KS test distribucijskog pomaka: {numeric_col} --")
    print(result_df.to_string(index=False))
    return result_df


if __name__ == "__main__":
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_modeling_set()

    print("=" * 65)
    print(f"TESTIRANJE HIPOTEZA (n={len(df)}, bez penala)")
    print("=" * 65)

    results = []
    results.append(h1_play_pattern_vs_goal(df))
    results.append(mann_whitney_by_outcome(
        df, "distance_to_goal", "H2: Udaljenost od gola (gol vs. promašaj)"))
    results.append(mann_whitney_by_outcome(
        df, "shot_angle_deg", "H3: Ugao šuta (gol vs. promašaj)"))
    results.append(h4_goal_rate_by_tournament(df))

    pd.DataFrame(results).to_csv(TABLES_DIR / "hypothesis_tests_summary.csv", index=False)

    print("\n" + "=" * 65)
    print("KS TEST - DISTRIBUCIJSKI POMAK KROZ TURNIRE")
    print("=" * 65)

    ks_distance = ks_distributional_shift(df, "distance_to_goal")
    ks_angle = ks_distributional_shift(df, "shot_angle_deg")

    pd.concat([ks_distance, ks_angle], ignore_index=True).to_csv(
        TABLES_DIR / "ks_test_distributional_shift.csv", index=False
    )
