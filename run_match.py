"""
Portugal vs Uzbekistan — 2026 FIFA World Cup, June 22
Prediction method: Dixon-Coles-style Poisson model on available API data.
Free API tier gives 6 Portugal matches (EC 2024 + WC 2026) and 1 Uzbekistan match.
"""
import os, json, time as _time, math, itertools, warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import poisson

# ── config ────────────────────────────────────────────────────────────────────
API_TOKEN   = os.environ["FOOTBALL_DATA_API_TOKEN"]
TEAM_A_NAME = "Portugal"
TEAM_A_ID   = 765
TEAM_B_NAME = "Uzbekistan"
TEAM_B_ID   = 8070
MATCH_DATE  = "2026-06-22"

BASE_URL = "https://api.football-data.org/v4"
HEADERS  = {"X-Auth-Token": API_TOKEN}

COMP_WEIGHTS     = {"WC": 3.0, "EC": 2.5, "CAN": 2.5, "UCL": 2.0,
                    "UNL": 1.8, "WCQ": 1.5, "FR": 0.6}
DEFAULT_CW       = 1.0
DECAY_HALF_LIFE  = 365   # days

TEAM_COLORS = {"Portugal": "#006600", "Uzbekistan": "#009900"}
COL_A = "#006600"; COL_B = "#1565C0"   # Portugal green, Uzbekistan blue

# ── API helpers ───────────────────────────────────────────────────────────────
def api_get(url, params=None, retries=4):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=15)
            r.raise_for_status()
            return r
        except requests.exceptions.HTTPError:
            if r.status_code in (429, 503) and attempt < retries - 1:
                _time.sleep(2 ** attempt)
            else:
                raise
        except requests.exceptions.RequestException:
            if attempt < retries - 1:
                _time.sleep(2 ** attempt)
            else:
                raise

# ── fetch in ≤700-day windows (free tier limit: 750 days) ────────────────────
def fetch_team_matches(team_id, name):
    cache_file = f".cache_{name.replace(' ','_')}_matches.json"
    if os.path.exists(cache_file):
        age_h = (_time.time() - os.path.getmtime(cache_file)) / 3600
        if age_h < 24:
            with open(cache_file) as f:
                data = json.load(f)
            print(f"  {name}: {len(data)} matches from cache ({age_h:.1f}h old)")
            return data

    all_matches = []
    # Free tier only allows access within recent competitions (EC 2024, WC 2026).
    # Start from 2024-01-01 to stay within accessible data and avoid 403s on older windows.
    start = pd.Timestamp("2024-01-01")
    end   = pd.Timestamp(MATCH_DATE)

    while start < end:
        window_end = min(start + pd.Timedelta(days=700), end)
        try:
            r = api_get(f"{BASE_URL}/teams/{team_id}/matches", params={
                "dateFrom": start.strftime("%Y-%m-%d"),
                "dateTo":   window_end.strftime("%Y-%m-%d"),
                "status":   "FINISHED",
                "limit":    100,
            })
            chunk = r.json().get("matches", [])
            all_matches.extend(chunk)
            print(f"  {name}: {start.date()} → {window_end.date()}: {len(chunk)} matches")
        except Exception as e:
            print(f"  {name}: window {start.date()}→{window_end.date()} failed ({e})")
        start = window_end + pd.Timedelta(days=1)

    # deduplicate
    seen, unique = set(), []
    for m in all_matches:
        if m["id"] not in seen:
            seen.add(m["id"]); unique.append(m)

    with open(cache_file, "w") as f:
        json.dump(unique, f)
    print(f"  {name}: {len(unique)} total matches cached")
    return unique

# ── parse ─────────────────────────────────────────────────────────────────────
def parse(matches, team_id):
    rows = []
    for m in matches:
        ft = m.get("score", {}).get("fullTime", {})
        if ft.get("home") is None:
            continue
        is_home = m["homeTeam"]["id"] == team_id
        gf = ft["home"] if is_home else ft["away"]
        ga = ft["away"] if is_home else ft["home"]
        comp_code = m.get("competition", {}).get("code", "UNK")
        comp_w    = COMP_WEIGHTS.get(comp_code, DEFAULT_CW)
        date      = pd.to_datetime(m["utcDate"][:10])
        days_back = (pd.Timestamp(MATCH_DATE) - date).days
        time_w    = 2 ** (-days_back / DECAY_HALF_LIFE)
        rows.append({
            "date": date, "comp": comp_code,
            "opponent": m["awayTeam"]["name"] if is_home else m["homeTeam"]["name"],
            "is_home": int(is_home),
            "gf": gf, "ga": ga, "gd": gf - ga,
            "win":  int(gf > ga), "draw": int(gf == ga), "loss": int(gf < ga),
            "points": 3 if gf > ga else (1 if gf == ga else 0),
            "weight": comp_w * time_w,
        })
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

# ── fetch + parse ─────────────────────────────────────────────────────────────
print(f"\nFetching {TEAM_A_NAME} (ID {TEAM_A_ID}) ...")
raw_a = fetch_team_matches(TEAM_A_ID, TEAM_A_NAME)
print(f"\nFetching {TEAM_B_NAME} (ID {TEAM_B_ID}) ...")
raw_b = fetch_team_matches(TEAM_B_ID, TEAM_B_NAME)

df_a = parse(raw_a, TEAM_A_ID)
df_b = parse(raw_b, TEAM_B_ID)

print(f"\n{TEAM_A_NAME}: {len(df_a)} finished matches available")
if len(df_a):
    for _, r in df_a.iterrows():
        res = "W" if r.win else ("D" if r.draw else "L")
        print(f"  {r.date.date()}  [{r.comp}]  {res} {r.gf}-{r.ga}  vs {r.opponent}")

print(f"\n{TEAM_B_NAME}: {len(df_b)} finished matches available")
if len(df_b):
    for _, r in df_b.iterrows():
        res = "W" if r.win else ("D" if r.draw else "L")
        print(f"  {r.date.date()}  [{r.comp}]  {res} {r.gf}-{r.ga}  vs {r.opponent}")

# ── weighted aggregate stats ──────────────────────────────────────────────────
def wstat(df, col):
    if len(df) == 0:
        return 0.0
    w = df["weight"].values
    return np.dot(df[col].values, w) / max(w.sum(), 1e-9)

def team_stats(df):
    return {
        "n":          len(df),
        "win_rate":   wstat(df, "win"),
        "draw_rate":  wstat(df, "draw"),
        "loss_rate":  wstat(df, "loss"),
        "gf_per_g":   wstat(df, "gf"),
        "ga_per_g":   wstat(df, "ga"),
        "gd_per_g":   wstat(df, "gd"),
        "pts_per_g":  wstat(df, "points"),
    }

sa = team_stats(df_a)
sb = team_stats(df_b)

print(f"\n{'Stat':<15} {TEAM_A_NAME:>12}  {TEAM_B_NAME:>12}")
print("-" * 42)
for k in ["n", "win_rate", "draw_rate", "loss_rate", "gf_per_g", "ga_per_g", "gd_per_g", "pts_per_g"]:
    a_val = f"{sa[k]:.3f}" if k != "n" else str(int(sa[k]))
    b_val = f"{sb[k]:.3f}" if k != "n" else str(int(sb[k]))
    print(f"  {k:<13} {a_val:>12}  {b_val:>12}")

# ── Poisson match prediction ──────────────────────────────────────────────────
# Expected goals:
#   λ_A = (A's weighted GF/g + B's weighted GA/g) / 2
#   λ_B = (B's weighted GF/g + A's weighted GA/g) / 2
# Falls back to global averages when data is very sparse.
GLOBAL_INTL_GF  = 1.35   # avg goals scored per team per intl match (rough prior)
GLOBAL_INTL_GA  = 1.35

def blend(val, prior, n, confidence_n=5):
    """Blend observed value towards prior when n < confidence_n."""
    alpha = min(n / confidence_n, 1.0)
    return alpha * val + (1 - alpha) * prior

gf_a = blend(sa["gf_per_g"], GLOBAL_INTL_GF, sa["n"])
ga_a = blend(sa["ga_per_g"], GLOBAL_INTL_GA, sa["n"])
gf_b = blend(sb["gf_per_g"], GLOBAL_INTL_GF, sb["n"])
ga_b = blend(sb["ga_per_g"], GLOBAL_INTL_GA, sb["n"])

lambda_a = (gf_a + ga_b) / 2
lambda_b = (gf_b + ga_a) / 2

print(f"\nPoisson λ  {TEAM_A_NAME}: {lambda_a:.3f}   {TEAM_B_NAME}: {lambda_b:.3f}")

# Compute W/D/L probabilities over goals 0..9
MAX_G = 9
p_a_win = p_draw = p_b_win = 0.0
for ga in range(MAX_G + 1):
    for gb in range(MAX_G + 1):
        p = poisson.pmf(ga, lambda_a) * poisson.pmf(gb, lambda_b)
        if ga > gb:
            p_a_win += p
        elif ga == gb:
            p_draw  += p
        else:
            p_b_win += p

# Normalise to 1 (covers ~99.99% of mass already)
total = p_a_win + p_draw + p_b_win
p_a_win /= total; p_draw /= total; p_b_win /= total

print("\n" + "=" * 55)
print(f"  {TEAM_A_NAME} vs {TEAM_B_NAME}")
print(f"  {MATCH_DATE}  |  FIFA World Cup 2026  |  Poisson model")
print("=" * 55)
print(f"  {TEAM_A_NAME:<22} win  : {p_a_win:>6.1%}")
print(f"  {'Draw':<22}      : {p_draw:>6.1%}")
print(f"  {TEAM_B_NAME:<22} win  : {p_b_win:>6.1%}")
print("=" * 55)
print(f"  Expected goals:  {TEAM_A_NAME} {lambda_a:.2f}  –  {lambda_b:.2f} {TEAM_B_NAME}")
print("\nImplied decimal odds:")
for label, p in [(f"{TEAM_A_NAME} win", p_a_win), ("Draw", p_draw), (f"{TEAM_B_NAME} win", p_b_win)]:
    odds = round(1 / p, 2) if p > 0.001 else "∞"
    print(f"  {label:<22} : {odds}")

data_note = (
    f"NOTE: Free API tier — {sa['n']} {TEAM_A_NAME} matches, "
    f"{sb['n']} {TEAM_B_NAME} match(es) available."
)
print(f"\n  ⚠  {data_note}")

# ── score probability heatmap + result donut ──────────────────────────────────
SHOW_G = 6
score_grid = np.zeros((SHOW_G, SHOW_G))
for ga in range(SHOW_G):
    for gb in range(SHOW_G):
        score_grid[ga, gb] = poisson.pmf(ga, lambda_a) * poisson.pmf(gb, lambda_b)

fig = plt.figure(figsize=(15, 10))
fig.suptitle(
    f"{TEAM_A_NAME} vs {TEAM_B_NAME}  |  {MATCH_DATE}  |  FIFA World Cup 2026",
    fontsize=14, fontweight="bold",
)
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.42)

# 1. Result probability donut
ax1 = fig.add_subplot(gs[0, 0])
wedges, _ = ax1.pie(
    [p_a_win, p_draw, p_b_win],
    colors=[COL_A, "#AAAAAA", COL_B],
    startangle=90,
    wedgeprops={"width": 0.55, "edgecolor": "white"},
)
ax1.legend(
    wedges,
    [f"{TEAM_A_NAME}\n{p_a_win:.1%}", f"Draw\n{p_draw:.1%}", f"{TEAM_B_NAME}\n{p_b_win:.1%}"],
    loc="lower center", bbox_to_anchor=(0.5, -0.18), fontsize=8,
)
ax1.set_title("Win / Draw / Loss Probability\n(Poisson model)")

# 2. Score probability heatmap
ax2 = fig.add_subplot(gs[0, 1:])
im = ax2.imshow(score_grid * 100, cmap="YlOrRd", aspect="auto")
ax2.set_xticks(range(SHOW_G)); ax2.set_xticklabels(range(SHOW_G))
ax2.set_yticks(range(SHOW_G)); ax2.set_yticklabels(range(SHOW_G))
ax2.set_xlabel(f"{TEAM_B_NAME} goals")
ax2.set_ylabel(f"{TEAM_A_NAME} goals")
ax2.set_title("Scoreline Probability Heatmap (%)")
for i in range(SHOW_G):
    for j in range(SHOW_G):
        val = score_grid[i, j] * 100
        ax2.text(j, i, f"{val:.1f}", ha="center", va="center",
                 fontsize=7, color="black" if val < 6 else "white")
plt.colorbar(im, ax=ax2, fraction=0.03)

# 3. Aggregate stats bar chart
ax3 = fig.add_subplot(gs[1, 0])
cats = ["GF/g", "GA/g", "Win%", "Draw%", "Pts/g"]
a_vals = [sa["gf_per_g"], sa["ga_per_g"], sa["win_rate"], sa["draw_rate"], sa["pts_per_g"]]
b_vals = [sb["gf_per_g"], sb["ga_per_g"], sb["win_rate"], sb["draw_rate"], sb["pts_per_g"]]
x = np.arange(len(cats))
ax3.bar(x - 0.2, a_vals, 0.4, label=TEAM_A_NAME, color=COL_A, alpha=0.85)
ax3.bar(x + 0.2, b_vals, 0.4, label=TEAM_B_NAME, color=COL_B, alpha=0.85)
ax3.set_xticks(x); ax3.set_xticklabels(cats, rotation=25, ha="right")
ax3.set_title(f"Weighted Stats\n({sa['n']} {TEAM_A_NAME} / {sb['n']} {TEAM_B_NAME} matches)")
ax3.legend(fontsize=8); ax3.spines[["top", "right"]].set_visible(False)

# 4. Portugal match timeline
ax4 = fig.add_subplot(gs[1, 1:])
if len(df_a) > 0:
    colors_a = [COL_A if w else ("#AAAAAA" if d else "#CC0000")
                for w, d in zip(df_a["win"], df_a["draw"])]
    ax4.bar(range(len(df_a)), df_a["gf"], color=colors_a, alpha=0.85, label="GF")
    ax4.bar(range(len(df_a)), -df_a["ga"], color=["#004400" if w else ("#888888" if d else "#880000")
            for w, d in zip(df_a["win"], df_a["draw"])], alpha=0.55, label="GA (negative)")
    ax4.axhline(0, color="black", lw=0.8)
    ax4.set_xticks(range(len(df_a)))
    ax4.set_xticklabels(
        [f"{r.date.strftime('%b %d')}\n{r.comp}" for _, r in df_a.iterrows()],
        rotation=30, ha="right", fontsize=8,
    )
    ax4.set_ylabel("Goals"); ax4.set_title(f"{TEAM_A_NAME} Recent Results (green=W, grey=D, red=L)")
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.legend(fontsize=8)

# ── Uzbekistan bar (if more than 1 match, show on same chart) ─────────────────
fig.text(
    0.5, 0.01,
    data_note + "  |  Blended with global international average where data is sparse.",
    ha="center", fontsize=8, color="gray",
)

out = f"{TEAM_A_NAME.replace(' ','_')}_vs_{TEAM_B_NAME.replace(' ','_')}_{MATCH_DATE}.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print(f"\nChart saved → {out}")
