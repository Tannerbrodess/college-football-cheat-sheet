#!/usr/bin/env python3
"""
Sooner Stats — Weekly Refresh
Rebuilds index.html from the CSVs in your project directory.

USAGE:
    python refresh.py                    # uses defaults (project dir + template)
    python refresh.py --project ./data   # custom project dir
    python refresh.py --template ./tmpl.html --out ./index.html

Auto-detects preseason vs in-season based on whether 2026 games have been played.
- Preseason mode: uses 2025 ratings + adjustments (achievement mean-reversion, QB delta, etc.)
- In-season mode: uses current-year 2026 ratings, drops all preseason adjustments

Colab-friendly. Requires: pandas, numpy, scikit-learn, scipy.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG — tweak these if you want to adjust model behavior
# ============================================================
CFG = {
    'projection_season': 2026,
    'base_season': 2025,  # only used in preseason mode

    # Composite weights (CORE at half strength)
    'composite_weights': {'sp': 1.0, 'fpi': 1.0, 'elo': 1.0, 'srs': 1.0, 'core': 0.5},

    # Preseason adjustments (dropped in-season)
    'preseason_regress': 0.75,          # regress 25% toward mean
    'achievement_coef': -0.11,          # half of fitted mean-reversion (-0.22)
    'coach_change_penalty': -1.5,       # points for new HC (Nov/Dec hire)
    'qb_delta_scale': 40.0,             # 40 PPA per SP+ point
    'qb_delta_cap': 2.5,

    # In-season adjustments (kept)
    'inseason_regress': 0.95,           # much lighter regression once games happen

    # Calibrated model params
    'default_scale': 11.544,
    'default_league_hfa': 4.47,

    # Monte Carlo
    'n_sims': 10000,
    'n_leverage_sims': 5000,

    # FCS grades
    'fcs_tier1_rating': -10,
    'fcs_tier2_rating': -22,
    'fcs_tier1_names': {
        'North Dakota State','South Dakota State','Montana','Montana State','Sacramento State',
        'Delaware','Villanova','William & Mary','New Hampshire','James Madison','Idaho',
        'Weber State','UC Davis','Furman','Chattanooga','Illinois State','Northern Iowa',
        'North Dakota','South Dakota','Missouri State','Sam Houston','Incarnate Word',
        'Southern Illinois','Eastern Kentucky','Rhode Island','Fordham','Lehigh',
        'Norfolk State','Hampton'
    },
}


def log(msg):
    print(f"[refresh] {msg}", flush=True)


# ============================================================
# LOAD CSVs
# ============================================================
def load_csvs(project_dir):
    """Load all needed CSVs from project directory. Missing files gracefully skipped."""
    p = Path(project_dir)
    required = ['ratings_sp.csv','ratings_fpi.csv','ratings_elo.csv','ratings_srs.csv',
                'core.csv','games.csv','clean_advanced.csv','cfb_2026_schedule.csv',
                'team_records.csv','teams_ats.csv','coaches.csv','rosters.csv',
                'player_ppa_season.csv','player_usage.csv']
    optional = ['team_ppa_season.csv','clean_metrics.csv','rankings.csv']

    missing = [f for f in required if not (p / f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required CSVs: {missing}")

    data = {}
    for f in required + optional:
        if (p / f).exists():
            data[f.replace('.csv','')] = pd.read_csv(p / f)
    log(f"Loaded {len(data)} CSVs from {project_dir}")
    return data


def load_metrics_csv(metrics_path):
    """Load the Sooner Stats all-teams metrics file (talent, BCR, achievement, etc.)."""
    p = Path(metrics_path)
    if not p.exists():
        log(f"WARNING: metrics file not found at {metrics_path} — talent/achievement adjustments will be skipped")
        return None
    return pd.read_csv(p)


# ============================================================
# DETECT MODE
# ============================================================
def detect_mode(data, projection_season):
    """Return 'preseason' or 'inseason' based on 2026 game data."""
    sp = data['ratings_sp']
    if projection_season in sp['year'].values:
        n_teams = (sp['year'] == projection_season).sum()
        if n_teams > 50:
            games = data['games']
            completed_2026 = ((games['season'] == projection_season) & (games['completed'])).sum()
            if completed_2026 > 20:
                log(f"IN-SEASON mode: {n_teams} teams with {projection_season} ratings, {completed_2026} games completed")
                return 'inseason'
    log(f"PRESEASON mode: using {projection_season - 1} data with adjustments")
    return 'preseason'


# ============================================================
# CALIBRATION (fit once, cache to disk)
# ============================================================
def build_calibration(data, project_dir):
    """Fit logistic scale and per-team HFA from historical games.
    Cached to <project>/_calibration.json + _team_hfa.json + _spread_lookup.json."""
    p = Path(project_dir)
    calib_path = p / '_calibration.json'
    hfa_path = p / '_team_hfa.json'
    lookup_path = p / '_spread_lookup.json'

    if calib_path.exists() and hfa_path.exists() and lookup_path.exists():
        log("Using cached calibration files")
        with open(calib_path) as f: calib = json.load(f)
        with open(hfa_path) as f: hfa = json.load(f)
        with open(lookup_path) as f: lookup = json.load(f)
        return calib, hfa, lookup

    log("Fitting calibration (one-time, will cache)...")
    from scipy.optimize import minimize_scalar
    from sklearn.linear_model import LinearRegression

    sp = data['ratings_sp']; fpi = data['ratings_fpi']; elo = data['ratings_elo']
    srs = data['ratings_srs']; core = data['core']; games = data['games']

    def build_year_ratings(year):
        prev = year - 1
        dfs = {
            'sp': sp[sp['year']==prev][['team','rating']].rename(columns={'rating':'sp'}),
            'fpi': fpi[fpi['year']==prev][['team','fpi']],
            'elo': elo[elo['year']==prev][['team','elo']],
            'srs': srs[srs['year']==prev][['team','rating']].rename(columns={'rating':'srs'}),
            'core': core[core['season']==prev][['team','core_overall']],
        }
        base = dfs['sp']
        for k in ['fpi','elo','srs','core']:
            base = base.merge(dfs[k], on='team', how='outer')
        for c in ['sp','fpi','elo','srs','core_overall']:
            m = base[c].mean(); s = base[c].std()
            base['z_'+c] = (base[c]-m)/s
        zcols = [c for c in base.columns if c.startswith('z_')]
        base['z_comp'] = base[zcols].mean(axis=1)
        sp_m, sp_s = base['sp'].mean(), base['sp'].std()
        base['power'] = (base['z_comp']*sp_s + sp_m) * 0.75
        return dict(zip(base['team'], base['power']))

    # Fit HFA
    hfa_pairs = []
    for yr in range(2019, 2026):
        ratings = build_year_ratings(yr)
        g = games[(games['season']==yr) & (games['completed']) & (~games['neutral_site'])]
        for _, r in g.iterrows():
            if r['home_team'] in ratings and r['away_team'] in ratings:
                diff = ratings[r['home_team']] - ratings[r['away_team']]
                margin = r['home_points'] - r['away_points']
                hfa_pairs.append({'team': r['home_team'], 'residual': margin - diff})
    hdf = pd.DataFrame(hfa_pairs)
    league_hfa = float(hdf['residual'].mean())
    team_hfa = hdf.groupby('team').agg(n=('residual','size'), hfa=('residual','mean')).reset_index()
    team_hfa = team_hfa[team_hfa['n'] >= 20].copy()
    alpha = 15
    team_hfa['hfa_shrunk'] = (team_hfa['n']*team_hfa['hfa'] + alpha*league_hfa) / (team_hfa['n']+alpha)
    hfa_map = dict(zip(team_hfa['team'], team_hfa['hfa_shrunk'].round(2)))

    # Fit logistic scale using per-team HFA
    pairs = []
    for yr in range(2019, 2026):
        ratings = build_year_ratings(yr)
        g = games[(games['season']==yr) & (games['completed'])]
        for _, r in g.iterrows():
            if r['home_team'] in ratings and r['away_team'] in ratings:
                hfa_val = 0 if r['neutral_site'] else hfa_map.get(r['home_team'], league_hfa)
                pred = ratings[r['home_team']] - ratings[r['away_team']] + hfa_val
                actual = r['home_points'] - r['away_points']
                pairs.append((pred, actual, 1 if actual>0 else 0))
    pdf = pd.DataFrame(pairs, columns=['pred','actual','home_win'])
    def logloss(scale):
        p = 1/(1+np.exp(-pdf['pred']/scale)); p = np.clip(p,1e-6,1-1e-6)
        return -(pdf['home_win']*np.log(p)+(1-pdf['home_win'])*np.log(1-p)).mean()
    scale = float(minimize_scalar(logloss, bounds=(3,15), method='bounded').x)
    p = 1/(1+np.exp(-pdf['pred']/scale))
    brier = float(((p - pdf['home_win'])**2).mean())
    acc = float(((p > 0.5) == (pdf['home_win']==1)).mean())

    # Spread → historical WP lookup
    pdf['bin'] = (pdf['pred']/2).round()*2
    lookup = pdf.groupby('bin').agg(n=('home_win','size'), wp=('home_win','mean')).reset_index()
    lookup = lookup[lookup['n']>=25]
    lookup['bin'] = lookup['bin'].astype(int)
    lookup_dict = {str(int(row['bin'])): [round(float(row['wp']),3), int(row['n'])] for _, row in lookup.iterrows()}

    calib = {'scale': round(scale,3), 'league_hfa': round(league_hfa,2),
             'logloss': round(logloss(scale),4), 'brier': round(brier,4),
             'accuracy': round(acc,4), 'n_games': len(pdf)}
    hfa = {'league_mean': round(league_hfa,2), 'per_team': hfa_map}

    with open(calib_path,'w') as f: json.dump(calib, f, indent=2)
    with open(hfa_path,'w') as f: json.dump(hfa, f)
    with open(lookup_path,'w') as f: json.dump(lookup_dict, f)
    log(f"Calibration done: scale={calib['scale']}, log-loss={calib['logloss']}, accuracy={calib['accuracy']}")
    return calib, hfa, lookup_dict


# ============================================================
# POWER RATINGS
# ============================================================
def build_power_ratings(data, metrics_df, mode, calib, hfa):
    """Build 2026 power ratings. Preseason: adjustments applied. In-season: current-year composite direct."""
    sp = data['ratings_sp']; fpi = data['ratings_fpi']; elo = data['ratings_elo']
    srs = data['ratings_srs']; core = data['core']
    weights = CFG['composite_weights']
    ps = CFG['projection_season']

    # Which season's ratings feed the composite?
    if mode == 'inseason':
        source_year = ps
    else:
        source_year = CFG['base_season']

    sp_src = sp[sp['year']==source_year][['team','conference','rating','offense_rating','defense_rating',
                'specialTeams_rating','ranking','offense_ranking','defense_ranking']].rename(columns={
                'rating':'sp','offense_rating':'sp_off','defense_rating':'sp_def','specialTeams_rating':'sp_st',
                'ranking':'sp_rank','offense_ranking':'sp_off_rank','defense_ranking':'sp_def_rank'})
    fpi_src = fpi[fpi['year']==source_year][['team','fpi']]
    elo_src = elo[elo['year']==source_year][['team','elo']]
    srs_src = srs[srs['year']==source_year][['team','rating']].rename(columns={'rating':'srs'})
    core_src = core[core['season']==source_year][['team','core_overall','core_offense','core_defense']]

    base = sp_src.merge(fpi_src,on='team',how='left').merge(elo_src,on='team',how='left'
              ).merge(srs_src,on='team',how='left').merge(core_src,on='team',how='left')

    if len(base) < 50:
        raise ValueError(f"Only {len(base)} teams found for source year {source_year}. Aborting.")

    def zsc(s): return (s - s.mean()) / s.std()
    for c in ['sp','fpi','elo','srs','core_overall']:
        base['z_'+c] = zsc(base[c])

    # Weighted composite
    zsum = sum(base['z_'+c]*weights[c.replace('core_overall','core')] for c in ['sp','fpi','elo','srs','core_overall'])
    wsum = sum(weights.values())
    base['z_comp'] = zsum / wsum
    sp_mean, sp_std = base['sp'].mean(), base['sp'].std()
    base['power_raw'] = base['z_comp']*sp_std + sp_mean

    # Add 2026 roster metrics (talent, returning prod)
    if metrics_df is not None:
        m_ps = metrics_df[metrics_df['season']==ps][['team','talent_top40_rank','bcr_rank',
                            'returning_prod','proven_vets_rank','stale_2026_roster']]
        base = base.merge(m_ps, on='team', how='left')
        m_prev = metrics_df[metrics_df['season']==ps-1][['team','talent_top40_rank','achievement',
                            'ach_offense','ach_defense']].rename(columns={'talent_top40_rank':'talent_rank_prev'})
        base = base.merge(m_prev, on='team', how='left')
    else:
        for c in ['talent_top40_rank','bcr_rank','returning_prod','proven_vets_rank','stale_2026_roster',
                  'talent_rank_prev','achievement','ach_offense','ach_defense']:
            base[c] = np.nan

    # Coach changes (from coaches.csv, Nov/Dec 2025 hires)
    coaches = data['coaches']
    coaches['hire_dt'] = pd.to_datetime(coaches['hireDate'], errors='coerce')
    new_hc = set()
    hires = coaches[coaches['year']==ps-1]
    for _, r in hires.iterrows():
        if pd.notna(r['hire_dt']) and r['hire_dt'].year==ps-1 and r['hire_dt'].month>=11:
            new_hc.add(r['school'])

    # QB delta
    ppa = data['player_ppa_season']
    usage = data['player_usage']
    rosters = data['rosters']
    r_ps = rosters[rosters['season']==ps][['player_id','team']].rename(columns={'player_id':'id','team':'team_ps'})
    p_prev = ppa[ppa['season']==ps-1][['id','name','position','team','totalPPA_pass']]
    u_prev = usage[usage['season']==ps-1][['id','usage_overall','usage_pass']]
    p_prev = p_prev.merge(u_prev, on='id', how='left')
    qb_ret = p_prev[(p_prev['position']=='QB') & (p_prev['usage_pass']>0.4)]
    qb_ret_on_ps = qb_ret.merge(r_ps, on='id', how='inner')
    top_qb_ppa = dict(zip(qb_ret_on_ps.sort_values('totalPPA_pass',ascending=False).drop_duplicates('team_ps')['team_ps'],
                          qb_ret_on_ps.sort_values('totalPPA_pass',ascending=False).drop_duplicates('team_ps')['totalPPA_pass']))
    prev_starters = p_prev[(p_prev['position']=='QB') & (p_prev['usage_pass']>0.4)].sort_values(
        'totalPPA_pass',ascending=False).drop_duplicates('team')

    def qb_delta(team):
        lost = prev_starters[prev_starters['team']==team]
        if len(lost)>0:
            lost_ppa = lost['totalPPA_pass'].iloc[0]
            if len(r_ps[(r_ps['id']==lost['id'].iloc[0]) & (r_ps['team_ps']==team)]) > 0:
                return 0
        else:
            lost_ppa = None
        new_ppa = top_qb_ppa.get(team)
        if lost_ppa is None and new_ppa is None: return 0
        if lost_ppa is None: lost_ppa = 40
        if new_ppa is None: new_ppa = 40
        return max(-CFG['qb_delta_cap'], min(CFG['qb_delta_cap'], (new_ppa - lost_ppa) / CFG['qb_delta_scale']))

    # Build power_2026 per team with breakdown
    def compute(row):
        r = row['power_raw']
        if pd.isna(r): return None, {}
        b = {'base_2025': round(float(r),2)}

        if mode == 'inseason':
            b['mode'] = 'inseason'
            r_adj = r * CFG['inseason_regress']
            b['after_regression'] = round(float(r_adj),2)
            b['final'] = round(float(r_adj),2)
            return round(float(r_adj),2), b

        # PRESEASON adjustments
        b['mode'] = 'preseason'
        r_adj = r * CFG['preseason_regress']
        b['after_regression'] = round(float(r_adj),2)

        rp = row.get('returning_prod')
        rp_adj = 0
        if pd.notna(rp):
            if rp>0.75: rp_adj=1.5
            elif rp>0.65: rp_adj=0.5
            elif rp<0.45: rp_adj=-1.5
            elif rp<0.55: rp_adj=-0.5
        r_adj += rp_adj
        b['returning_prod_adj'] = round(rp_adj,2)

        t_prev = row.get('talent_rank_prev'); t_curr = row.get('talent_top40_rank')
        stale = row.get('stale_2026_roster', False)
        t_adj = 0
        if pd.notna(t_prev) and pd.notna(t_curr) and not stale:
            d = t_prev - t_curr
            if d>20: t_adj=1.0
            elif d<-20: t_adj=-1.0
        r_adj += t_adj
        b['talent_shift_adj'] = round(t_adj,2)

        ach = row.get('achievement')
        ach_adj = round(float(ach) * CFG['achievement_coef'], 2) if pd.notna(ach) else 0
        r_adj += ach_adj
        b['mean_reversion'] = round(ach_adj,2)

        team = row['team']
        c_adj = CFG['coach_change_penalty'] if team in new_hc else 0
        r_adj += c_adj
        b['coach_change'] = round(c_adj,2)

        q_adj = round(qb_delta(team),2)
        r_adj += q_adj
        b['qb_delta'] = q_adj

        b['final'] = round(float(r_adj),2)
        return round(float(r_adj),2), b

    results = base.apply(compute, axis=1)
    base['power_2026'] = [r[0] for r in results]
    base['adj_breakdown'] = [r[1] for r in results]
    base['power_2026_rank'] = base['power_2026'].rank(ascending=False, method='min').astype('Int64')

    # 2026 conf from schedule (auth source)
    sched = data['cfb_2026_schedule']
    conf_map = {}
    for _, g in sched.iterrows():
        if pd.notna(g['homeTeam']) and pd.notna(g['homeConference']):
            conf_map.setdefault(g['homeTeam'], g['homeConference'])
        if pd.notna(g['awayTeam']) and pd.notna(g['awayConference']):
            conf_map.setdefault(g['awayTeam'], g['awayConference'])
    base['conference'] = base['team'].map(conf_map).fillna(base['conference'])

    log(f"Power ratings built for {len(base)} teams ({mode} mode)")
    return base


# ============================================================
# MERGE ADVANCED STATS
# ============================================================
def merge_advanced(base, data, mode):
    """Add PPA/success/explosive/havoc/etc. stats for the CURRENT display year."""
    adv = data['clean_advanced']
    tr = data['team_records']
    ats_df = data['teams_ats']
    display_year = CFG['projection_season'] if mode == 'inseason' else CFG['base_season']

    adv_disp = adv[adv['season']==display_year]
    cols = ['team','offense_ppa','offense_successRate','offense_explosiveness',
            'offense_rushingPlays_ppa','offense_passingPlays_ppa',
            'offense_havoc_total','offense_lineYards','offense_stuffRate','offense_powerSuccess',
            'defense_ppa','defense_successRate','defense_explosiveness',
            'defense_rushingPlays_ppa','defense_passingPlays_ppa',
            'defense_havoc_total','defense_lineYards','defense_stuffRate','defense_powerSuccess']
    cols = [c for c in cols if c in adv_disp.columns]
    base = base.merge(adv_disp[cols], on='team', how='left')

    tr_disp = tr[tr['year']==display_year][['team','total_wins','total_losses']].rename(
        columns={'total_wins':'w25','total_losses':'l25'})
    base = base.merge(tr_disp, on='team', how='left')
    ats_disp = ats_df[ats_df['year']==display_year][['team','atsWins','atsLosses','atsPushes','avgCoverMargin']]
    base = base.merge(ats_disp, on='team', how='left')

    def add_rank(df, col, asc=False, name=None):
        df[name or (col+'_rank')] = df[col].rank(ascending=asc, method='min')
    for c, asc, nm in [('offense_ppa',False,'off_ppa_rank'),('defense_ppa',True,'def_ppa_rank'),
                      ('offense_successRate',False,'off_sr_rank'),('defense_successRate',True,'def_sr_rank'),
                      ('offense_explosiveness',False,'off_expl_rank'),('defense_explosiveness',True,'def_expl_rank'),
                      ('offense_havoc_total',False,'off_havoc_rank'),('defense_havoc_total',False,'def_havoc_rank'),
                      ('offense_lineYards',False,'off_ly_rank'),('defense_lineYards',True,'def_ly_rank')]:
        if c in base.columns:
            add_rank(base, c, asc, nm)
    return base


# ============================================================
# PROJECT GAMES
# ============================================================
def project_all_games(base, data, calib, hfa):
    """For every 2026 game, compute spread + win probabilities."""
    sched = data['cfb_2026_schedule']
    teams_fbs = set(base['team'])

    # FCS gradings
    fcs_opps = set()
    for _, g in sched.iterrows():
        for t in [g['homeTeam'], g['awayTeam']]:
            if pd.notna(t) and t not in teams_fbs:
                fcs_opps.add(t)
    fcs_ratings = {}
    for opp in fcs_opps:
        fcs_ratings[opp] = CFG['fcs_tier1_rating'] if opp in CFG['fcs_tier1_names'] else CFG['fcs_tier2_rating']

    scale = calib['scale']
    league_hfa = hfa['league_mean']
    team_hfa = hfa['per_team']
    power_map = {r['team']: r['power_2026'] for _, r in base.iterrows() if pd.notna(r['power_2026'])}

    def sigmoid(sp): return 1.0/(1.0+np.exp(-sp/scale))
    def get_power(t):
        if t in power_map: return power_map[t]
        return fcs_ratings.get(t)

    def project(home, away, neutral):
        hp = get_power(home); ap = get_power(away)
        if hp is None or ap is None: return None
        h = 0 if neutral else team_hfa.get(home, league_hfa)
        diff = hp - ap + h
        return {'spread': round(float(diff),1),
                'p_home': round(float(sigmoid(diff)),3),
                'p_away': round(1-float(sigmoid(diff)),3),
                'hfa': round(float(h),1),
                'is_fcs': home not in teams_fbs or away not in teams_fbs}

    schedule_by_team = {}
    projections_all = []
    for _, g in sched.iterrows():
        home = g['homeTeam']; away = g['awayTeam']
        if pd.isna(home) or pd.isna(away): continue
        proj = project(home, away, bool(g['neutralSite']))
        obj = {'game_id': int(g['id']), 'week': int(g['week']),
               'kickoff': g['kickoff_ct'] if not pd.isna(g['kickoff_ct']) else None,
               'tbd': bool(g['startTimeTBD']), 'home': home, 'away': away,
               'homeConf': g['homeConference'] if not pd.isna(g['homeConference']) else '',
               'awayConf': g['awayConference'] if not pd.isna(g['awayConference']) else '',
               'neutral': bool(g['neutralSite']),
               'conf_game': bool(g['conferenceGame']) if not pd.isna(g['conferenceGame']) else False,
               'venue': g['venue'] if not pd.isna(g['venue']) else '',
               'tv': g['tv'] if not pd.isna(g['tv']) else '',
               'notes': g['notes'] if not pd.isna(g['notes']) else '',
               'proj': proj}
        if home in teams_fbs:
            schedule_by_team.setdefault(home, []).append({**obj, 'is_home': True})
        if away in teams_fbs:
            schedule_by_team.setdefault(away, []).append({**obj, 'is_home': False})
        if proj and home in teams_fbs and away in teams_fbs:
            projections_all.append({'game_id':int(g['id']),'week':int(g['week']),'home':home,'away':away,
                                    'neutral':bool(g['neutralSite']),'spread':proj['spread'],
                                    'p_home':proj['p_home']})
    for t in schedule_by_team: schedule_by_team[t].sort(key=lambda x: x['week'])
    log(f"Projected {len(projections_all)} FBS-vs-FBS games; {len(schedule_by_team)} team schedules")
    return schedule_by_team, projections_all, fcs_ratings


# ============================================================
# MONTE CARLO + LEVERAGE
# ============================================================
def monte_carlo(schedule_by_team):
    rng = np.random.default_rng(42)
    N = CFG['n_sims']
    win_dists = {}
    for team, gms in schedule_by_team.items():
        probs = np.array([g['proj']['p_home'] if g['is_home'] else g['proj']['p_away']
                         for g in gms if g['proj']])
        if len(probs)==0: continue
        rolls = rng.random((N, len(probs)))
        wins = (rolls < probs).sum(axis=1)
        n = len(probs)
        dist = np.bincount(wins, minlength=n+1) / N
        win_dists[team] = {
            'n_games': int(n),
            'mean': float(round(wins.mean(),2)),
            'median': int(np.median(wins)),
            'p10': int(np.percentile(wins,10)),
            'p90': int(np.percentile(wins,90)),
            'dist': dist.round(4).tolist(),
            'p_undefeated': float(round((wins==n).mean(),4)),
            'p_9plus': float(round((wins>=9).mean(),4)),
            'p_10plus': float(round((wins>=10).mean(),4)),
            'p_6plus': float(round((wins>=6).mean(),4)),
            'p_losing': float(round((wins<n/2).mean(),4))
        }

    NL = CFG['n_leverage_sims']
    key_games = {}
    for team in schedule_by_team:
        gms = [g for g in schedule_by_team[team] if g['proj']]
        if not gms: continue
        probs = np.array([g['proj']['p_home'] if g['is_home'] else g['proj']['p_away'] for g in gms])
        res = []
        for i, g in enumerate(gms):
            other = np.delete(probs, i)
            r1 = rng.random((NL, len(other))); w1 = (r1<other).sum(axis=1)+1
            r2 = rng.random((NL, len(other))); w2 = (r2<other).sum(axis=1)
            opp = g['away'] if g['is_home'] else g['home']
            res.append({'game_id':g['game_id'],'week':g['week'],'opp':opp,'is_home':g['is_home'],
                        'p_win':round(float(probs[i]),3),
                        'lev_9plus':round(float((w1>=9).mean()-(w2>=9).mean()),3),
                        'lev_bowl':round(float((w1>=6).mean()-(w2>=6).mean()),3)})
        key_games[team] = sorted(res, key=lambda x:-max(x['lev_9plus'],x['lev_bowl']))[:3]
    log(f"Monte Carlo done for {len(win_dists)} teams ({N} sims each)")
    return win_dists, key_games


# ============================================================
# BUILD ANCILLARY DATA (H2H, trends, upsets, tags, video ideas)
# ============================================================
def build_h2h(data):
    games = data['games']
    h2h = {}
    for _, r in games[(games['completed'])].iterrows():
        a,b = r['home_team'], r['away_team']
        k = tuple(sorted([a,b]))
        h2h.setdefault(k, []).append({'season':int(r['season']),'week':int(r['week']),
                                       'home':a,'away':b,'homeP':int(r['home_points']),
                                       'awayP':int(r['away_points']),'neutral':bool(r['neutral_site'])})
    for k in h2h:
        h2h[k].sort(key=lambda x:(x['season'],x['week']), reverse=True)
    return h2h


def build_projection_why(schedule_by_team, teams_dict):
    def why(home, away, proj):
        if proj is None: return ''
        hd = teams_dict.get(home); ad = teams_dict.get(away)
        if not hd or not ad: return ''
        parts = []
        diff = (hd.get('power_2026') or 0) - (ad.get('power_2026') or 0)
        if abs(diff) > 5:
            stronger = home if diff>0 else away
            parts.append(f"{stronger} rated {abs(diff):.1f} pts higher")
        if not proj.get('is_fcs') and abs(proj.get('hfa',0)) > 0.1:
            parts.append(f"HFA at {home} ({proj['hfa']:+.1f})")
        def gaps(a, b):
            edges = []
            for op,dp,name in [('off_ppa_rank','def_ppa_rank','PPA'),
                              ('off_sr_rank','def_sr_rank','success rate'),
                              ('off_expl_rank','def_expl_rank','explosive plays'),
                              ('off_ly_rank','def_ly_rank','line yards')]:
                if a.get(op) and b.get(dp):
                    edges.append((b[dp]-a[op], name, a[op], b[dp]))
            return sorted(edges, key=lambda x:-x[0])
        hg = gaps(hd, ad); ag = gaps(ad, hd)
        if hg and hg[0][0] > 30:
            parts.append(f"{home}'s {hg[0][1]} (#{hg[0][2]}) vs {away}'s D (#{hg[0][3]})")
        if ag and ag[0][0] > 30:
            parts.append(f"{away}'s {ag[0][1]} (#{ag[0][2]}) vs {home}'s D (#{ag[0][3]})")
        return '. '.join(parts[:3]) + '.'

    for team, gms in schedule_by_team.items():
        for g in gms:
            g['why'] = why(g['home'], g['away'], g['proj'])


def build_upset_alerts(projections_all, teams_dict):
    alerts = []
    P4 = {'SEC','Big Ten','Big 12','ACC'}
    def flags(fav, dog):
        f = []
        fd = teams_dict.get(fav,{}); dd = teams_dict.get(dog,{})
        def gap(a,b,g=25): return a is not None and b is not None and (b-a)>=g
        if gap(dd.get('off_expl_rank'), fd.get('def_expl_rank'), 30):
            f.append(f"Explosive offense edge (#{dd.get('off_expl_rank')} vs #{fd.get('def_expl_rank')})")
        if gap(dd.get('off_ly_rank'), fd.get('def_ly_rank'), 30):
            f.append(f"Line yards edge (#{dd.get('off_ly_rank')} vs #{fd.get('def_ly_rank')})")
        if gap(dd.get('def_havoc_rank'), fd.get('off_havoc_rank'), 30):
            f.append(f"Havoc edge (#{dd.get('def_havoc_rank')} vs #{fd.get('off_havoc_rank')})")
        if gap(dd.get('def_ppa_rank'), fd.get('off_ppa_rank'), 30):
            f.append(f"PPA defense edge (#{dd.get('def_ppa_rank')} vs #{fd.get('off_ppa_rank')})")
        dr = dd.get('returning_prod_2026'); fr = fd.get('returning_prod_2026')
        if dr is not None and fr is not None and dr-fr>0.15:
            f.append(f"Returning-production edge ({int(dr*100)}% vs {int(fr*100)}%)")
        return f
    for p in projections_all:
        if p['spread']>0: fav=p['home']; dog=p['away']; sa=p['spread']; dwp=1-p['p_home']
        else: fav=p['away']; dog=p['home']; sa=-p['spread']; dwp=p['p_home']
        if 3.5 <= sa <= 17.0:
            fl = flags(fav, dog)
            if len(fl) >= 2:
                fc = teams_dict.get(fav,{}).get('conference','')
                dc = teams_dict.get(dog,{}).get('conference','')
                alerts.append({'game_id':p['game_id'],'week':p['week'],'favorite':fav,'underdog':dog,
                               'fav_conf':fc,'dog_conf':dc,'spread':round(sa,1),
                               'dog_win_pct':round(dwp*100,1),'flags':fl,'edge_count':len(fl),
                               'is_p4':fc in P4 or dc in P4})
    alerts.sort(key=lambda x:(-x['edge_count'],-x['dog_win_pct']))
    return alerts


def build_tags(teams_dict, trends):
    risers = set(r['team'] for r in trends['form_risers'][:20])
    faders = set(f['team'] for f in trends['form_faders'][:20])
    model_risers = set(r['team'] for r in trends['rank_risers'][:15] if r['shift']>=15)
    model_fallers = set(r['team'] for r in trends['rank_fallers'][:15] if r['shift']<=-15)
    for team, t in teams_dict.items():
        tags = []
        if team in risers: tags.append('ended_hot')
        if team in faders: tags.append('ended_cold')
        if team in model_risers: tags.append('model_riser')
        if team in model_fallers: tags.append('model_faller')
        tr = t.get('talent_rank_2026'); pr = t.get('power_2026_rank')
        if tr and pr and not t.get('stale_roster_2026'):
            if tr-pr>=20: tags.append('overachiever')
            elif pr-tr>=20: tags.append('underachiever')
        if t.get('adj_breakdown',{}).get('coach_change',0)<0: tags.append('new_hc')
        qb = t.get('adj_breakdown',{}).get('qb_delta',0)
        if qb>=1.5: tags.append('qb_upgrade')
        elif qb<=-1.5: tags.append('qb_downgrade')
        rp = t.get('returning_prod_2026')
        if rp is not None:
            if rp<0.45: tags.append('rebuild')
            elif rp>0.75: tags.append('loaded_returning')
        if t.get('stale_roster_2026'): tags.append('stale')
        hfa = t.get('hfa')
        if hfa is not None:
            if hfa>=7: tags.append('fortress')
            elif hfa<=3: tags.append('roadkill')
        ach = t.get('ach_25')
        if ach is not None:
            if ach>=5: tags.append('overachieved_25')
            elif ach<=-5: tags.append('underachieved_25')
        t['tags'] = tags


def build_trends(data, teams_dict, metrics_df):
    ppa = data['player_ppa_season']
    usage = data['player_usage']
    rosters = data['rosters']
    games = data['games']
    ats = data['teams_ats']
    ps = CFG['projection_season']
    fbs = set(teams_dict.keys())

    r_ps = rosters[rosters['season']==ps][['player_id','team']].rename(columns={'player_id':'id','team':'team_ps'})
    p_prev = ppa[ppa['season']==ps-1][['id','name','position','team','totalPPA_all','totalPPA_pass','totalPPA_rush']]
    u_prev = usage[usage['season']==ps-1][['id','usage_overall','usage_pass','usage_rush']]
    p_prev = p_prev.merge(u_prev, on='id', how='left')
    ret = p_prev.merge(r_ps, on='id', how='inner')
    ret['transferred'] = ret['team'] != ret['team_ps']
    ret = ret[ret['team_ps'].isin(fbs)]

    def cl(v,d=1):
        if pd.isna(v): return None
        return round(float(v),d)

    top_qbs = [{'name':r['name'],'team':r['team_ps'],'was':r['team'] if r['transferred'] else None,
                'ppa_pass':cl(r['totalPPA_pass'],1),'ppa_rush':cl(r['totalPPA_rush'],1),
                'usage':cl(r['usage_overall'],2)}
               for _,r in ret[(ret['position']=='QB') & (ret['usage_overall']>=0.30)].nlargest(20,'totalPPA_pass').iterrows()]
    top_skill = [{'name':r['name'],'team':r['team_ps'],'pos':r['position'],
                  'was':r['team'] if r['transferred'] else None,
                  'ppa':cl(r['totalPPA_all'],1),'usage':cl(r['usage_overall'],2)}
                 for _,r in ret[(ret['position'].isin(['WR','RB','TE'])) & (ret['usage_overall']>=0.10)].nlargest(20,'totalPPA_all').iterrows()]
    top_transfers = [{'name':r['name'],'pos':r['position'],'from':r['team'],'to':r['team_ps'],
                      'ppa':cl(r['totalPPA_all'],1),'usage':cl(r['usage_overall'],2)}
                     for _,r in ret[(ret['transferred']) & (ret['totalPPA_all']>40)].nlargest(20,'totalPPA_all').iterrows()]

    g_prev = games[(games['season']==ps-1)&(games['completed'])].copy()
    recs = []
    for _, r in g_prev.iterrows():
        recs.append({'team':r['home_team'],'week':r['week'],'margin':r['home_points']-r['away_points']})
        recs.append({'team':r['away_team'],'week':r['week'],'margin':r['away_points']-r['home_points']})
    dff = pd.DataFrame(recs)
    form_list = []
    for team, grp in dff.groupby('team'):
        if team not in fbs: continue
        grp = grp.sort_values('week')
        if len(grp)<6: continue
        early = grp.iloc[:-4]['margin'].mean(); late = grp.iloc[-4:]['margin'].mean()
        form_list.append({'team':team,'conf':teams_dict[team]['conference'],
                          'early_margin':round(float(early),1),'late_margin':round(float(late),1),
                          'delta':round(float(late-early),1)})
    form_list.sort(key=lambda x:-x['delta'])
    risers = form_list[:15]; faders = form_list[-15:][::-1]

    p25_list = sorted([(t,d['power_2025']) for t,d in teams_dict.items() if d.get('power_2025') is not None],
                      key=lambda x:-x[1])
    p25_rk = {t:i+1 for i,(t,_) in enumerate(p25_list)}
    movers = []
    for t,d in teams_dict.items():
        if d.get('power_2026_rank') and t in p25_rk:
            movers.append({'team':t,'conf':d['conference'],'p25_rank':p25_rk[t],
                           'p26_rank':d['power_2026_rank'],'shift':p25_rk[t]-d['power_2026_rank']})
    movers.sort(key=lambda x:-x['shift'])

    disagreements = []
    for t,d in teams_dict.items():
        vals = [d[k] for k in ['sp_2025','fpi_2025','core_overall_2025','srs_2025'] if d.get(k) is not None]
        if len(vals)<4: continue
        disagreements.append({'team':t,'conf':d['conference'],'sp':d.get('sp_2025'),
                              'fpi':d.get('fpi_2025'),'core':d.get('core_overall_2025'),
                              'srs':d.get('srs_2025'),
                              'spread':round(float(max(vals)-min(vals)),1)})
    disagreements.sort(key=lambda x:-x['spread'])

    overachievers_25 = []; underachievers_25 = []
    if metrics_df is not None:
        m_prev = metrics_df[metrics_df['season']==ps-1].dropna(subset=['achievement'])
        overs = m_prev.nsmallest(15,'achievement_rank')
        unders = m_prev.nlargest(15,'achievement_rank')
        overachievers_25 = [{'team':r['team'],'conf':r['conference'],'ach':round(float(r['achievement']),1),
                             'ach_o':round(float(r['ach_offense']),1) if pd.notna(r['ach_offense']) else None,
                             'ach_d':round(float(r['ach_defense']),1) if pd.notna(r['ach_defense']) else None,
                             'talent_rank':int(r['talent_top40_rank']) if pd.notna(r['talent_top40_rank']) else None}
                            for _,r in overs.iterrows()]
        underachievers_25 = [{'team':r['team'],'conf':r['conference'],'ach':round(float(r['achievement']),1),
                              'ach_o':round(float(r['ach_offense']),1) if pd.notna(r['ach_offense']) else None,
                              'ach_d':round(float(r['ach_defense']),1) if pd.notna(r['ach_defense']) else None,
                              'talent_rank':int(r['talent_top40_rank']) if pd.notna(r['talent_top40_rank']) else None}
                             for _,r in unders.iterrows()]

    ats_prev = ats[ats['year']==ps-1].sort_values('avgCoverMargin',ascending=False)
    ats_leaders = [{'team':r['team'],'conf':r['conference'],'w':int(r['atsWins']),'l':int(r['atsLosses']),
                    'margin':round(float(r['avgCoverMargin']),1)}
                   for _,r in ats_prev.head(10).iterrows() if r['team'] in fbs]
    ats_losers = [{'team':r['team'],'conf':r['conference'],'w':int(r['atsWins']),'l':int(r['atsLosses']),
                   'margin':round(float(r['avgCoverMargin']),1)}
                  for _,r in ats_prev.tail(10).iloc[::-1].iterrows() if r['team'] in fbs]

    return {'top_qbs':top_qbs,'top_skill':top_skill,'top_transfers':top_transfers,
            'form_risers':risers,'form_faders':faders,
            'rank_risers':movers[:15],'rank_fallers':movers[-15:][::-1],
            'model_disagreements':disagreements[:12],
            'overachievers_25':overachievers_25,'underachievers_25':underachievers_25,
            'ats_leaders':ats_leaders,'ats_losers':ats_losers}


def build_video_ideas(teams_dict, trends, projections_all, upsets):
    ideas = []
    ou = teams_dict.get('Oklahoma')
    if ou and ou.get('win_dist'):
        wd = ou['win_dist']
        ideas.append({'title':f"OU projects at #{ou['power_2026_rank']} — realistic 2026 take",
                      'angle':f"Model averages {wd['mean']} wins with {int(wd['p_9plus']*100)}% shot at 9+.",
                      'category':'OU','urgency':'high'})
    for d in trends['model_disagreements'][:5]:
        if d['team'] in teams_dict:
            ideas.append({'title':f"Nobody agrees on {d['team']}: SP+ {d['sp']}, FPI {d['fpi']}",
                          'angle':f"{d['spread']:.1f} pts of disagreement across systems.",
                          'category':'National','urgency':'medium'})
    for team, dd in teams_dict.items():
        wd = dd.get('win_dist')
        if wd and wd['p_undefeated']>0.10 and dd.get('power_2026_rank',999)>5:
            ideas.append({'title':f"{team} is a live undefeated candidate",
                          'angle':f"{wd['p_undefeated']*100:.1f}% chance. P90: {wd['p90']} wins.",
                          'category':'National','urgency':'medium'})
    for f in trends['form_faders'][:3]:
        ideas.append({'title':f"{f['team']} ended 2025 at {f['late_margin']:+.0f} margin",
                      'angle':f"Swung {f['delta']:.1f} pts from earlier. Fade watch.",
                      'category':'National','urgency':'medium'})
    seen = set(); out = []
    for v in ideas:
        if v['title'] not in seen: seen.add(v['title']); out.append(v)
    return out


# ============================================================
# ASSEMBLE FINAL DATA
# ============================================================
def build_teams_dict(base, win_dists, key_games, team_hfa_map, league_hfa):
    def num(v,d=2):
        if pd.isna(v): return None
        try: return round(float(v),d)
        except: return None
    def rk(v):
        if pd.isna(v): return None
        return int(v)
    out = {}
    for _, row in base.iterrows():
        t = row['team']
        out[t] = {
            'team': t, 'conference': row.get('conference'),
            'power_2026': num(row['power_2026']), 'power_2026_rank': rk(row['power_2026_rank']),
            'power_2025': num(row.get('power_raw')), 'adj_breakdown': row.get('adj_breakdown',{}),
            'hfa': team_hfa_map.get(t, league_hfa),
            'sp_2025': num(row.get('sp')), 'sp_rank_2025': rk(row.get('sp_rank')),
            'sp_off_2025': num(row.get('sp_off')), 'sp_off_rank_2025': rk(row.get('sp_off_rank')),
            'sp_def_2025': num(row.get('sp_def')), 'sp_def_rank_2025': rk(row.get('sp_def_rank')),
            'sp_st_2025': num(row.get('sp_st')), 'fpi_2025': num(row.get('fpi')),
            'elo_2025': int(row['elo']) if pd.notna(row.get('elo')) else None,
            'srs_2025': num(row.get('srs')),
            'core_overall_2025': num(row.get('core_overall')),
            'core_offense_2025': num(row.get('core_offense')),
            'core_defense_2025': num(row.get('core_defense')),
            'w25': int(row['w25']) if pd.notna(row.get('w25')) else None,
            'l25': int(row['l25']) if pd.notna(row.get('l25')) else None,
            'ats_w': int(row['atsWins']) if pd.notna(row.get('atsWins')) else None,
            'ats_l': int(row['atsLosses']) if pd.notna(row.get('atsLosses')) else None,
            'ats_p': int(row['atsPushes']) if pd.notna(row.get('atsPushes')) else None,
            'ats_margin': num(row.get('avgCoverMargin')),
            'talent_rank_2026': rk(row.get('talent_top40_rank')),
            'bcr_rank_2026': rk(row.get('bcr_rank')),
            'returning_prod_2026': num(row.get('returning_prod'),3),
            'proven_vets_rank_2026': rk(row.get('proven_vets_rank')),
            'stale_roster_2026': bool(row.get('stale_2026_roster', False)),
            'ach_25': num(row.get('achievement')),
            'ach_off_25': num(row.get('ach_offense')),
            'ach_def_25': num(row.get('ach_defense')),
            'off_ppa': num(row.get('offense_ppa'),3), 'off_ppa_rank': rk(row.get('off_ppa_rank')),
            'def_ppa': num(row.get('defense_ppa'),3), 'def_ppa_rank': rk(row.get('def_ppa_rank')),
            'off_sr': num(row.get('offense_successRate'),3), 'off_sr_rank': rk(row.get('off_sr_rank')),
            'def_sr': num(row.get('defense_successRate'),3), 'def_sr_rank': rk(row.get('def_sr_rank')),
            'off_expl': num(row.get('offense_explosiveness'),3), 'off_expl_rank': rk(row.get('off_expl_rank')),
            'def_expl': num(row.get('defense_explosiveness'),3), 'def_expl_rank': rk(row.get('def_expl_rank')),
            'off_havoc': num(row.get('offense_havoc_total'),3), 'off_havoc_rank': rk(row.get('off_havoc_rank')),
            'def_havoc': num(row.get('defense_havoc_total'),3), 'def_havoc_rank': rk(row.get('def_havoc_rank')),
            'off_ly': num(row.get('offense_lineYards'),3), 'off_ly_rank': rk(row.get('off_ly_rank')),
            'def_ly': num(row.get('defense_lineYards'),3), 'def_ly_rank': rk(row.get('def_ly_rank')),
            'off_rush_ppa': num(row.get('offense_rushingPlays_ppa'),3),
            'off_pass_ppa': num(row.get('offense_passingPlays_ppa'),3),
            'def_rush_ppa': num(row.get('defense_rushingPlays_ppa'),3),
            'def_pass_ppa': num(row.get('defense_passingPlays_ppa'),3),
            'win_dist': win_dists.get(t),
            'key_games': key_games.get(t, [])
        }
    return out


def add_hardest_easiest(teams_dict, schedule_by_team):
    for team, gms in schedule_by_team.items():
        if team not in teams_dict: continue
        with_wp = [(g, g['proj']['p_home'] if g['is_home'] else g['proj']['p_away'])
                   for g in gms if g['proj']]
        if not with_wp:
            teams_dict[team]['hardest_game'] = None
            teams_dict[team]['easiest_game'] = None
            continue
        with_wp.sort(key=lambda x: x[1])
        h = with_wp[0][0]; e = with_wp[-1][0]
        def summ(g):
            opp = g['away'] if g['is_home'] else g['home']
            wp = g['proj']['p_home'] if g['is_home'] else g['proj']['p_away']
            return {'week':g['week'],'opp':opp,'is_home':g['is_home'],'wp':round(wp,3),'game_id':g['game_id']}
        teams_dict[team]['hardest_game'] = summ(h)
        teams_dict[team]['easiest_game'] = summ(e)


def build_sos_and_wins(schedule_by_team, teams_dict, win_dists):
    fbs = set(teams_dict.keys())
    sos_list = []
    for team, gms in schedule_by_team.items():
        opps = []
        for g in gms:
            opp = g['away'] if g['is_home'] else g['home']
            if opp in fbs and teams_dict[opp].get('power_2026') is not None:
                opps.append(teams_dict[opp]['power_2026'])
        if opps: sos_list.append({'team':team,'sos':round(sum(opps)/len(opps),2),'fbs_games':len(opps)})
    sos_list.sort(key=lambda x:-x['sos'])
    sos_by_team = {s['team']:{'sos':s['sos'],'sos_rank':i+1,'fbs_games':s['fbs_games']}
                   for i,s in enumerate(sos_list)}

    proj_wins = []
    for team, wd in win_dists.items():
        proj_wins.append({'team':team,'proj_wins':wd['mean'],'games':wd['n_games'],'p_9plus':wd['p_9plus']})
    proj_wins.sort(key=lambda x:-x['proj_wins'])
    pw_by_team = {p['team']:{'proj_wins':p['proj_wins'],'games':p['games'],'proj_rank':i+1,
                             'p_9plus':p['p_9plus']} for i,p in enumerate(proj_wins)}
    return sos_by_team, pw_by_team


def build_week_bounds(data):
    sched = data['cfb_2026_schedule'].copy()
    sched['kickoff_ct'] = pd.to_datetime(sched['kickoff_ct'], errors='coerce', utc=True)
    wb = {}
    for wk, grp in sched.groupby('week'):
        v = grp['kickoff_ct'].dropna()
        if len(v): wb[int(wk)] = {'start':v.min().strftime('%Y-%m-%d'),
                                   'end':v.max().strftime('%Y-%m-%d')}
    return wb


# ============================================================
# INJECT INTO TEMPLATE
# ============================================================
def render_html(template_path, out_path, data_obj):
    with open(template_path) as f:
        tmpl = f.read()
    if '__DATA_JSON__' not in tmpl:
        raise ValueError(f"Template at {template_path} missing __DATA_JSON__ placeholder")
    data_json = json.dumps(data_obj, separators=(',',':'))
    out = tmpl.replace('__DATA_JSON__', data_json)
    with open(out_path, 'w') as f:
        f.write(out)
    log(f"Wrote {out_path} ({os.path.getsize(out_path)/1024/1024:.2f} MB)")


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser(description='Sooner Stats weekly refresh')
    ap.add_argument('--project', default='./project', help='Dir with CFBD CSVs')
    ap.add_argument('--metrics', default='./sooner_stats_all_teams_metrics.csv',
                    help='Path to all-teams metrics CSV')
    ap.add_argument('--template', default='./template.html', help='HTML template file')
    ap.add_argument('--out', default='./index.html', help='Output HTML file')
    ap.add_argument('--built-at', default=None, help='Override built-at date')
    args = ap.parse_args()

    if not Path(args.project).exists():
        log(f"ERROR: project dir {args.project} not found")
        sys.exit(1)
    if not Path(args.template).exists():
        log(f"ERROR: template {args.template} not found")
        sys.exit(1)

    # 1. Load data
    data = load_csvs(args.project)
    metrics_df = load_metrics_csv(args.metrics)

    # 2. Detect mode
    mode = detect_mode(data, CFG['projection_season'])

    # 3. Calibration
    calib, hfa, spread_lookup = build_calibration(data, args.project)

    # 4. Power ratings
    base = build_power_ratings(data, metrics_df, mode, calib, hfa)
    base = merge_advanced(base, data, mode)

    # 5. Project games
    schedule_by_team, projections_all, fcs_ratings = project_all_games(base, data, calib, hfa)

    # 6. Monte Carlo
    win_dists, key_games = monte_carlo(schedule_by_team)

    # 7. Build teams dict + attach ancillary
    teams_dict = build_teams_dict(base, win_dists, key_games, hfa['per_team'], hfa['league_mean'])
    add_hardest_easiest(teams_dict, schedule_by_team)

    # 8. H2H, trends, upsets, tags, video ideas
    h2h = build_h2h(data)
    trends = build_trends(data, teams_dict, metrics_df)
    build_projection_why(schedule_by_team, teams_dict)
    upsets = build_upset_alerts(projections_all, teams_dict)
    build_tags(teams_dict, trends)
    videos = build_video_ideas(teams_dict, trends, projections_all, upsets)

    # 9. SOS + wins summaries
    sos_by_team, pw_by_team = build_sos_and_wins(schedule_by_team, teams_dict, win_dists)

    # 10. Week bounds
    wb = build_week_bounds(data)

    # 11. Assemble final data
    from datetime import datetime
    built_at = args.built_at or datetime.now().strftime('%Y-%m-%d')
    tossups = sorted([p for p in projections_all if abs(p['spread'])<=3.0], key=lambda x:abs(x['spread']))
    blowouts = sorted(projections_all, key=lambda x:-abs(x['spread']))[:30]
    P4 = {'SEC','Big Ten','Big 12','ACC'}
    p4_blowouts = [b for b in blowouts if teams_dict.get(b['home'],{}).get('conference') in P4
                   and teams_dict.get(b['away'],{}).get('conference') in P4][:12]
    trends['tossups'] = tossups[:15]
    trends['p4_blowouts'] = p4_blowouts

    final_data = {
        'teams': teams_dict, 'schedule_by_team': schedule_by_team,
        'h2h': {f"{k[0]}|{k[1]}":v for k,v in h2h.items()},
        'sos_by_team': sos_by_team, 'proj_wins_by_team': pw_by_team,
        'upset_alerts': upsets, 'tossups': tossups[:30], 'blowouts': blowouts,
        'trends': trends, 'video_ideas': videos, 'week_bounds': wb,
        'spread_lookup': spread_lookup,
        'meta': {
            'built_at': built_at, 'base_season': CFG['base_season'],
            'projection_season': CFG['projection_season'],
            'scale': calib['scale'], 'league_hfa': calib['league_hfa'],
            'model_logloss': calib['logloss'], 'model_brier': calib['brier'],
            'model_accuracy': calib['accuracy'], 'n_games_calibration': calib['n_games'],
            'n_sims': CFG['n_sims'], 'mode': mode,
            'notes': f'Refresh mode: {mode}. Achievement coef {CFG["achievement_coef"]}. CORE weight {CFG["composite_weights"]["core"]}.'
        }
    }

    # 12. Render
    render_html(args.template, args.out, final_data)
    log(f"DONE. Mode: {mode}. Teams: {len(teams_dict)}. Upsets: {len(upsets)}. Videos: {len(videos)}.")


if __name__ == '__main__':
    main()
