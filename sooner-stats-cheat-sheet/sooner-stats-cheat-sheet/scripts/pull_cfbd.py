#!/usr/bin/env python3
"""
Sooner Stats — CFBD Data Pull
Pulls fresh CFBD CSVs into your project directory.

USAGE:
    export CFBD_API_KEY="your_key_here"
    python pull_cfbd.py                          # pulls current-week essentials
    python pull_cfbd.py --full                   # pulls everything
    python pull_cfbd.py --out ./project          # custom output dir

Get a free API key at collegefootballdata.com.

Skips files by category:
    --essential (default): ratings, games, advanced stats, CORE, schedule
    --full: everything (players, rosters, portal, recruits, etc.)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests


CFBD_BASE = 'https://apinext.collegefootballdata.com'


def _norm(records):
    """Flatten CFBD's nested JSON to flat dict rows."""
    def flatten(obj, prefix=''):
        out = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                nk = f"{prefix}_{k}" if prefix else k
                if isinstance(v, (dict, list)):
                    out.update(flatten(v, nk))
                else:
                    out[nk] = v
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                out.update(flatten(item, f"{prefix}_{i}"))
        else:
            out[prefix] = obj
        return out
    return [flatten(r) for r in records]


class CFBDClient:
    def __init__(self, api_key):
        self.headers = {'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'}
        self.session = requests.Session()

    def get(self, path, **params):
        params = {k: v for k, v in params.items() if v is not None}
        try:
            r = self.session.get(f'{CFBD_BASE}{path}', headers=self.headers, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            print(f"  ! HTTP {r.status_code} on {path}: {e}", file=sys.stderr)
            return []
        except Exception as e:
            print(f"  ! Error on {path}: {e}", file=sys.stderr)
            return []


ESSENTIAL_FILES = {
    'ratings_sp.csv': lambda c, yr: c.get('/ratings/sp', year=yr),
    'ratings_fpi.csv': lambda c, yr: c.get('/ratings/fpi', year=yr),
    'ratings_srs.csv': lambda c, yr: c.get('/ratings/srs', year=yr),
    'ratings_elo.csv': lambda c, yr: c.get('/ratings/elo', year=yr),
    'games.csv': lambda c, yr: c.get('/games', year=yr, seasonType='both'),
    'clean_advanced.csv': lambda c, yr: c.get('/stats/season/advanced', year=yr),
    'cfb_2026_schedule.csv': lambda c, yr: c.get('/games', year=2026, seasonType='regular'),
    'team_records.csv': lambda c, yr: c.get('/records', year=yr),
}

# CORE isn't in the standard CFBD API — placeholder. Skip if endpoint 404s.
def fetch_core(c, yr):
    return c.get('/ratings/core', year=yr)

# Extras (slower cadence)
EXTRA_FILES = {
    'teams_ats.csv': lambda c, yr: c.get('/records/ats', year=yr),
    'rankings.csv': lambda c, yr: c.get('/rankings', year=yr),
    'coaches.csv': lambda c, yr: c.get('/coaches', year=yr),
    'rosters.csv': lambda c, yr: c.get('/roster', year=yr),
    'player_ppa_season.csv': lambda c, yr: c.get('/ppa/players/season', year=yr),
    'player_usage.csv': lambda c, yr: c.get('/player/usage', year=yr),
    'talent.csv': lambda c, yr: c.get('/talent', year=yr),
    'clean_metrics.csv': lambda c, yr: c.get('/stats/season', year=yr),
    'team_ppa_season.csv': lambda c, yr: c.get('/ppa/teams', year=yr),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='./project', help='Output CSV dir')
    ap.add_argument('--years', default='2025,2026', help='Years to pull (comma-separated)')
    ap.add_argument('--full', action='store_true', help='Pull player/roster/coach files too')
    ap.add_argument('--historical', action='store_true', help='Pull 2019-2026 for calibration rebuild')
    args = ap.parse_args()

    api_key = os.environ.get('CFBD_API_KEY')
    if not api_key:
        print("ERROR: set CFBD_API_KEY environment variable")
        print("Get one at https://collegefootballdata.com")
        sys.exit(1)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    years = [int(y.strip()) for y in args.years.split(',')]
    if args.historical:
        years = list(range(2019, 2027))

    client = CFBDClient(api_key)

    # Multi-year files: accumulate rows across years
    def pull_multiyear(name, fn):
        print(f"→ {name}")
        all_rows = []
        for yr in years:
            time.sleep(0.5)  # be nice to the API
            data = fn(client, yr)
            if data:
                all_rows.extend(data)
                print(f"  {yr}: {len(data)} rows")
        if all_rows:
            df = pd.DataFrame(_norm(all_rows))
            df.to_csv(out / name, index=False)
            print(f"  saved: {out/name} ({len(df)} total)")
        else:
            print(f"  ! no data returned")

    # Pull essentials
    for name, fn in ESSENTIAL_FILES.items():
        pull_multiyear(name, fn)

    # CORE (may not exist on CFBD — user has as separate file)
    print("→ core.csv (attempting CFBD endpoint; may not exist)")
    try:
        all_core = []
        for yr in years:
            time.sleep(0.5)
            data = fetch_core(client, yr)
            if data:
                all_core.extend(data)
        if all_core:
            pd.DataFrame(_norm(all_core)).to_csv(out / 'core.csv', index=False)
            print("  saved")
        else:
            print("  ! CORE endpoint returned nothing — will use existing core.csv if present")
    except Exception as e:
        print(f"  ! CORE unavailable: {e}")

    if args.full:
        print("\n--- Pulling extras (--full mode) ---")
        for name, fn in EXTRA_FILES.items():
            pull_multiyear(name, fn)

    print("\nDone. Files in:", out)


if __name__ == '__main__':
    main()
