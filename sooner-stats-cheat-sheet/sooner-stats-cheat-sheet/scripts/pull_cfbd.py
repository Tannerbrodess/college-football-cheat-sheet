#!/usr/bin/env python3
"""Sooner Stats — CFBD Data Pull (rate-limit friendly)"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests


CFBD_BASE = 'https://apinext.collegefootballdata.com'


def _norm(records):
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
    def __init__(self, api_key, base_delay=3.0):
        self.headers = {'Authorization': f'Bearer {api_key}', 'Accept': 'application/json'}
        self.session = requests.Session()
        self.base_delay = base_delay
        self.consecutive_429s = 0

    def get(self, path, **params):
        params = {k: v for k, v in params.items() if v is not None}
        max_retries = 4
        for attempt in range(max_retries):
            try:
                sleep_time = self.base_delay + (self.consecutive_429s * 5)
                if attempt == 0:
                    time.sleep(sleep_time)
                r = self.session.get(f'{CFBD_BASE}{path}', headers=self.headers, params=params, timeout=60)
                if r.status_code == 429:
                    self.consecutive_429s += 1
                    wait = int(r.headers.get('Retry-After', 30 * (attempt + 1)))
                    print(f"  ⏳ 429 on {path} (attempt {attempt+1}/{max_retries}), waiting {wait}s", file=sys.stderr)
                    time.sleep(wait)
                    continue
                if r.status_code >= 500:
                    wait = 10 * (attempt + 1)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                self.consecutive_429s = 0
                return r.json()
            except requests.exceptions.HTTPError as e:
                print(f"  ! HTTP {r.status_code} on {path}: {e}", file=sys.stderr)
                return []
            except Exception as e:
                print(f"  ! Error on {path}: {e}", file=sys.stderr)
                if attempt < max_retries - 1:
                    time.sleep(15)
                    continue
                return []
        if self.consecutive_429s >= 6:
            print(f"\n!!! Aborting: {self.consecutive_429s} consecutive rate limits.", file=sys.stderr)
            sys.exit(2)
        return []


# NOTE: CORE is NOT fetched — the CFBD /ratings/core endpoint returns a different schema
# than the model expects. Keep the existing core.csv in project/ (computed externally by your metrics pipeline).
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


EXTRA_FILES = {
    'coaches.csv': lambda c, yr: c.get('/coaches', year=yr),
    'rosters.csv': lambda c, yr: c.get('/roster', year=yr),
    'player_ppa_season.csv': lambda c, yr: c.get('/ppa/players/season', year=yr),
    'player_usage.csv': lambda c, yr: c.get('/player/usage', year=yr),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='./project')
    ap.add_argument('--years', default='2025,2026')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--delay', type=float, default=3.0)
    args = ap.parse_args()

    api_key = os.environ.get('CFBD_API_KEY')
    if not api_key:
        print("ERROR: set CFBD_API_KEY environment variable")
        sys.exit(1)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    years = [int(y.strip()) for y in args.years.split(',')]
    client = CFBDClient(api_key, base_delay=args.delay)

    def pull_multiyear(name, fn):
        print(f"→ {name}")
        all_rows = []
        for yr in years:
            data = fn(client, yr)
            if data:
                all_rows.extend(data)
                print(f"  ✓ {yr}: {len(data)} rows")
        if all_rows:
            df = pd.DataFrame(_norm(all_rows))
            df.to_csv(out / name, index=False)
            print(f"  saved: {out/name} ({len(df)} total)")
        else:
            print(f"  ! no data returned — keeping existing {name} if present")

    for name, fn in ESSENTIAL_FILES.items():
        pull_multiyear(name, fn)

    # CORE intentionally not fetched — keep existing core.csv
    if (out / 'core.csv').exists():
        print("→ core.csv (kept as-is from your metrics pipeline)")
    else:
        print("! core.csv is not present in project/ — model will run without CORE (4-system composite)")

    if args.full:
        print("\n--- Pulling extras (--full mode) ---")
        for name, fn in EXTRA_FILES.items():
            pull_multiyear(name, fn)

    print("\nDone. Files in:", out)


if __name__ == '__main__':
    main()
