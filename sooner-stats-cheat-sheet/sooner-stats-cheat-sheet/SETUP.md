# Sooner Stats — Automated Weekly Refresh

Zero-touch setup. Push code once, updates deploy weekly forever.

## The end result

- Every Sunday at 11 PM Central, GitHub Actions runs the whole pipeline automatically
- Pulls fresh CFBD data → rebuilds `index.html` → commits it → Netlify redeploys
- You do nothing
- You can also trigger it manually with a button click, or after any specific game

## One-time setup (15 minutes total)

### Step 1: Create the GitHub repo (2 minutes)

1. Go to [github.com/new](https://github.com/new)
2. Repo name: `sooner-stats-cheat-sheet` (or whatever)
3. Public
4. No README, no .gitignore, no license (we'll add them)
5. **Create repository**

### Step 2: Populate the repo (5 minutes)

You need to upload these files, keeping the folder structure:

```
sooner-stats-cheat-sheet/
├── .github/workflows/weekly-refresh.yml
├── scripts/refresh.py
├── scripts/pull_cfbd.py
├── project/sooner_stats_all_teams_metrics.csv
├── template.html
├── requirements.txt
└── README.md
```

Two ways to do this:

**Option A — Web upload (mobile-friendly):**
1. On the empty repo page, click **"uploading an existing file"**
2. Drag `template.html`, `requirements.txt`, `README.md` in → Commit
3. Click **Add file → Create new file** → type `scripts/refresh.py` in the filename box (this creates the `scripts/` folder). Paste in the contents of `refresh.py` → Commit.
4. Repeat for `scripts/pull_cfbd.py`
5. Create `.github/workflows/weekly-refresh.yml` the same way
6. Create `project/` folder by uploading the metrics CSV: **Add file → Upload files** → drag `sooner_stats_all_teams_metrics.csv` in → **before committing**, in the filename field above the drop zone, change it to `project/sooner_stats_all_teams_metrics.csv` → Commit.

**Option B — Command line (if you have git anywhere):**
```bash
git clone https://github.com/YOUR_USERNAME/sooner-stats-cheat-sheet
cd sooner-stats-cheat-sheet
mkdir -p .github/workflows scripts project
# copy files into their spots
git add .
git commit -m "Initial setup"
git push
```

### Step 3: Add your CFBD API key as a repo secret (1 minute)

1. In your repo → **Settings** (tab at top of repo)
2. Left sidebar → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `CFBD_API_KEY`
5. Secret: paste your CFBD API key (get one free at [collegefootballdata.com](https://collegefootballdata.com))
6. **Add secret**

### Step 4: Give Actions permission to push commits (1 minute)

1. Repo → **Settings** → **Actions** → **General**
2. Scroll down to **Workflow permissions**
3. Select **"Read and write permissions"**
4. Check **"Allow GitHub Actions to create and approve pull requests"**
5. **Save**

### Step 5: Hook up Netlify (3 minutes)

1. Go to [app.netlify.com](https://app.netlify.com) → **Add new site** → **Import an existing project**
2. **Deploy with GitHub** → authorize → pick your `sooner-stats-cheat-sheet` repo
3. Build settings:
   - Build command: *(leave blank)*
   - Publish directory: `.`
4. **Deploy site**
5. Optional: **Domain settings** → add a custom domain if you want

Netlify auto-detects that new commits should trigger a redeploy. Nothing else to configure.

### Step 6: Test the automated pipeline (3 minutes)

Rather than wait for Sunday, trigger it manually:

1. Repo → **Actions** tab
2. Click **Weekly Refresh** in the left sidebar
3. Click **Run workflow** → **Run workflow** (green button)
4. Watch the yellow spinning icon (~90 seconds)
5. Green checkmark = success
6. Go to your Netlify site — the updated `index.html` should be live

If anything went wrong, click the failed run to see the error log.

---

## Every week from now on

**You do nothing.**

Sunday 11 PM CT: GitHub Actions runs, pulls fresh data, rebuilds the site, commits it, Netlify redeploys.

Monday morning: fresh site.

## Manual triggers

**After a big Saturday game:** don't wait for Sunday. Go to **Actions → Weekly Refresh → Run workflow**. Site is fresh within 2 minutes.

**Testing a change:** any commit to `scripts/` or `template.html` auto-triggers a fresh build.

## What each file does

| File | Purpose | How often you edit it |
|---|---|---|
| `.github/workflows/weekly-refresh.yml` | The automation script | Almost never |
| `scripts/refresh.py` | Builds `index.html` from CSVs | If you want to tweak the model |
| `scripts/pull_cfbd.py` | Downloads fresh CSVs from CFBD | Never |
| `template.html` | The UI shell | If we're changing UI |
| `project/sooner_stats_all_teams_metrics.csv` | Your uploaded metrics file | When you re-run your metrics pipeline |
| `project/*.csv` (auto-generated) | Fresh CFBD data | Never (workflow overwrites weekly) |
| `index.html` | The live app | Never (workflow overwrites weekly) |
| `requirements.txt` | Python packages | Never |

## Adjusting the schedule

The cron in `.github/workflows/weekly-refresh.yml` runs at `0 5 * * 1` — that's 5 AM UTC Monday, which is 11 PM CT Sunday.

Common alternatives:
- Twice a week (Sun + Wed nights): `0 5 * * 1,4`
- Sunday morning: `0 15 * * 0` (10 AM CT)
- Every day: `0 5 * * *`

Edit the workflow file, commit, done.

## What to expect the first Sunday

If you set this up on a Tuesday, the first scheduled run is next Monday morning. Verify manually with Step 6 before then so you're not waiting a week to see if it works.

## Costs

- **GitHub Actions**: 2,000 free minutes/month on public repos. Each run takes ~2 min = 8 min/month. Plenty of headroom.
- **CFBD API**: free tier is 1000 requests/month. Each weekly pull uses ~10. ~40/month = fine.
- **Netlify**: free tier is 100 GB bandwidth/month + 300 build minutes. Static site with a few MB — you'll never hit any limit.

Everything free.

## Troubleshooting

**"Bad credentials" in Actions logs:**
- Your CFBD API key is wrong. Regenerate at collegefootballdata.com, update the secret.

**"Permission denied" trying to push:**
- Step 4 wasn't done. Settings → Actions → General → workflow permissions → read and write.

**Workflow says "no changes to commit":**
- Fine! Means CFBD hasn't updated anything since the last run. Site stays where it is.

**Netlify doesn't redeploy:**
- Verify it's connected to the right repo and branch (main). Netlify → your site → Site settings → Build & deploy.

**CFBD endpoint 404s for CORE:**
- Known issue — CFBD's public API may not expose CORE. The pull script gracefully skips it and reuses your existing `core.csv` in `project/`. Make sure that file exists in the repo.

**Actions run failed with Python error:**
- Click the failed run → click the failed step → read the traceback. Most common: a CSV column changed in the CFBD API. Ping me with the error, I'll patch `refresh.py`.

## What I skipped

**Netlify build hooks / branch previews / staging:** you don't need them for a static site with one file. Keep it simple.

**Environment separation (staging vs prod):** overkill for this. If you want it later, create a `staging` branch, add a second Netlify site pointing at it, done.

**Automated tests:** the pipeline itself is a test — if it produces an `index.html` that's not empty, it works. If it fails, the workflow fails visibly.

## Optional: add a status badge to your README

At the top of your GitHub `README.md`, add:

```markdown
![Weekly Refresh](https://github.com/YOUR_USERNAME/sooner-stats-cheat-sheet/actions/workflows/weekly-refresh.yml/badge.svg)
```

Now the README shows a green/red badge for the last workflow run. Nice at-a-glance health check.

---

## Summary

- **You upload files once** (Step 2)
- **You set two settings** (Steps 3 + 4)
- **You connect Netlify once** (Step 5)
- **After that, forever, it runs itself**

Fresh site every Monday morning. No touching anything unless you want to change the model or the UI.
