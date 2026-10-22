# Deploying Amara

This covers going from "runs on my machine with local MongoDB" to "runs on a
free-tier host with a real database." Read the whole thing once before
starting -- a couple of steps (seeding, the model build step) are easy to
miss and will leave you with a broken deploy that *looks* like it started
fine.

## 0. What changed to make this deployable

Two things in the code were fixed specifically for this:

1. **Doctor verification documents** used to be saved to local disk. Free
   hosting tiers (Render, Railway) have an *ephemeral filesystem* -- anything
   written to disk is wiped on every restart or redeploy. Documents are now
   stored as base64 inside the doctor's own MongoDB document instead, so they
   persist exactly as long as the account does. No action needed from you,
   just don't be surprised if you go looking for a `verification_docs/`
   folder and it's gone -- it's not used anymore.
2. **`models/baseline_pipeline.joblib` is gitignored** (per your original
   `.gitignore`) and isn't committed to your repo. Instead, the deploy's
   *build step* runs `python3 models/train_baseline.py` fresh, which reads
   your committed `data/processed_dataset.csv` and regenerates the model.
   This means: **whatever's in `processed_dataset.csv` in your repo is what
   trains the deployed model.** If you've been augmenting it locally (e.g.
   the small-talk augmentation script), make sure that version is the one
   committed, or the deployed model won't have those fixes.

## 1. MongoDB Atlas (replaces your local MongoDB)

Render/Railway can't reach `mongodb://localhost:27017` on your laptop --
you need a database reachable from the internet. Atlas's free tier (M0) is
enough for this project.

1. Go to https://www.mongodb.com/cloud/atlas/register and create a free account.
2. Create a new project, then "Build a Database" -> choose the **M0 Free**
   tier -> pick any cloud/region close to you -> Create.
3. **Database Access** (left sidebar) -> Add New Database User -> username +
   password (autogenerate is fine, just save it) -> "Read and write to any
   database" role.
4. **Network Access** (left sidebar) -> Add IP Address -> **Allow Access from
   Anywhere** (`0.0.0.0/0`). This is the simplest option for a student
   project; note it does mean anyone with the credentials can connect, so
   don't commit the connection string or password anywhere public.
5. Go back to your cluster -> "Connect" -> "Drivers" -> copy the connection
   string. It looks like:
   ```
   mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
   ```
   Replace `<username>` and `<password>` with the ones from step 3. This
   full string is your `MONGODB_URI`.

## 2. Seed the Atlas database once, from your own machine

Do this *before or after* deploying, doesn't matter -- but it needs to
happen at least once so an admin account exists (there's no signup form for
admins by design).

```bash
# From your project root, on your own machine:
export MONGODB_URI="mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
python3 seed.py
```

This creates `admin@ppd.com` / `admin123` and the two demo doctors
(pre-verified, pre-online) directly in Atlas. **Change the admin password**
once you can log in, or better, edit `seed.py`'s `admin123` before running
this if you're at all worried about it being guessed before you get to it.

## 3. Push to GitHub

Both Render and Railway deploy from a git repo. If you haven't already:

```bash
git init
git add .
git commit -m "Ready for deployment"
git remote add origin <your-repo-url>
git push -u origin main
```

Double check `.gitignore` is doing its job -- `git status` shouldn't show
`.env`, `*.joblib`, or `models/transformer/`.

## 4. Deploy on Render (recommended)

**Option A -- Blueprint (one click, uses the included `render.yaml`):**
1. Push this repo to GitHub if you haven't (step 3).
2. In the Render dashboard: "New +" -> "Blueprint" -> connect your repo.
   Render reads `render.yaml` and creates the service automatically.
3. It'll prompt you for the two secret env vars it can't read from the
   file (`PPD_APP_SECRET`, `MONGODB_URI`) -- enter them there.

**Option B -- Manual (if you'd rather see every setting):**
1. "New +" -> "Web Service" -> connect your repo.
2. **Runtime**: Python 3.
3. **Build Command**: `pip install -r requirements.txt && python3 models/train_baseline.py`
4. **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. **Instance Type**: Free.
6. **Environment variables** (Environment tab):
   - `MONGODB_URI` = your Atlas connection string from step 1
   - `PPD_APP_SECRET` = a long random string (e.g. run `python3 -c "import secrets; print(secrets.token_hex(32))"` locally and paste the output) -- **do not skip this**, the app will print a loud warning in its logs if you do, and it means anyone can forge login tokens.
   - `MONGODB_DB_NAME` = `ppd_app` (optional, this is already the default)
7. Create Web Service. First build will take a few minutes (installing
   scikit-learn, shap, etc. and training the baseline model).

**Free tier heads up:** Render's free web services spin down after 15
minutes of no traffic, and take 30-60 seconds to wake back up on the next
request. Fine for a demo/viva where you control the timing; just don't be
alarmed if the first load after a break is slow.

## 5. Deploy on Railway (alternative)

Railway auto-detects the `Procfile` in this repo, but does **not** run a
separate build command by default the way Render does -- you need to add
one manually or it'll try to start the app without ever training the model.

1. "New Project" -> "Deploy from GitHub repo" -> select this repo.
2. In the service's **Settings** -> **Deploy** section, set:
   - **Build Command**: `pip install -r requirements.txt && python3 models/train_baseline.py`
   - **Start Command**: (leave as default -- it'll use the `Procfile`)
3. In **Variables**, add `MONGODB_URI`, `PPD_APP_SECRET`, and optionally
   `MONGODB_DB_NAME`, same values as the Render instructions above.
4. Deploy. Railway's free tier works on a trial-credit system rather than
   an always-free tier like Render's -- check their current pricing page,
   since this has changed over time and you don't want to be surprised by
   a bill mid-semester.

## 6. After it's live

- Visit the deployed URL, log in as `admin@ppd.com`, confirm the two seeded
  demo doctors show up under Users.
- Log in as a demo doctor, confirm the Verification tab shows them as
  already verified (seeded that way) and the availability toggle works.
- Register a brand new mother and doctor account through the live UI to
  confirm the full signup -> verification -> booking flow works against
  the real Atlas database, not just against my local test mocks.
- Check the deploy logs once for the `PPD_APP_SECRET` warning -- if you see
  it, you forgot to set the environment variable and every login token is
  currently forgeable.

## Known limitations to mention if asked

- The transformer model (`models/transformer/`) is not included in this
  deploy -- the app runs on the TF-IDF baseline, same as your local setup,
  since the transformer needs a separate Colab run with GPU/internet access
  that this build pipeline doesn't do.
- Video calls use Jitsi Meet (a free third-party embed), not custom-built
  video infrastructure.
- Free-tier hosts cold-start after inactivity; there's no way around this
  without a paid tier.