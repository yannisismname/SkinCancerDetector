# Skin cancer v4 run — simple checklist for Mon Oct 12
Prepared Sun 11 Oct 2026 (evening) by Klaus, background research.
Notebook: `training/skin_cancer_training_v4.ipynb` — checked cell by cell tonight.

## Why Monday
- No classes Mon Oct 12 (Thanksgiving) or all study-break week.
- This is the last training round. Round 1 is still the best so far:
  66.7% test accuracy. Your job applications are parked until the
  model is as good as it can get, and the IWK résumé (due Sat/Sun
  Oct 17–18) uses the 67% number — if v4 beats it and is checked
  before then, that number can be swapped in. If it does not beat
  it, 67% stays and nothing is lost.
- Running it Monday leaves Tue–Fri to redo it if Kaggle stops or
  something breaks. Running it later in the week leaves no spare day.

## Before you start (5 minutes)
1. Plug the laptop in and keep it awake — the run takes about
   1.5 hours and stops if the laptop sleeps.
2. Open Kaggle, make a new notebook, upload
   `skin_cancer_training_v4.ipynb`.
3. Right panel → **+ Add Data** → search `skin-cancer-mnist-ham10000`
   → add it. (Skip this and the notebook stops early with
   "Dataset not found".)
4. Right panel → Settings → Accelerator → **GPU**. Check it is on
   before running anything — on CPU this would take many hours.

## Run it
5. Run all cells in order, top to bottom. Nothing to type or change.
   - First part (head training): up to 40 rounds, stops early on
     its own if it stops improving.
   - Second part (fine-tuning): up to 25 rounds. This is the part
     that broke in round 2 — in v4 the sensitive layers stay frozen,
     which is the fix that already worked in round 3.
6. At the end (section 10), take a screenshot or copy these 3 things:
   - the line that says **TEST ACCURACY: ...%**
   - the full per-class report above it (especially the melanoma row)
   - the confusion matrix picture, if it shows
   Send those to Klaus — that is all he needs to judge the run.

## The decision rule (already agreed — no guessing)
- **v4 beats 66.7%** → v4 wins. Download `model.h5` + `classes.json`
  from Kaggle's Output panel into the project's `model/` folder.
  They MUST move together with the new backend loader
  (`backend/model_loader_v4.py`, which uses the bigger 300px
  pictures) — never swap the model without the loader, or the
  other way round. Klaus prepped the swap kit; swapping takes
  about 2 minutes once the files land.
- **v4 is 66.7% or lower** → round 1 stays the winner, with the
  test-time augmentation already added to the backend. Nothing
  breaks, nothing needs redoing, and the 67% on the IWK résumé
  stays exactly as it is.

## Watch out for
- Kaggle gives 30 free GPU hours per week — this run uses about
  1.5, so quota is not a problem unless the week was already used up.
- If a class in the final report has almost no correct guesses,
  still send the screenshot — do not re-run on your own. The
  notebook's own gate says not to export a model where one class
  takes over, and the numbers decide that, not a feeling.
