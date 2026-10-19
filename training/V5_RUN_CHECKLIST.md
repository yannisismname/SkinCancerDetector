# Skin cancer v5 run — simple checklist
Prepared Mon 12 Oct 2026 by Klaus (background pre-flight check).
Notebook: `training/skin_cancer_training_v5.ipynb` — checked cell by cell today.

## One fix made before you run it
The very last cell used `json.dump(...)` to write `classes.json` but
`json` was never imported in the notebook — the same miss was in v3 and
v4. It would have crashed *after* the long training finished, right at
export: `model.h5` would be saved, `classes.json` would not.
Fixed today (12 Oct): `import json` added to the export cell, the stale
"EfficientNetB0 / round 4" and "MobileNetV2" headings corrected, and
every code cell syntax-checked. Nothing about how the model trains was
changed.

## Why this run, and why this week
- v5 is the accuracy push: softer class weights (~3.5x for rare classes
  instead of ~12x), stronger fine-tuning (last 120 layers, learning
  rate 3e-5, up to 40 rounds), BatchNorm still frozen (the round-2 fix).
- Best so far is still round 1: **66.7% test accuracy, macro F1 0.452**.
  Round 3 scored 63.7% with macro F1 0.492. v4 did not beat 66.7%.
- Honest expectation for v5: low-to-mid 70s. Not a promise.
- The IWK résumé (due Sun 18 Oct, 11:59pm) uses the 67% number. If v5
  wins and is checked before then, that number can be swapped in. If
  v5 does not win, 67% stays and nothing is lost.
- You deferred v5 yourself — run it on the day you picked; this week
  is study break (no classes), so there is spare time to re-run if
  Kaggle stops. It is a longer run than v4: plan roughly 2–3 hours
  (v4 was ~1.5h; v5 fine-tunes more layers for more rounds).

## Before you start (5 minutes)
1. Plug the laptop in and keep it awake — the run stops if it sleeps.
2. Open Kaggle, make a new notebook, upload
   `skin_cancer_training_v5.ipynb`.
3. Right panel → **+ Add Data** → search `skin-cancer-mnist-ham10000`
   → add it. (Skip this and the notebook stops early with
   "Dataset not found".)
4. Right panel → Settings → Accelerator → **GPU**. Check it is on
   before running anything — on CPU this would take many hours.

## Run it
5. Run all cells in order, top to bottom. Nothing to type or change.
   - First part (head training): up to 40 rounds, stops early on its
     own if it stops improving.
   - Second part (fine-tuning): up to 40 rounds, BatchNorm layers stay
     frozen — that is the fix that already worked in round 3.
6. At the end (section 10), screenshot or copy these:
   - **TEST ACCURACY: ...%**
   - **BALANCED ACCURACY: ...%** and **MACRO F1: ...** (v5 prints all
     three — round 1's macro F1 was 0.452)
   - the full per-class report (especially the melanoma row)
   Send those to Klaus — that is all he needs to judge the run.

## The decision rule (no guessing)
- **v5 beats 66.7% test accuracy** → v5 wins. Download `model.h5` +
  `classes.json` from Kaggle's Output panel into the project's
  `model/` folder, and swap the backend loader at the same time:
  `backend/model_loader_v4.py` (300px) becomes `model_loader.py` —
  the current loader is set to 224px and must never be paired with
  the 300px model. Klaus prepped the swap; it takes about 2 minutes
  once the files land. Then the IWK résumé number can be updated to
  the new verified score before 18 Oct.
- **v5 is 66.7% or lower** → round 1 stays the winner, with the
  test-time augmentation already in the backend. Nothing breaks, and
  the 67% on the IWK résumé stays exactly as it is. If v5's macro F1
  beats 0.492 while accuracy ties, send the numbers anyway — that is
  a judgement call to make with Klaus, not on your own mid-run.

## Watch out for
- Kaggle gives 30 free GPU hours per week — this run uses roughly
  2–3, so quota is fine unless the week was already used up.
- If one class in the final report has almost no correct guesses,
  still send the screenshot — do not re-run on your own. The
  notebook's own gate says not to export a model where one class
  takes over.
