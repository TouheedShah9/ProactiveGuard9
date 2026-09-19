# Engineering Notes

A record of the non-obvious problems found and fixed while turning
ProactiveGuard from a set of independently-working scripts into one
integrated system. Kept here deliberately, unedited in substance,
because *how* a bug got fixed is often more informative than the fact
that it did.

---

## 1. Config duplication across seven files

**Found**: the worker roster (names, resting HR, acclimatization days,
zone assignment) and site definitions were independently hand-typed
into `v2_data_engine.py`, `v2_llm_engine.py`, `v2_zone_engine.py`,
`v2_rl_scheduler.py`, `v2_safety_infrastructure.py`, and the
dashboard — six separate copies, plus one more found later. They had
already drifted: the dashboard's copy listed a role as `"Instrument
Tech"` while every other copy said `"Instrument Technician"`.

**Fix**: extracted into `v2_config.py` as the single source of truth.
Every other module imports from it. A change to one worker's data now
changes everywhere, because there's only one place for it to exist.

---

## 2. SHAP + XGBoost multi-class incompatibility (three attempts)

`shap.TreeExplainer` failed on the trained model with:
```
ValueError: could not convert string to float: '[0E0,0E0,0E0]'
```

Each fix attempt taught something real about *where* the bug actually
lived, rather than being three guesses at the same thing:

1. **Attempt 1** patched `booster.save_config()` — the wrong API
   entirely (booster hyperparameters, not the model's embedded
   `base_score`). Silently changed nothing.
2. **Attempt 2** patched the model dump directly and reloaded it.
   Confirmed the patch *was* applying — but XGBoost's C++ core
   unconditionally re-expands `base_score` into a per-class array for
   any multi-class model on `load_model()`, regardless of input. Not
   a bug to work around; the library's actual design.
3. **Attempt 3** monkey-patched `save_raw()` to intercept the exact
   bytes shap requests — and revealed shap actually requests a
   *binary* dump with its own internal decoder, not JSON text.

**Final fix**: stopped trying to make `shap.TreeExplainer` cooperate
with this XGBoost version at all. Used XGBoost's own native
`Booster.predict(..., pred_contribs=True)` — mathematically identical
SHAP values, zero cross-library serialization to break.

---

## 3. RL agent pickling across process boundaries

Training (`python v2_rl_scheduler.py`) makes that file `__main__`.
Pickling a custom class instance under `__main__` embeds a reference
to `__main__.QAgent`. Loading it from a *different* script (the
dashboard, via `streamlit run`) fails, because that script's own
`__main__` has no `QAgent` class:
```
AttributeError: Can't get attribute 'QAgent' on <module '__main__'
from '...v2_dashboard.py'>
```

The tempting fix — forcing `QAgent.__module__ = "v2_rl_scheduler"` —
was tested in isolation before shipping and found to break the
*training script's own save step* (forces a self-referential
re-import that produces a class object that doesn't match the
original, raising `PicklingError`).

**Final fix**: don't pickle the class at all. The agent's entire
learned state is its Q-table — a plain `numpy` array with no
class-identity concerns. Save only that; reconstruct a fresh `QAgent`
around it on load.

---

## 4. A model-quality bug hiding in a naming mismatch

Optuna's `trial.suggest_float("lr", ...)` names the *Optuna*
parameter `"lr"` for its own bookkeeping — a label with no required
relationship to the dict key it gets assigned to. Four of six tuned
hyperparameters used shortened Optuna names (`lr`, `sub`, `col`,
`mcw`) that didn't match XGBoost's actual constructor parameter names
(`learning_rate`, `subsample`, `colsample_bytree`,
`min_child_weight`).

Each individual trial during tuning still trained correctly (the
`objective()` function assigned values to the right keys internally).
But the **final production model**, built from `**study.best_params`,
silently received `lr=`, `sub=`, `col=`, `mcw=` as unrecognized
kwargs — XGBoost accepts unknown parameters without erroring, it just
doesn't use them. The reported tuning F1 and the deployed model's
actual behavior had quietly diverged.

**Fix**: renamed Optuna's parameter labels to match XGBoost's real
constructor arguments directly, so `study.best_params` produces a
dict the classifier actually uses correctly.

---

## 5. A metric that was measuring the wrong thing — twice

The digital twin's "acclimatization anomaly" check compared
heart-rate readings at minute 0, minute 200, and minute 420 —
*within a single 480-minute shift* — and labeled them "day 0," "day
7," "day 14." There is no multi-day data anywhere in this pipeline;
every worker's dataset is one shift. Since heart rate naturally rises
through any hot shift, this flagged **10 of 10 workers** as "poor
acclimatizers," including one with 180 days on site.

First fix relabeled the metric honestly (within-shift strain, not
acclimatization) — but the underlying percentage-of-early-shift-HR
calculation had a second, structural problem: for light-duty,
well-shaded, well-acclimatized workers, the denominator (early-shift
HR elevation) was naturally tiny, so *any* absolute rise produced a
disproportionate percentage. The single highest "strain" reading
ended up belonging to the safety inspector on light duty in full
shade — backwards.

**Final fix**: switched to peak HR deviation from personal resting
rate, reusing the same Karvonen-based threshold already established
elsewhere in the same file, instead of inventing a second ad hoc
metric.

---

## 6. Alert-rate methodology: counting minutes instead of episodes

The false-alarm-rate analysis counted every individual **minute** a
worker's risk stayed above threshold as a separate alert. A worker
elevated for 45 consecutive minutes counted as 45 alerts. Real
alerting systems fire once per episode and stay silent while risk
remains elevated. This inflated "alerts per shift" from a realistic
~6 to an implausible 73.7, which would have materially skewed
threshold-selection.

**Fix**: grouped contiguous above-threshold minutes per worker into
discrete episodes before counting. Verified against a synthetic test
case with a known correct answer before trusting it.

---

## 7. Fabricated demo data disguised as real output

Three separate places generated output that *looked* like it came
from the live pipeline but didn't:

- `v2_safety_infrastructure.py`'s "demo shift data" step logged
  hand-typed alert records (`score=82`, fixed driver strings)
  regardless of what the actual simulation produced.
- The same file's "model drift detection" compared against a
  hardcoded `baseline_f1 = 0.94` — bearing no relationship to the
  real trained model's actual F1 of 0.9941.
- `v2_llm_engine.py`'s demo function used a hand-typed vitals
  dictionary for "Samir Hassan," unconnected to
  `data/v2_twin_predictions.csv` despite the file's own docstring
  claiming that as an input.

**Fix**: all three now read the actual pipeline output — real
per-worker peak-risk moments, the real saved metrics report, and
(for the LLM demo) a live-computed SHAP explanation using the
actual trained model. The `run_demo()` output is different every
time you regenerate the dataset, because it's reading real,
regenerable data instead of returning a fixed string.

---

## 8. Screen-width detection was structurally non-functional

The dashboard's mobile-responsive logic (`is_mobile()`, column-count
adaptation) was already fully built — but depended on a JavaScript
bridge that sent `postMessage({type:'streamlit:setComponentValue'})`,
a message type only understood by Streamlit's custom-component
protocol. The plain `components.html()` call used here has no
receiving side for it. The message went nowhere, silently, on every
device — meaning the entire mobile layout system was dead code in
production; real phones always received the full desktop column
layout.

**Fix**: replaced with a working technique — a one-time URL
query-param round-trip (JS appends real viewport width, triggers a
single reload, Python reads it via `st.query_params`) — verified
against real phone screenshots.

A second-order bug surfaced from this fix: the forced reload wipes
session state on first mobile load, which meant sparkline chart
history buffers (which started empty) stayed empty long enough for
users to reliably see blank chart areas during their first minute on
the app. Fixed by pre-seeding history buffers instead of leaving them
empty, plus a "Collecting data…" fallback as a backstop.

---

## 9. A systematic re-audit found two more copies of the same bug class

After believing the config-duplication problem (item #1) was fully resolved,
a deliberate, systematic re-sweep of every file — not just the ones already
touched — found **two more** independent hardcoded worker rosters, in
`v2_digital_twin.py` and `v2_model.py`. Both had been missed during the
original consolidation pass because that pass worked bug-report-driven
(fix what surfaced), not exhaustively (grep every file for the pattern).

Same re-sweep found **two more instances** of the fabricated-demo-data
pattern (item #7) — in `v2_rl_scheduler.py`'s "Generating proactive
schedules" step and `v2_zone_engine.py`'s coordinated-risk-detection demo,
both using hardcoded (risk, WBGT, minutes) tuples that never changed
regardless of what the real pipeline produced. One of them was honest about
it in its own comment ("Simulate scenario…") — but still shipped as the
default behavior rather than reading the real output that was, by that
point, sitting right there in `data/v2_twin_predictions.csv`.

**Lesson applied**: fixed both the same way as their sibling instances
(import from `v2_config.py`; read real pipeline output with an honest,
clearly-labeled fallback if it doesn't exist yet) — and then re-ran the
exhaustive grep sweep a second time to confirm zero remaining instances of
either pattern, rather than trusting that "found some, fixed some" meant
"found all."

---

## 10. A duplicate RL lookup silently bypassed an already-tested fix

Item #3 documents fixing `QAgent.best_action()` to fall back to a
risk-graded heuristic when the Q-table has never seen a given state,
instead of trusting `argmax`'s meaningless tie-breaking default. That fix
was correct — but the live dashboard's per-tick RL panel called a
**second, separate implementation** (`get_rl_action()` in
`v2_dashboard.py`) that reimplemented the same state discretization and a
raw, unprotected `np.argmax()` inline, never routed through the fixed
method at all. Fixing the bug once didn't fix it everywhere it lived.

**Fix**: deleted the duplicate logic; `get_rl_action()` now calls
`rl_scheduler.discretize_state()` and `agent.best_action()` directly — the
same single, already-tested implementation used everywhere else. There is
now exactly one place this logic can be wrong, not two.

---

## 11. A live feature-vector approximation fed the model physically impossible inputs

The dashboard's live SHAP panel approximates the trained model's 100
engineered features from a single current snapshot (since replaying full
historical feature engineering on every UI tick isn't practical). The
approximation filled every lag/rolling-mean/rate-of-change/rolling-std
variant of a signal with that signal's raw current value.

For lag and rolling-mean features (same units as the raw signal) that's a
reasonable flat-trend approximation. But **rate-of-change** and
**rolling-volatility** features are a different scale entirely — e.g.
`heart_rate_bpm_slope5` represents `diff(5)/5`, a small per-minute rate
typically around 0.5–2, not the raw heart rate itself. Filling it with a
raw value like 118 fed the model a "118 bpm-per-minute rate of change" —
physically impossible, and after scaling, an extreme outlier far outside
anything seen in training. This is the most likely explanation for
counterintuitive SHAP directions observed in testing (e.g. an elevated
heart rate showing as *decreasing* danger risk).

**Fix**: rate-of-change and rolling-volatility features now correctly
default to 0 (no recent change / no recent volatility) instead of the raw
signal value; only same-unit lag/rolling-mean features use the flat-trend
approximation.

---

## What changed about the process after this round

The first several rounds of bug-fixing were reactive: something looked
wrong in a screenshot, trace it, fix it. That's necessary but not
sufficient — it only catches what happens to get looked at. Round two was
different in kind: systematically grep every file for each bug's *pattern*
(not just its one known instance) after fixing it once, on the assumption
that if a mistake happened once in a codebase built this way, it probably
happened more than once. It had — four more times, across three files that
had never come up in any screenshot.


---

## What this file is for

Not every fix here mattered equally to a user clicking through the
dashboard. Some mattered a great deal (the model-parameter mismatch
changed what actually got deployed; the fabricated demo data would
have been an honesty problem in an interview). The point of keeping
this log isn't to claim the code is now perfect — it's that the
process of finding these was systematic: run it for real, read the
actual error, verify the fix against the real behavior rather than
assuming it worked, and re-check for the same mistake elsewhere
rather than assuming one fix meant the pattern was gone.
