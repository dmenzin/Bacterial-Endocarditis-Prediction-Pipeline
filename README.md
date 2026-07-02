# Bacterial Endocarditis Prediction from Prior ICD-9 History

This is my portfolio-facing clinical ML notebook for predicting bacterial endocarditis from prior ICD-9 history. It uses a back end helper .py file to simplify the readability of this notebook.

Open:

```text
endocarditis_prediction.ipynb
```

The notebook includes cohort design, leakage controls, temporal ICD features, model evaluation, and interpretation, while implementation details live in `be_icd_portfolio_helpers.py`.

## Design choices

**Task.** Predict a patient's first bacterial endocarditis diagnosis (ICD-9 `4210`) from their prior inpatient ICD-9 history in MIMIC-III. My design choices span cohort design, leakage control, and outcome evaluation rather than clinical deployment.

**Cohort.** Cases are patients with a first BE diagnosis with at least one admission prior to index admission; controls are inpatients who never had BE — any patient carrying the BE code or a BE-adjacent admission is excluded from the control pool entirely. Controls are balanced to the cases' prior-admission-count distribution, then the cohort is frozen to a saved ID list (`model_cohort_ids.csv`) so every downstream step runs against the same patients.

**Leakage control (the central design problem).** Three safeguards:

- **Pre-diagnosis restriction** — for each case, only admissions strictly before the first BE admission are used, so the model never sees the index event or anything after it.
- **Near-index scrubbing** — within a 42-day window before the index admission, valve-procedure and BE-workup codes are removed from case histories, preventing the model from keying on the diagnostic cascade that immediately precedes a BE diagnosis **due to suspicion for BE**.
- **Target removal** — the BE target code itself is stripped from admission histories during feature-building, so the outcome can never leak in as a predictor.

**Features.** Temporal diagnosis encoding — each ICD-9 diagnosis is prefixed by admission recency (`T-1`, `T-2`, ...) so the model sees *when* a code occurred relative to the index, with procedures one-hot encoded. Added to this are engineered prior-window clinical flags (e.g. cardiac devices, prosthetic valves, renal access), length of stay, and age. See *Feature-set control* below for the reduced-vs-full toggle.

**Models.** Three base learners trained per fold: XGBoost, Random Forest, and L1-regularized logistic regression (Lasso). RF and Lasso are trained on oversampled folds (`RandomOverSampler`, ratio 0.1); XGBoost handles imbalance via `scale_pos_weight`. A soft-vote and a stacked logistic meta-learner combine Random Forest + Lasso into ensembles.

**Evaluation.** Repeated cross-validation (seed 42): controls are split into non-overlapping partitions, each partition paired with all BE cases and run through 5-fold CV, so the rare positive class is reused across every control partition rather than thinned. Because the outcome is rare, precision–recall / average precision is reported alongside ROC-AUC, with calibration (Brier) and bootstrap confidence intervals. A Table-1-style descriptive (with p-values) and an events-per-feature check guard against an over-parameterized model. All three base models plus both ensembles appear in the metric tables.

**Interpretability.** SHAP is used as a model-audit view, not a causal claim. The notebook shows SHAP beeswarms for Random Forest and Lasso, each accompanied by a ranked top-driver table saved to CSV. (XGBoost is scored and evaluated but not SHAP-plotted.)

**Reproducibility.** Runs top-to-bottom (Kernel → Restart & Run All). All knobs live in the config cell. Raw MIMIC-III is credentialed and not included — place the required CSV/XLSX files in the working directory to run.

## Feature-set control

At the top of the notebook there is one simple setting:

```python
USE_REDUCED_FEATURE_SET = True
```

- `True` uses `top_reduced_BASE_features.csv`. These are the features that were previously selected using union feature selection of Chi-squared, RF, Lasso, XGBoost, and DT importances.
- `False` uses the full candidate feature matrix.

## Files

- `endocarditis_prediction.ipynb` — main portfolio notebook
- `be_icd_portfolio_helpers.py` — helper functions that keep the notebook readable
- `top_reduced_BASE_features.csv` — reduced feature list used when `USE_REDUCED_FEATURE_SET = True`
- `model_cohort_ids.csv` — frozen case/control subject IDs for a reproducible cohort
- `requirements.txt` — reproducibility dependencies
