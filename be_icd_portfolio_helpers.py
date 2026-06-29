"""
Portfolio helper functions for the bacterial-endocarditis ICD-history project.

The notebook should communicate the design choices; this module holds the bulky
implementation details: MIMIC table joins, leakage scrubs, feature construction,
model training, SHAP aggregation, and plotting helpers.
"""

from __future__ import annotations

import json
import os
import platform
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import imblearn
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import shap
import sklearn
import xgboost
from imblearn.over_sampling import RandomOverSampler
from scipy.sparse import csr_matrix, hstack
from scipy.stats import chi2_contingency, mannwhitneyu
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    auc,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.preprocessing import MaxAbsScaler, MultiLabelBinarizer
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------

OUTPUT_ROOT = Path("model_outputs")
FIGURE_DIR = OUTPUT_ROOT / "figures"
DATA_DIR = OUTPUT_ROOT / "data"
FEATURE_IMPORTANCE_DIR = DATA_DIR / "feature_importance"

_FEATURE_IMPORTANCE_MARKERS = (
    "importance",
    "agreement",
    "cumulative",
    "shap_feature",
    "temporal_feature",
)


def configure_output_dirs(output_root: str | Path = "model_outputs") -> Mapping[str, Path]:
    """Create standard output folders and return their paths."""
    global OUTPUT_ROOT, FIGURE_DIR, DATA_DIR, FEATURE_IMPORTANCE_DIR
    OUTPUT_ROOT = Path(output_root)
    FIGURE_DIR = OUTPUT_ROOT / "figures"
    DATA_DIR = OUTPUT_ROOT / "data"
    FEATURE_IMPORTANCE_DIR = DATA_DIR / "feature_importance"
    for output_dir in (FIGURE_DIR, DATA_DIR, FEATURE_IMPORTANCE_DIR):
        output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "output_root": OUTPUT_ROOT,
        "figure_dir": FIGURE_DIR,
        "data_dir": DATA_DIR,
        "feature_importance_dir": FEATURE_IMPORTANCE_DIR,
    }


def _route_output_path(path, kind):
    path = Path(path)
    if path.is_absolute() or path.parent != Path("."):
        return path
    if kind == "fig":
        return FIGURE_DIR / path.name
    if path.suffix.lower() == ".csv" and any(
        marker in path.name.lower() for marker in _FEATURE_IMPORTANCE_MARKERS
    ):
        return FEATURE_IMPORTANCE_DIR / path.name
    return DATA_DIR / path.name


def save_fig(filename, *args, **kwargs):
    """Save a matplotlib figure into the standard figures directory."""
    output_path = _route_output_path(filename, kind="fig")
    plt.savefig(output_path, *args, **kwargs)
    return output_path


def save_csv(df, filename, *args, **kwargs):
    """Save a dataframe into the standard data directory."""
    output_path = _route_output_path(filename, kind="csv")
    df.to_csv(output_path, *args, **kwargs)
    return output_path


# -----------------------------------------------------------------------------
# Clinical prefix groups used by engineered features
# -----------------------------------------------------------------------------

WINDOW_START_DAYS = 395
WINDOW_GAP_DAYS = 0
MIN_DAYS_BEFORE = max(
    1, WINDOW_GAP_DAYS
)
CARDIAC_PREFIXES = [  # CHF + cardiac arrhythmia (Elixhauser); valvular excluded
    "39891",
    "40201",
    "40211",
    "40291",
    "40401",
    "40403",
    "40411",
    "40413",
    "40491",
    "40493",
    "4254",
    "4255",
    "4257",
    "4258",
    "4259",
    "428",
    "4280",
    "4281",
    "4282",
    "42820",
    "42821",
    "42822",
    "42823",
    "42830",
    "42831",
    "42832",
    "42833",
    "42840",
    "42841",
    "42842",
    "42843",
    "4289",
    "4260",
    "42610",
    "42611",
    "42612",
    "42613",
    "4267",
    "42650",
    "42651",
    "42652",
    "42653",
    "42654",
    "4269",
    "4270",
    "4271",
    "4272",
    "42731",
    "42732",
    "4273",
    "42741",
    "42742",
    "4275",
    "4276",
    "42760",
    "42761",
    "42769",
    "42781",
    "42789",
    "4279",
    "7850",
    "7851",
]
VASCULAR_PREFIXES = [  # peripheral vascular + pulmonary circulation (Elixhauser)
    "0930",
    "4373",
    "440",
    "4400",
    "4401",
    "44020",
    "44021",
    "44022",
    "44023",
    "44024",
    "44029",
    "44030",
    "44031",
    "44032",
    "4408",
    "4409",
    "441",
    "4410",
    "4411",
    "4412",
    "4413",
    "4414",
    "4415",
    "4416",
    "4417",
    "4419",
    "4421",
    "4423",
    "44281",
    "44282",
    "44283",
    "44284",
    "44289",
    "4429",
    "4431",
    "4432",
    "4433",
    "44381",
    "44389",
    "4439",
    "4471",
    "5571",
    "5579",
    "V434",
    "4150",
    "41511",
    "41512",
    "41513",
    "41519",
    "416",
    "4160",
    "4161",
    "4162",
    "4168",
    "4169",
    "4170",
    "4178",
    "4179",
]
DIABETES_COMP_PREFIXES = [  # COMPLICATED diabetes (end-organ; Elixhauser DMcx)
    "2504",
    "25040",
    "25041",
    "25042",
    "25043",
    "2505",
    "25050",
    "25051",
    "25052",
    "25053",
    "2506",
    "25060",
    "25061",
    "25062",
    "25063",
    "2507",
    "25070",
    "25071",
    "25072",
    "25073",
    "2508",
    "25080",
    "25081",
    "25082",
    "25083",
    "2509",
    "25090",
    "25091",
    "25092",
    "25093",
    "3572",
    "36201",
    "36202",
    "36203",
    "36204",
    "36205",
    "36206",
    "36641",
]
LIVER_PREFIXES = [  # chronic liver disease / cirrhosis (Elixhauser)
    "07022",
    "07023",
    "07032",
    "07033",
    "07044",
    "07054",
    "0706",
    "0709",
    "4560",
    "4561",
    "45620",
    "45621",
    "570",
    "571",
    "5710",
    "5711",
    "5712",
    "5713",
    "5714",
    "57140",
    "57141",
    "57142",
    "57149",
    "5715",
    "5716",
    "5718",
    "5719",
    "5723",
    "5724",
    "5728",
    "V427",
]
VALVULAR_PREFIXES = [  # rheumatic + non-rheumatic valve disease substrate; excludes BE 421x
    "394",
    "395",
    "396",
    "397",
    "4240",
    "4241",
    "4242",
    "4243",
    "42490",
    "42491",
    "42499",
]
CONGENITAL_PREFIXES = [  # congenital heart / circulatory anomalies; interpret 7455 cautiously in adults
    "745",
    "746",
    "747",
]
RENAL_ACCESS_PREFIXES = [  # CKD/ESRD + dialysis status/access
    "585",
    "5851",
    "5852",
    "5853",
    "5854",
    "5855",
    "5856",
    "5859",
    "586",
    "V451",
    "V4511",
    "V4512",
    "V56",
    "V560",
    "V561",
    "V562",
    "V568",
    "40301",
    "40311",
    "40391",
    "40402",
    "40403",
    "40412",
    "40413",
    "40492",
    "40493",
    "99673",
    "99674",
    "99656",
]
DEVICE_PREFIXES = [  # cardiac/intravascular device in situ or device complication status (DX)
    "V450",
    "V4500",
    "V4501",
    "V4502",
    "V4509",
    "V533",
    "V5331",
    "V5332",
    "V5339",
    "99601",
    "99602",
    "99604",
    "9961",
    "99661",
    "99662",
    "99671",
    "99672",
    "99931",
    "99932",
    "99933",
]
PROSTHETIC_VALVE_DX_PREFIXES = [  # prosthetic heart valve / valve transplant / valve-prosthesis complication (DX)
    "V422",
    "V433",
    "99602",
    "99661",
    "99671",
]
IVDU_PREFIXES = [  # drug dependence / abuse (Elixhauser drug abuse)
    "3040",
    "30400",
    "30401",
    "30402",
    "30403",
    "3041",
    "3042",
    "3044",
    "3045",
    "3047",
    "3048",
    "3049",
    "3055",
    "30550",
    "30551",
    "30552",
    "30553",
    "3056",
    "30560",
    "3057",
    "3058",
    "3059",
    "6483",
    "6484",
    "65550",
    "V6542",
]
VASCULAR_DEVICE_PROC_PREFIXES = [  # vascular access / intravascular device implantation (PROCEDURES)
    "3891",
    "3892",
    "3893",
    "3894",
    "3895",
    "3897",
    "3898",
    "3899",
    "3927",
    "3928",
    "3929",
    "3990",
    "3993",
    "3994",
    "3995",
    "3996",
    "3997",
    "3998",
    "3999",
    "8607",
    "3491",
    "0050",
    "0051",
    "0052",
    "0053",
    "0054",
    "3770",
    "3771",
    "3772",
    "3773",
    "3774",
    "3775",
    "3776",
    "3777",
    "3778",
    "3779",
    "3780",
    "3781",
    "3782",
    "3783",
    "3785",
    "3786",
    "3787",
    "3789",
    "3794",
    "3795",
    "3796",
    "3797",
    "3798",
]
HEART_VALVE_PROC_PREFIXES = [  # valve repair/replacement/prosthetic valve procedures (PROCEDURES)
    "3500",
    "3501",
    "3502",
    "3503",
    "3504",
    "3505",
    "3506",
    "3507",
    "3508",
    "3509",
    "3510",
    "3511",
    "3512",
    "3513",
    "3514",
    "3520",
    "3521",
    "3522",
    "3523",
    "3524",
    "3525",
    "3526",
    "3527",
    "3528",
    "3531",
    "3532",
    "3533",
    "3534",
    "3535",
]



def capture_environment_versions() -> pd.DataFrame:
    """Return the package versions used to generate the notebook outputs."""
    library_versions = [
        ("numpy", np.__version__),
        ("pandas", pd.__version__),
        ("scipy", scipy.__version__),
        ("scikit-learn", sklearn.__version__),
        ("xgboost", xgboost.__version__),
        ("imbalanced-learn", imblearn.__version__),
        ("shap", shap.__version__),
    ]
    return pd.DataFrame(
        [("Python", platform.python_version())] + library_versions,
        columns=["Package", "Version"],
    )


def load_mimic_tables(data_dir: str | Path = ".") -> Dict[str, pd.DataFrame]:
    """Load the MIMIC-III tables required by the diagnosis-history pipeline."""
    data_dir = Path(data_dir)
    return {
        "PATIENTS": pd.read_csv(data_dir / "PATIENTS.csv"),
        "ADMISSIONS": pd.read_csv(data_dir / "ADMISSIONS.csv"),
        "DIAGNOSES_ICD": pd.read_csv(data_dir / "DIAGNOSES_ICD.csv"),
        "PROCEDURES_ICD": pd.read_csv(data_dir / "PROCEDURES_ICD.csv"),
        "ICD_9_DX_DESCRIPTIONS": pd.read_excel(data_dir / "CMS32_DESC_LONG_SHORT_DX.xlsx", dtype=str),
        "ICD_9_SG_DESCRIPTIONS": pd.read_excel(data_dir / "CMS32_DESC_LONG_SHORT_SG.xlsx", dtype=str),
    }


def initialize_analysis_state(
    tables: Mapping[str, pd.DataFrame],
    random_seed: int = 42,
    valve_proc_gap_days: int = 42,
    workup_gap_days: int = 42,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    """Set reproducibility and leakage-scrub parameters; summarize code universes."""
    np.random.seed(random_seed)
    patients = tables["PATIENTS"].copy()
    patient_gender = patients[["SUBJECT_ID", "GENDER"]].copy()
    patients_dob = patients[["SUBJECT_ID", "DOB"]].copy()
    patients_dob["DOB"] = pd.to_datetime(patients_dob["DOB"], errors="coerce")

    config = {
        "random_seed": random_seed,
        "target_icd9_codes": ["4210"],
        "dx_codes": set(tables["DIAGNOSES_ICD"]["ICD9_CODE"].astype(str).str.strip()),
        "pr_codes": set(tables["PROCEDURES_ICD"]["ICD9_CODE"].astype(str).str.strip()),
        "drop_valve_procedures": False,
        "valve_procedure_codes": {"3521", "3522", "V433", "3961"},
        "valve_proc_gap_days": valve_proc_gap_days,
        "ablate_workup_in_window": True,
        "workup_gap_days": workup_gap_days,
        "workup_codes": {"8872", "0389", "78552"},
        "patients_dob": patients_dob,
        "patient_gender": patient_gender,
    }

    summary = pd.DataFrame(
        [
            {"Item": "Diagnosis-code universe", "Value": f"{len(config['dx_codes']):,} codes"},
            {"Item": "Procedure-code universe", "Value": f"{len(config['pr_codes']):,} codes"},
            {"Item": "Valve timing scrub", "Value": f"case admissions within {valve_proc_gap_days} days of index"},
            {"Item": "Duke/workup scrub", "Value": f"case admissions within {workup_gap_days} days of index"},
            {"Item": "Random seed", "Value": random_seed},
        ]
    )
    return config, summary


def _build_icd9_full_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ICD9_FULL"] = df["ICD9_CODE"]
    mask = (df["Description"].notna()) & (df["Description"] != "")
    df.loc[mask, "ICD9_FULL"] = df.loc[mask, "ICD9_CODE"] + ": " + df.loc[mask, "Description"]
    return df.drop(columns=["ICD9_REF", "Description"])


def attach_diagnosis_descriptions(df: pd.DataFrame, dx_table: pd.DataFrame) -> pd.DataFrame:
    """Merge diagnosis code descriptions and build `ICD9_FULL`."""
    merged = df.merge(
        dx_table[["DIAGNOSIS CODE", "LONG DESCRIPTION"]].rename(
            columns={"DIAGNOSIS CODE": "ICD9_REF", "LONG DESCRIPTION": "Description"}
        ),
        left_on="ICD9_CODE",
        right_on="ICD9_REF",
        how="left",
    )
    return _build_icd9_full_and_clean(merged)


def attach_procedure_descriptions(df: pd.DataFrame, sg_table: pd.DataFrame) -> pd.DataFrame:
    """Merge procedure code descriptions and build `ICD9_FULL`."""
    merged = df.merge(
        sg_table[["PROCEDURE CODE", "LONG DESCRIPTION"]].rename(
            columns={"PROCEDURE CODE": "ICD9_REF", "LONG DESCRIPTION": "Description"}
        ),
        left_on="ICD9_CODE",
        right_on="ICD9_REF",
        how="left",
    )
    return _build_icd9_full_and_clean(merged)


def define_be_case_control_ids(
    tables: Mapping[str, pd.DataFrame], config: Mapping[str, object]
) -> Tuple[Dict[str, object], Dict[str, pd.DataFrame]]:
    """Define target/control subject IDs and normalize ICD description tables."""
    diagnoses = tables["DIAGNOSES_ICD"].dropna(subset=["ICD9_CODE"]).copy()
    procedures = tables["PROCEDURES_ICD"].dropna(subset=["ICD9_CODE"]).copy()
    admissions = tables["ADMISSIONS"].copy()
    dx_desc = tables["ICD_9_DX_DESCRIPTIONS"].copy()
    sg_desc = tables["ICD_9_SG_DESCRIPTIONS"].copy()

    target_codes = list(config["target_icd9_codes"])
    procedures["ICD9_CODE"] = procedures["ICD9_CODE"].astype(str).str.zfill(4)
    sg_desc["PROCEDURE CODE"] = sg_desc["PROCEDURE CODE"].astype(str).str.zfill(4)

    diagnoses = attach_diagnosis_descriptions(diagnoses, dx_desc)
    procedures = attach_procedure_descriptions(procedures, sg_desc)

    disease_subject_ids = diagnoses.loc[
        diagnoses["ICD9_CODE"].isin(target_codes), "SUBJECT_ID"
    ].unique()
    disease_text_ids = admissions[
        admissions["DIAGNOSIS"].str.contains("BACTERIAL ENDOCARDITIS", na=False)
    ]["SUBJECT_ID"].unique()
    exclude_from_controls = np.union1d(disease_subject_ids, disease_text_ids)
    control_subject_ids = diagnoses.loc[
        ~diagnoses["SUBJECT_ID"].isin(exclude_from_controls), "SUBJECT_ID"
    ].unique()
    disease_hadm_ids = diagnoses.loc[
        diagnoses["ICD9_CODE"].isin(target_codes), "HADM_ID"
    ].unique()

    cohort = {
        "disease_subject_ids": disease_subject_ids,
        "control_subject_ids": control_subject_ids,
        "disease_hadm_ids": disease_hadm_ids,
        "target_icd9_codes": target_codes,
    }
    normalized_tables = dict(tables)
    normalized_tables.update({
        "DIAGNOSES_ICD": diagnoses,
        "PROCEDURES_ICD": procedures,
        "PATIENTS": config["patients_dob"],
    })
    return cohort, normalized_tables


def compress_codes(
    df: pd.DataFrame, subject_ids: Sequence[int], col_name: str, target_codes: Sequence[str]
) -> pd.DataFrame:
    """Collapse all codes for an admission into a single sorted list."""
    filtered = df[df["SUBJECT_ID"].isin(subject_ids)]
    if col_name == "DIAGNOSES":
        filtered = filtered[~filtered["ICD9_CODE"].isin(target_codes)]
    return (
        filtered.sort_values(["HADM_ID", "SEQ_NUM"])
        .groupby(["SUBJECT_ID", "HADM_ID"])["ICD9_FULL"]
        .apply(list)
        .reset_index(name=col_name)
    )


def merge_into_admissions(
    subject_ids: Sequence[int],
    diag_table: pd.DataFrame,
    proc_table: pd.DataFrame,
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
) -> pd.DataFrame:
    """Build an admission-level master table: ADMISSIONS + diagnosis/procedure lists + DOB."""
    return (
        admissions[admissions["SUBJECT_ID"].isin(subject_ids)]
        .merge(diag_table, on=["HADM_ID", "SUBJECT_ID"], how="left")
        .merge(proc_table, on=["HADM_ID", "SUBJECT_ID"], how="left")
        .merge(patients, on="SUBJECT_ID", how="left")
    )


def tidy_admissions_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rename label/code columns, drop ROW_ID if present, and parse ADMITTIME."""
    out = df.copy()
    out.rename(
        columns={"DIAGNOSIS": "DIAGNOSIS (LABEL)", "DIAGNOSES": "DIAGNOSIS (ICD_9)"},
        inplace=True,
    )
    if "ROW_ID" in out.columns:
        out.drop(["ROW_ID"], axis=1, inplace=True)
    out["ADMITTIME"] = pd.to_datetime(out["ADMITTIME"], errors="coerce")
    return out


def build_admission_level_history_tables(
    tables: Mapping[str, pd.DataFrame], cohort: Mapping[str, object], config: Mapping[str, object]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter ICD codes, compress admission-level code lists, and merge to admissions."""
    diagnoses = tables["DIAGNOSES_ICD"].copy()
    procedures = tables["PROCEDURES_ICD"].copy()
    diagnoses = diagnoses[diagnoses["ICD9_CODE"].isin(config["dx_codes"])]
    procedures = procedures[procedures["ICD9_CODE"].isin(config["pr_codes"])]
    diagnoses = diagnoses.groupby("ICD9_CODE").filter(lambda x: x["SUBJECT_ID"].nunique() > 7)

    disease_ids = cohort["disease_subject_ids"]
    control_ids = cohort["control_subject_ids"]
    target_codes = cohort["target_icd9_codes"]

    patient_diagnoses = compress_codes(diagnoses, disease_ids, "DIAGNOSES", target_codes)
    control_diagnoses = compress_codes(diagnoses, control_ids, "DIAGNOSES", target_codes)
    patient_procedures = compress_codes(procedures, disease_ids, "PROCEDURE TYPE", target_codes)
    control_procedures = compress_codes(procedures, control_ids, "PROCEDURE TYPE", target_codes)

    patient_admissions = merge_into_admissions(
        disease_ids, patient_diagnoses, patient_procedures, tables["ADMISSIONS"], tables["PATIENTS"]
    )
    control_admissions = merge_into_admissions(
        control_ids, control_diagnoses, control_procedures, tables["ADMISSIONS"], tables["PATIENTS"]
    )
    return tidy_admissions_table(patient_admissions), tidy_admissions_table(control_admissions), diagnoses


def _strip_codes_from_list(codes, strip_set: set):
    if not isinstance(codes, list):
        return codes
    return [c for c in codes if str(c).replace(".", "").strip().upper() not in strip_set]


def _count_codes_in_list(codes, strip_set: set) -> int:
    if not isinstance(codes, list):
        return 0
    return sum(str(c).replace(".", "").strip().upper() in strip_set for c in codes)


def scrub_case_history_codes_near_index(
    out: pd.DataFrame,
    comparator_col: str,
    code_set: set,
    gap_days: int,
) -> Tuple[pd.DataFrame, int]:
    """Remove specified diagnosis/procedure codes from case admissions close to index."""
    out = out.copy()
    days_before = (out[comparator_col] - out["ADMITTIME"]).dt.days
    in_window = days_before <= gap_days
    n_scrubbed = 0
    for col in ["DIAGNOSIS (ICD_9)", "PROCEDURE TYPE"]:
        if col not in out.columns:
            continue
        mask = in_window & out[col].apply(lambda x: isinstance(x, list))
        n_scrubbed += int(out.loc[mask, col].apply(lambda codes: _count_codes_in_list(codes, code_set)).sum())
        out.loc[mask, col] = out.loc[mask, col].apply(lambda codes: _strip_codes_from_list(codes, code_set))
    return out, n_scrubbed


def restrict_cases_to_pre_diagnosis_history(
    patient_admissions: pd.DataFrame,
    disease_hadm_ids: Sequence[int],
    config: Mapping[str, object],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Keep only admissions before first BE admission and apply leakage scrubs."""
    disease_first_admissions = (
        patient_admissions[patient_admissions["HADM_ID"].isin(disease_hadm_ids)]
        .groupby("SUBJECT_ID", as_index=False)["ADMITTIME"]
        .min()
        .rename(columns={"ADMITTIME": "Comparator"})
    )
    out = patient_admissions.merge(disease_first_admissions, on="SUBJECT_ID", how="left")
    out = out[out["ADMITTIME"] < out["Comparator"]].copy()

    valve_count = 0
    workup_count = 0
    if not config.get("drop_valve_procedures", False):
        out, valve_count = scrub_case_history_codes_near_index(
            out,
            "Comparator",
            set(config.get("valve_procedure_codes", set())),
            int(config.get("valve_proc_gap_days", 42)),
        )
    if config.get("ablate_workup_in_window", False):
        out, workup_count = scrub_case_history_codes_near_index(
            out,
            "Comparator",
            set(config.get("workup_codes", set())),
            int(config.get("workup_gap_days", 42)),
        )

    out.drop(["Comparator"], axis=1, inplace=True)
    out = add_event_index(out)
    diseased_counts = out.groupby("SUBJECT_ID")["HADM_ID"].nunique()
    summary = pd.DataFrame(
        [
            {"Check": "Case admissions retained before BE index", "Value": f"{out.shape[0]:,} rows"},
            {"Check": "Cases with 1 prior admission", "Value": int((diseased_counts == 1).sum())},
            {"Check": "Cases with 2+ prior admissions", "Value": int((diseased_counts >= 2).sum())},
            {"Check": "Valve/treatment code occurrences scrubbed", "Value": valve_count},
            {"Check": "Duke/workup code occurrences scrubbed", "Value": workup_count},
        ]
    )
    return out, summary, diseased_counts


def add_event_index(df: pd.DataFrame) -> pd.DataFrame:
    """Sort newest-first within patient and label admissions T-1, T-2, ..."""
    out = df.copy()
    out.sort_values(["SUBJECT_ID", "ADMITTIME"], ascending=[True, False], inplace=True)
    out["event_index"] = (out.groupby("SUBJECT_ID").cumcount() + 1) * -1
    return out


def balance_controls_to_case_admission_distribution(
    control_admissions: pd.DataFrame,
    diseased_counts: pd.Series,
    disease_subject_ids: Sequence[int],
    random_seed: int = 42,
    cohort_path: str | Path = "model_cohort_ids.csv",
) -> Tuple[np.ndarray, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Subsample controls to match case admission-count structure and freeze the draw."""
    np.random.seed(random_seed)
    cohort_path = Path(cohort_path)
    control_counts = control_admissions.groupby("SUBJECT_ID")["HADM_ID"].nunique()

    if cohort_path.exists():
        frozen = pd.read_csv(cohort_path)
        new_control_ids = frozen.loc[frozen["label"] == 0, "SUBJECT_ID"].unique()
        draw_status = "reused frozen cohort"
    else:
        diseased_1 = (diseased_counts == 1).sum()
        diseased_2 = (diseased_counts == 2).sum()
        num_to_keep = round(diseased_1 / diseased_2 * (control_counts == 2).sum())
        c_ids_1 = (
            control_admissions.groupby("SUBJECT_ID")
            .filter(lambda x: x["HADM_ID"].nunique() == 1)["SUBJECT_ID"]
            .unique()
        )
        sampled_c1_ids = np.random.choice(c_ids_1, size=num_to_keep, replace=False)
        c_ids_multi = control_counts[(control_counts >= 2) & (control_counts < 9)].index
        new_control_ids = np.concatenate([sampled_c1_ids, c_ids_multi])
        cohort_ids = pd.concat(
            [
                pd.DataFrame({"SUBJECT_ID": np.asarray(disease_subject_ids, dtype=int), "label": 1}),
                pd.DataFrame({"SUBJECT_ID": np.asarray(new_control_ids, dtype=int), "label": 0}),
            ],
            ignore_index=True,
        ).drop_duplicates(["SUBJECT_ID", "label"])
        cohort_ids.sort_values(["label", "SUBJECT_ID"], ascending=[False, True]).to_csv(cohort_path, index=False)
        draw_status = "sampled matched controls"

    filtered = control_admissions[control_admissions["SUBJECT_ID"].isin(new_control_ids)].copy()
    filtered = add_event_index(filtered)
    filtered_control_counts = filtered.groupby("SUBJECT_ID")["HADM_ID"].nunique()
    cohort_summary = pd.DataFrame(
        [
            {"Item": "Control balancing", "Value": draw_status},
            {"Item": "Control admission rows", "Value": f"{filtered.shape[0]:,}"},
            {"Item": "Control patients", "Value": f"{len(new_control_ids):,}"},
        ]
    )
    frozen = pd.read_csv(cohort_path)
    freeze_summary = pd.DataFrame(
        [
            {
                "Status": "Cohort freeze present",
                "Path": str(cohort_path),
                "Cases": int((frozen.label == 1).sum()),
                "Controls": int((frozen.label == 0).sum()),
            }
        ]
    )
    return new_control_ids, filtered, control_counts, filtered_control_counts, cohort_summary, freeze_summary


def assert_cohort_matches_freeze(
    disease_subject_ids: Sequence[int],
    control_subject_ids: Sequence[int],
    cohort_path: str | Path = "model_cohort_ids.csv",
) -> pd.DataFrame:
    """Assert active case/control IDs exactly match the cohort-freeze file."""
    frozen = pd.read_csv(cohort_path)
    frozen_cases = set(frozen.loc[frozen.label == 1, "SUBJECT_ID"])
    frozen_controls = set(frozen.loc[frozen.label == 0, "SUBJECT_ID"])
    here_cases = set(np.asarray(disease_subject_ids).tolist())
    here_controls = set(np.asarray(control_subject_ids).tolist())
    assert here_cases == frozen_cases, f"case mismatch vs {cohort_path}"
    assert here_controls == frozen_controls, f"control mismatch vs {cohort_path}"
    return pd.DataFrame(
        [{"Guard": "Frozen cohort matches notebook cohort", "Cases": len(here_cases), "Controls": len(here_controls)}]
    )


def preprocess_admission_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, int, List[str]]:
    """Compute LOS/Age, remove invalid rows, cap de-identified ages, and drop leaky columns."""
    out = df.copy()
    out["ADMITTIME"] = pd.to_datetime(out["ADMITTIME"])
    out["DISCHTIME"] = pd.to_datetime(out["DISCHTIME"])
    out["LOS_DAYS"] = round((out["DISCHTIME"] - out["ADMITTIME"]).dt.total_seconds() / 86400)
    out["Age"] = (out["ADMITTIME"] - out["DOB"]).dt.days // 365
    initial_count = len(out)
    out = out[(out["LOS_DAYS"] >= 0) & (out["Age"] > 0)].copy()
    dropped_erroneous = initial_count - len(out)
    out.loc[out["Age"] > 100, "Age"] = 92
    leaky_columns = [
        "HADM_ID", "ADMITTIME", "DISCHTIME", "DEATHTIME", "EDREGTIME", "EDOUTTIME",
        "HOSPITAL_EXPIRE_FLAG", "DIAGNOSIS (LABEL)", "HAS_CHARTEVENTS_DATA", "DOB",
    ]
    existing_drops = [c for c in leaky_columns if c in out.columns]
    return out.drop(columns=existing_drops), dropped_erroneous, existing_drops


def clean_case_control_admission_features(
    patient_admissions: pd.DataFrame, control_admissions: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Preprocess case/control admission rows and return a compact summary."""
    patient_clean, p_dropped, p_cols = preprocess_admission_features(patient_admissions)
    control_clean, c_dropped, c_cols = preprocess_admission_features(control_admissions)
    summary = pd.DataFrame(
        [
            {"Cohort": "BE cases", "Invalid rows dropped": p_dropped, "Leaky columns removed": len(p_cols)},
            {"Cohort": "Controls", "Invalid rows dropped": c_dropped, "Leaky columns removed": len(c_cols)},
        ]
    )
    return patient_clean, control_clean, summary


def build_temporal_diagnosis_feature_matrix(
    patient_clean: pd.DataFrame,
    control_clean: pd.DataFrame,
    disease_subject_ids: Sequence[int],
    control_subject_ids: Sequence[int],
) -> Tuple[csr_matrix, np.ndarray, MultiLabelBinarizer, pd.Index, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Prefix diagnosis codes by admission recency and build the sparse patient-level matrix."""
    df = pd.concat([patient_clean, control_clean], axis=0)

    def prefix_codes(row, col_name):
        codes = row[col_name]
        if not isinstance(codes, list):
            return []
        return [f"T{int(row['event_index'])} {code}" for code in codes]

    df["diag_temporal"] = df.apply(lambda x: prefix_codes(x, "DIAGNOSIS (ICD_9)"), axis=1)
    df["proc_temporal"] = df.apply(lambda x: prefix_codes(x, "PROCEDURE TYPE"), axis=1)
    patient_df = (
        df.groupby("SUBJECT_ID")
        .agg({"diag_temporal": "sum", "proc_temporal": "sum", "LOS_DAYS": "max", "Age": "max"})
        .sort_index()
    )
    be_patients_only = patient_df.loc[patient_df.index.isin(disease_subject_ids)]
    controls_patients_only = patient_df.loc[patient_df.index.isin(control_subject_ids)]
    be_prevalence = pd.Series(Counter([c for sub in be_patients_only["diag_temporal"].apply(set) for c in sub])).sort_values(ascending=False)
    controls_prevalence = pd.Series(Counter([c for sub in controls_patients_only["diag_temporal"].apply(set) for c in sub])).sort_values(ascending=False)
    x_num = patient_df[["LOS_DAYS", "Age"]].fillna(0)
    diag_list = patient_df["diag_temporal"]
    patient_ids = patient_df.index
    mlb_diag = MultiLabelBinarizer(sparse_output=True).fit(diag_list)
    X_sparse = hstack([mlb_diag.transform(diag_list), csr_matrix(x_num.astype(float))]).tocsr()
    y = np.array([1 if pid in disease_subject_ids else 0 for pid in patient_ids])
    summary = pd.DataFrame(
        [
            {"Metric": "Patient-level matrix shape", "Value": str(X_sparse.shape)},
            {"Metric": "Target shape", "Value": str(y.shape)},
            {"Metric": "BE cases", "Value": int(y.sum())},
            {"Metric": "Controls", "Value": int(len(y) - y.sum())},
            {"Metric": "Procedure one-hots", "Value": "Excluded for eICU transportability"},
        ]
    )
    return X_sparse, y, mlb_diag, patient_ids, be_prevalence, controls_prevalence, patient_df, summary


def _norm(code):
    return str(code).replace(".", "").strip().upper()


def _matches(code, prefixes):
    c = _norm(code)
    return any(c.startswith(p) for p in prefixes)


def _per_patient(g):
    codes = g["ICD9_CODE"].map(_norm).unique()
    return pd.Series(
        {
            "cardiac_icd_mentions_1y": sum(_matches(c, CARDIAC_PREFIXES) for c in codes),
            "vascular_icd_mentions_1y": sum(_matches(c, VASCULAR_PREFIXES) for c in codes),
            "diabetes_comp_mentions_1y": sum(_matches(c, DIABETES_COMP_PREFIXES) for c in codes),
            "liver_icd_mentions_1y": sum(_matches(c, LIVER_PREFIXES) for c in codes),
            "valvular_icd_mentions_1y": sum(_matches(c, VALVULAR_PREFIXES) for c in codes),
            "congenital_icd_mentions_1y": sum(_matches(c, CONGENITAL_PREFIXES) for c in codes),
            "prior_total_icd_mentions_1y": len(codes),
        }
    )


def _ever_flags_dx(g):
    codes = g["ICD9_CODE"].map(_norm).unique()
    return pd.Series(
        {
            "renal_access_ever": int(any(_matches(c, RENAL_ACCESS_PREFIXES) for c in codes)),
            "cardiac_device_ever": int(any(_matches(c, DEVICE_PREFIXES) for c in codes)),
            "prosthetic_valve_dx_ever": int(any(_matches(c, PROSTHETIC_VALVE_DX_PREFIXES) for c in codes)),
            "ivdu_ever": int(any(_matches(c, IVDU_PREFIXES) for c in codes)),
        }
    )


def append_prior_history_clinical_features(
    X_sparse: csr_matrix,
    tables: Mapping[str, pd.DataFrame],
    diagnoses_icd_filtered: pd.DataFrame,
    cohort: Mapping[str, object],
    patient_ids: Sequence[int],
    patient_gender: pd.DataFrame,
    excluded_features: Optional[set] = None,
) -> Tuple[csr_matrix, pd.DataFrame, List[str], pd.DataFrame]:
    """Append prior-window clinical count/flag features while excluding generic utilization features."""
    excluded_features = excluded_features or {"prior_admit_count_1y", "prior_total_icd_mentions_1y"}
    admissions = tables["ADMISSIONS"].copy()
    admissions["ADMITTIME"] = pd.to_datetime(admissions["ADMITTIME"], errors="coerce")
    admissions["DISCHTIME"] = pd.to_datetime(admissions["DISCHTIME"], errors="coerce")

    disease_hadm_ids = cohort["disease_hadm_ids"]
    disease_subject_ids = cohort["disease_subject_ids"]
    control_subject_ids = cohort["control_subject_ids"]

    case_index = admissions[admissions["HADM_ID"].isin(disease_hadm_ids)].groupby("SUBJECT_ID")["ADMITTIME"].min()
    ctrl_index = admissions[admissions["SUBJECT_ID"].isin(control_subject_ids)].groupby("SUBJECT_ID")["DISCHTIME"].max()
    index_time = pd.concat([case_index, ctrl_index])
    index_time = index_time[~index_time.index.duplicated(keep="first")]

    dx = diagnoses_icd_filtered[["SUBJECT_ID", "HADM_ID", "ICD9_CODE"]].dropna(subset=["ICD9_CODE"]).copy()
    dx = dx.merge(admissions[["HADM_ID", "ADMITTIME"]], on="HADM_ID", how="left")
    dx["idx"] = dx["SUBJECT_ID"].map(index_time)
    dx["days_before"] = (dx["idx"] - dx["ADMITTIME"]).dt.days
    dx_win = dx[(dx["days_before"] >= MIN_DAYS_BEFORE) & (dx["days_before"] <= WINDOW_START_DAYS)].copy()
    dx_ever = dx[dx["days_before"] >= MIN_DAYS_BEFORE].copy()

    adm_idx = admissions[["SUBJECT_ID", "HADM_ID", "ADMITTIME"]].copy()
    adm_idx["idx"] = adm_idx["SUBJECT_ID"].map(index_time)
    adm_idx["days_before"] = (adm_idx["idx"] - adm_idx["ADMITTIME"]).dt.days
    adm_win = adm_idx[(adm_idx["days_before"] >= MIN_DAYS_BEFORE) & (adm_idx["days_before"] <= WINDOW_START_DAYS)]

    comorb = dx_win.groupby("SUBJECT_ID").apply(_per_patient)
    ever_dx = dx_ever.groupby("SUBJECT_ID").apply(_ever_flags_dx)
    admit_counts = adm_win.groupby("SUBJECT_ID")["HADM_ID"].nunique().rename("prior_admit_count_1y")

    gender_col = next((c for c in patient_gender.columns if str(c).upper() == "GENDER"), None)
    if gender_col is None:
        sex = pd.Series(0, index=pd.Index(patient_ids, name="SUBJECT_ID"), name="is_male")
    else:
        sex = (
            patient_gender[["SUBJECT_ID", gender_col]]
            .drop_duplicates("SUBJECT_ID")
            .set_index("SUBJECT_ID")[gender_col]
            .map(lambda g: 1 if str(g).upper().startswith("M") else 0)
            .rename("is_male")
        )

    new_feat = pd.DataFrame(index=pd.Index(patient_ids, name="SUBJECT_ID"))
    new_feat = new_feat.join(admit_counts).join(comorb).join(ever_dx).join(sex)
    all_feature_names = [
        "prior_admit_count_1y",
        "cardiac_icd_mentions_1y",
        "vascular_icd_mentions_1y",
        "diabetes_comp_mentions_1y",
        "liver_icd_mentions_1y",
        "valvular_icd_mentions_1y",
        "congenital_icd_mentions_1y",
        "prior_total_icd_mentions_1y",
        "renal_access_ever",
        "cardiac_device_ever",
        "prosthetic_valve_dx_ever",
        "is_male",
    ]
    for col in all_feature_names:
        if col not in new_feat.columns:
            new_feat[col] = 0
        new_feat[col] = new_feat[col].fillna(0)

    new_feature_names = [name for name in all_feature_names if name not in excluded_features]
    new_block = new_feat[new_feature_names].to_numpy(dtype=float)
    X_out = hstack([X_sparse, csr_matrix(new_block)]).tocsr()
    summary = pd.DataFrame(
        [
            {"Item": "Excluded generic utilization features", "Value": ", ".join(sorted(excluded_features))},
            {"Item": "Added engineered features", "Value": len(new_feature_names)},
            {"Item": "Feature window", "Value": f"index-{WINDOW_START_DAYS}d to index-{WINDOW_GAP_DAYS}d + ever-prior flags"},
            {"Item": "Final sparse matrix shape", "Value": str(X_out.shape)},
        ]
    )
    return X_out, new_feat, new_feature_names, summary


def build_cohort_descriptive_table(
    disease_subject_ids: Sequence[int],
    control_subject_ids: Sequence[int],
    patient_gender: pd.DataFrame,
    patient_clean: pd.DataFrame,
    control_clean: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the Table-1 style cohort summary and compact plots inputs."""
    def demo_frame(subject_ids, clean_df):
        df = pd.DataFrame({"SUBJECT_ID": subject_ids})
        df = df.merge(patient_gender, on="SUBJECT_ID", how="left")
        age = clean_df.groupby("SUBJECT_ID")["Age"].max()
        los = clean_df.groupby("SUBJECT_ID")["LOS_DAYS"].max()
        df = df.merge(age.reset_index(), on="SUBJECT_ID", how="left")
        df = df.merge(los.reset_index(), on="SUBJECT_ID", how="left")
        return df

    def sig_label(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "ns"

    def fmt_median_iqr(s):
        s = s.dropna()
        if s.empty:
            return "NA"
        return f"{s.median():.1f} [{s.quantile(0.25):.1f}-{s.quantile(0.75):.1f}]"

    cases_demo = demo_frame(disease_subject_ids, patient_clean)
    controls_demo = demo_frame(control_subject_ids, control_clean)
    n_cases, n_controls = len(cases_demo), len(controls_demo)
    n_cases_f = (cases_demo["GENDER"] == "F").sum()
    n_cases_m = (cases_demo["GENDER"] == "M").sum()
    n_controls_f = (controls_demo["GENDER"] == "F").sum()
    n_controls_m = (controls_demo["GENDER"] == "M").sum()
    pct_cases_f = n_cases_f / n_cases if n_cases else np.nan
    pct_cases_m = n_cases_m / n_cases if n_cases else np.nan
    pct_controls_f = n_controls_f / n_controls if n_controls else np.nan
    pct_controls_m = n_controls_m / n_controls if n_controls else np.nan
    contingency_gender = np.array([[n_cases_f, n_cases - n_cases_f], [n_controls_f, n_controls - n_controls_f]])
    _, p_gender, _, _ = chi2_contingency(contingency_gender)
    _, p_age = mannwhitneyu(cases_demo["Age"].dropna(), controls_demo["Age"].dropna(), alternative="two-sided")
    _, p_los = mannwhitneyu(cases_demo["LOS_DAYS"].dropna(), controls_demo["LOS_DAYS"].dropna(), alternative="two-sided")

    table = pd.DataFrame(
        [
            {"Characteristic": "Female, n (%)", "Positive class / BE cases": f"{n_cases_f} ({pct_cases_f*100:.1f}%)", "Negative class / Controls": f"{n_controls_f} ({pct_controls_f*100:.1f}%)", "p_value": p_gender, "Significance": sig_label(p_gender)},
            {"Characteristic": "Male, n (%)", "Positive class / BE cases": f"{n_cases_m} ({pct_cases_m*100:.1f}%)", "Negative class / Controls": f"{n_controls_m} ({pct_controls_m*100:.1f}%)", "p_value": p_gender, "Significance": sig_label(p_gender)},
            {"Characteristic": "Age, median [IQR]", "Positive class / BE cases": fmt_median_iqr(cases_demo["Age"]), "Negative class / Controls": fmt_median_iqr(controls_demo["Age"]), "p_value": p_age, "Significance": sig_label(p_age)},
            {"Characteristic": "LOS (days), median [IQR]", "Positive class / BE cases": fmt_median_iqr(cases_demo["LOS_DAYS"]), "Negative class / Controls": fmt_median_iqr(controls_demo["LOS_DAYS"]), "p_value": p_los, "Significance": sig_label(p_los)},
        ]
    )
    return table, cases_demo, controls_demo


def summarize_sample_size(X_sparse: csr_matrix, y: np.ndarray) -> pd.DataFrame:
    """Return sample size, prevalence, feature count, and events-per-feature."""
    n_cases = int(np.sum(y == 1))
    n_controls = int(np.sum(y == 0))
    n_features = X_sparse.shape[1]
    prevalence = n_cases / (n_cases + n_controls)
    events_per_feature = n_cases / n_features
    return pd.DataFrame(
        [
            {"Metric": "Cases (BE)", "Value": f"{n_cases:,}"},
            {"Metric": "Controls", "Value": f"{n_controls:,}"},
            {"Metric": "Prevalence", "Value": f"{prevalence:.3%}"},
            {"Metric": "Features", "Value": f"{n_features:,}"},
            {"Metric": "Events per feature", "Value": f"{events_per_feature:.3f}"},
        ]
    )


def ensure_2d_shap_array(shap_output):
    """Normalize SHAP outputs (Explanation, ndarray, 3D) to a 2D ndarray."""
    if hasattr(shap_output, "values"):
        shap_output = shap_output.values
    arr = np.asarray(shap_output)
    if arr.ndim == 3:
        arr = arr[:, :, -1]
    elif arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def small_dense_rows(X, row_idx):
    """Materialize a small subset of rows from a sparse matrix as a dense ndarray."""
    X_sub = X[row_idx]
    return X_sub.toarray() if hasattr(X_sub, "toarray") else np.asarray(X_sub)


def compute_fold_metrics(y_true, probas):
    """AUC, sensitivity, and specificity at Youden-optimal threshold."""
    probas = np.asarray(probas)
    auc_score = roc_auc_score(y_true, probas)
    fpr, tpr, thresholds = roc_curve(y_true, probas)
    j_scores = tpr + (1 - fpr) - 1
    best_idx = np.argmax(j_scores)
    best_threshold = float(np.clip(thresholds[best_idx], 0, 1))
    preds = (probas >= best_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return auc_score, sens, spec, best_threshold


def normalize_importance(arr):
    """Normalize an importance vector so it sums to 1."""
    s = arr.sum()
    return arr / s if s > 0 else arr


def partition_controls(control_subject_ids, control_sample_size, seed=42):
    """Shuffle controls and split into non-overlapping chunks."""
    rng_global = np.random.default_rng(seed)
    shuffled_controls = rng_global.permutation(control_subject_ids)
    parts = [
        shuffled_controls[i : i + control_sample_size]
        for i in range(0, len(shuffled_controls), control_sample_size)
    ]
    if parts and len(parts[-1]) < control_sample_size:
        parts = parts[:-1]
    return parts


def build_fold_models(X_tr, y_tr, seed):
    """Fit XGBoost, Random Forest, and L1 logistic regression using fold-only training data."""
    cases_orig = int(np.sum(y_tr))
    controls_orig = int(len(y_tr) - cases_orig)
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        scale_pos_weight=controls_orig / cases_orig,
        max_depth=3,
        random_state=seed,
        eval_metric="logloss",
    )
    xgb.fit(X_tr, y_tr)

    ros = RandomOverSampler(random_state=seed, sampling_strategy=0.1)
    X_tr_res, y_tr_res = ros.fit_resample(X_tr, y_tr)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_tr_res, y_tr_res)

    lasso = LogisticRegression(
        solver="saga", l1_ratio=1.0, C=1.0, random_state=seed, max_iter=2000
    )
    lasso.fit(X_tr_res, y_tr_res)

    models = {"XGBoost": xgb, "RandomForest": rf, "Lasso": lasso}
    counts = {"cases_orig": cases_orig, "controls_orig": controls_orig, "X_tr_res": X_tr_res}
    return models, counts


def compute_fold_shap(models, X_tr, X_tr_res, X_te, rng, shap_bg_size, shap_test_size):
    """Return sampled SHAP values for the headline models.

    RF and Lasso are trained on the oversampled fold, so their SHAP background is
    sampled from the oversampled training matrix. XGBoost is trained on the
    original imbalanced fold, so it keeps an original-fold background. If XGBoost
    SHAP fails in a local environment, RF/Lasso SHAP still completes.
    """
    bg_n_xgb = min(shap_bg_size, X_tr.shape[0])
    bg_n_resampled = min(shap_bg_size, X_tr_res.shape[0])
    te_n = min(shap_test_size, X_te.shape[0])

    xgb_bg_idx = rng.choice(X_tr.shape[0], size=bg_n_xgb, replace=False)
    resampled_bg_idx = rng.choice(X_tr_res.shape[0], size=bg_n_resampled, replace=False)
    test_idx = rng.choice(X_te.shape[0], size=te_n, replace=False)

    X_bg_xgb_dense = small_dense_rows(X_tr, xgb_bg_idx)
    X_bg_resampled_dense = small_dense_rows(X_tr_res, resampled_bg_idx)
    X_te_shap_dense = small_dense_rows(X_te, test_idx)

    shap_arrays = {}

    # XGBoost can be brittle across SHAP/XGBoost versions. Keep it optional so the
    # portfolio notebook still produces the RF/Lasso interpretability views.
    try:
        xgb_explainer = shap.TreeExplainer(
            models["XGBoost"],
            data=X_bg_xgb_dense,
            feature_perturbation="interventional",
        )
        shap_arrays["XGBoost"] = ensure_2d_shap_array(
            xgb_explainer(X_te_shap_dense, check_additivity=False)
        )
    except Exception:
        shap_arrays["XGBoost"] = np.zeros((X_te_shap_dense.shape[0], X_te_shap_dense.shape[1]))

    rf_explainer = shap.TreeExplainer(
        models["RandomForest"],
        data=X_bg_resampled_dense,
        feature_perturbation="interventional",
    )
    shap_arrays["RandomForest"] = ensure_2d_shap_array(
        rf_explainer(X_te_shap_dense, check_additivity=False)
    )

    lasso_explainer = shap.LinearExplainer(models["Lasso"], X_bg_resampled_dense)
    shap_arrays["Lasso"] = ensure_2d_shap_array(lasso_explainer(X_te_shap_dense))

    return shap_arrays, X_te_shap_dense, te_n


def train_models_with_repeated_cv(
    X_sparse: csr_matrix,
    y: np.ndarray,
    patient_ids: Sequence[int],
    disease_subject_ids: Sequence[int],
    control_subject_ids: Sequence[int],
    feature_names: List[str],
    control_sample_size: Optional[int] = None,
    n_splits: int = 5,
    shap_bg_size: int = 200,
    shap_test_size: int = 300,
    random_seed: int = 42,
    compute_shap: bool = True,
) -> Dict[str, object]:
    """Run repeated control-partition CV, collect predictions, importances, and SHAP values.

    Set ``compute_shap=False`` to skip the per-fold SHAP computation. Predictions,
    fold metrics, and feature-importance aggregates are unaffected, so this is the
    cheap path for secondary runs (e.g. a feature-set sensitivity comparison) where
    only discrimination/precision metrics are needed. When skipped, ``final_X_te_df``
    and ``final_shap_dfs`` come back empty.
    """
    fold_metrics, all_X_te_dfs = [], []
    control_sample_size = control_sample_size or min(2800, len(control_subject_ids))
    partitions = partition_controls(control_subject_ids, control_sample_size, seed=random_seed)
    feature_importance_store = {"XGBoost": [], "RandomForest": [], "Lasso": [], "Chi2": []}
    model_predictions = {m: {"y": [], "probs": []} for m in ["XGBoost", "RandomForest", "Lasso"]}
    all_shap_dfs = {"XGBoost": [], "RandomForest": [], "Lasso": []}
    rf_fold_roc_data = []

    for seed, sampled_control_ids in enumerate(partitions):
        rng = np.random.default_rng(seed)
        keep_ids = np.concatenate([disease_subject_ids, sampled_control_ids])
        mask = np.isin(patient_ids, keep_ids)
        X_seed = X_sparse[mask]
        y_seed = y[mask]
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_seed, y_seed), start=1):
            scaler = MaxAbsScaler()
            X_tr, X_te = X_seed[train_idx], X_seed[test_idx]
            y_tr, y_te = y_seed[train_idx], y_seed[test_idx]
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            chi_vals, _ = chi2(X_tr.maximum(0), y_tr)
            models, counts = build_fold_models(X_tr, y_tr, seed)

            for model_name, model in models.items():
                probs = model.predict_proba(X_te)[:, 1]
                auc_score, sens, spec, threshold = compute_fold_metrics(y_te, probs)
                if model_name == "RandomForest":
                    rf_fold_roc_data.append((y_te.copy(), probs.copy()))
                model_predictions[model_name]["y"].extend(y_te.tolist())
                model_predictions[model_name]["probs"].extend(probs.tolist())
                fold_metrics.append({"Seed": seed, "Fold": fold, "Model": model_name, "AUC": auc_score, "Sens": sens, "Spec": spec, "Threshold": threshold})

            feature_importance_store["Chi2"].append(pd.DataFrame([normalize_importance(chi_vals)], columns=feature_names))
            feature_importance_store["XGBoost"].append(pd.DataFrame([normalize_importance(models["XGBoost"].feature_importances_)], columns=feature_names))
            feature_importance_store["RandomForest"].append(pd.DataFrame([normalize_importance(models["RandomForest"].feature_importances_)], columns=feature_names))
            feature_importance_store["Lasso"].append(pd.DataFrame([normalize_importance(np.abs(models["Lasso"].coef_[0]))], columns=feature_names))

            if compute_shap:
                shap_arrays, X_te_shap_dense, _ = compute_fold_shap(
                    models, X_tr, counts["X_tr_res"], X_te, rng, shap_bg_size, shap_test_size
                )
                for model_name in all_shap_dfs:
                    all_shap_dfs[model_name].append(pd.DataFrame(shap_arrays[model_name], columns=feature_names))
                all_X_te_dfs.append(pd.DataFrame(X_te_shap_dense, columns=feature_names))

    master = {k: pd.concat(v).mean().fillna(0) for k, v in feature_importance_store.items()}
    if compute_shap:
        final_X_te_df = pd.concat(all_X_te_dfs, axis=0).fillna(0).reset_index(drop=True)
        final_shap = {
            k: pd.concat(v, axis=0).reindex(columns=final_X_te_df.columns).fillna(0).reset_index(drop=True)
            for k, v in all_shap_dfs.items()
        }
    else:
        final_X_te_df = pd.DataFrame()
        final_shap = {k: pd.DataFrame() for k in all_shap_dfs}
    fold_metrics_df = pd.DataFrame(fold_metrics)
    summary_metrics_df = (
        fold_metrics_df.groupby("Model")[["AUC", "Sens", "Spec", "Threshold"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary_metrics_df.columns = [f"{metric}_{stat}" for metric, stat in summary_metrics_df.columns]
    summary_metrics_df = summary_metrics_df.reset_index()
    master_feat_df = pd.DataFrame(
        {"XGBoost": master["XGBoost"], "RandomForest": master["RandomForest"], "Lasso": master["Lasso"], "Chi2": master["Chi2"]}
    ).reset_index().rename(columns={"index": "Feature"})

    return {
        "model_predictions": model_predictions,
        "fold_metrics_df": fold_metrics_df,
        "summary_metrics_df": summary_metrics_df,
        "master_feat_df": master_feat_df,
        "final_X_te_df": final_X_te_df,
        "final_shap_dfs": final_shap,
        "rf_fold_roc_data": rf_fold_roc_data,
    }


def add_rf_lasso_ensembles(model_predictions: Dict[str, Dict[str, list]]) -> Tuple[Dict[str, Dict[str, list]], pd.DataFrame]:
    """Add soft-vote and stacked RF+Lasso predictions to the pooled prediction dictionary."""
    members = ["RandomForest", "Lasso"]
    y_ens = np.asarray(model_predictions[members[0]]["y"])
    for m in members[1:]:
        assert np.array_equal(np.asarray(model_predictions[m]["y"]), y_ens), "pooled label order differs across models"
    P = np.column_stack([np.asarray(model_predictions[m]["probs"]) for m in members])
    soft_vote_probs = P.mean(axis=1)
    stack_probs = cross_val_predict(LogisticRegression(max_iter=1000), P, y_ens, cv=5, method="predict_proba")[:, 1]
    model_predictions["SoftVote(RF+Lasso)"] = {"y": y_ens.tolist(), "probs": soft_vote_probs.tolist()}
    model_predictions["Stack(RF+Lasso)"] = {"y": y_ens.tolist(), "probs": stack_probs.tolist()}

    rows = []
    for model_name in ["RandomForest", "XGBoost", "Lasso", "SoftVote(RF+Lasso)", "Stack(RF+Lasso)"]:
        yt = np.asarray(model_predictions[model_name]["y"])
        pp = np.asarray(model_predictions[model_name]["probs"])
        fpr, tpr, _ = roc_curve(yt, pp)
        rows.append({"Model": model_name, "ROC_AUC": round(auc(fpr, tpr), 4), "Average_Precision": round(average_precision_score(yt, pp), 4)})
    return model_predictions, pd.DataFrame(rows).sort_values("Average_Precision", ascending=False)


def plot_pooled_roc_all_models(model_predictions, filename="final_roc_curve_all_models.png"):
    """Plot pooled ROC curves and return ranked AUC table."""
    fig, ax = plt.subplots(figsize=(8, 6))
    rows = []
    colors = {"XGBoost": "darkorange", "RandomForest": "mediumpurple", "Lasso": "seagreen", "SoftVote(RF+Lasso)": "crimson", "Stack(RF+Lasso)": "black"}
    for model_name, pred in model_predictions.items():
        y_true = np.asarray(pred["y"])
        y_prob = np.asarray(pred["probs"])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = auc(fpr, tpr)
        rows.append({"Model": model_name, "Pooled_ROC_AUC": auc_score})
        ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC={auc_score:.3f})", color=colors.get(model_name))
    ax.plot([0, 1], [0, 1], "navy", lw=1.5, linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Pooled ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(filename, dpi=300)
    plt.show()
    return pd.DataFrame(rows).sort_values("Pooled_ROC_AUC", ascending=False)


def plot_pr_curves_all_models(model_predictions, filename="pr_curves_all_models.png"):
    """Plot pooled precision-recall curves and return AP ranking."""
    fig, ax = plt.subplots(figsize=(8, 6))
    rows, baselines = [], []
    colors = {"XGBoost": "darkorange", "RandomForest": "mediumpurple", "Lasso": "seagreen", "SoftVote(RF+Lasso)": "crimson", "Stack(RF+Lasso)": "black"}
    for model_name, pred in model_predictions.items():
        y_true = np.asarray(pred["y"])
        y_prob = np.asarray(pred["probs"])
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        prevalence = y_true.mean()
        baselines.append(prevalence)
        rows.append({"Model": model_name, "Average_Precision": ap, "Baseline_prevalence": prevalence})
        ax.plot(recall, precision, lw=2, label=f"{model_name} (AP={ap:.3f})", color=colors.get(model_name))
    ax.axhline(np.mean(baselines), color="gray", linestyle="--", lw=1.5, label=f"Baseline={np.mean(baselines):.3f}")
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall Curves (Pooled Predictions)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(filename, dpi=300)
    plt.show()
    return pd.DataFrame(rows).sort_values("Average_Precision", ascending=False)


def shap_top_features_table(shap_df, model_name, max_display=15, min_mean_abs=1e-8):
    """Return a labeled table of the top features by mean absolute SHAP value."""
    if shap_df is None or shap_df.empty:
        return pd.DataFrame(columns=["Rank", "Feature", "Mean_abs_SHAP", "Model"])
    mean_abs = shap_df.abs().mean(axis=0).sort_values(ascending=False)
    mean_abs = mean_abs[mean_abs > min_mean_abs].head(max_display)
    return pd.DataFrame({
        "Rank": range(1, len(mean_abs) + 1),
        "Feature": mean_abs.index,
        "Mean_abs_SHAP": mean_abs.values.round(5),
        "Model": model_name,
    })


def shap_beeswarm(shap_df, X_df, model_name, output_path, max_display=30, min_mean_abs=1e-8):
    """Draw a SHAP beeswarm for the top non-zero features."""
    if shap_df is None or X_df is None or shap_df.empty or X_df.empty:
        return []

    aligned_columns = [c for c in shap_df.columns if c in X_df.columns]
    if not aligned_columns:
        return []

    shap_df = shap_df[aligned_columns]
    X_df = X_df[aligned_columns]

    mean_abs = shap_df.abs().mean(axis=0).sort_values(ascending=False)
    keep = mean_abs[mean_abs > min_mean_abs].head(max_display).index.tolist()
    if not keep:
        return []

    shap.summary_plot(
        shap_df[keep].values,
        X_df[keep].values,
        feature_names=keep,
        max_display=len(keep),
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.title(f"SHAP beeswarm: {model_name}")
    plt.tight_layout()
    save_fig(output_path, dpi=300, bbox_inches="tight")
    plt.show()
    return keep


def plot_calibration_all_models(model_predictions, filename="calibration_all_models.png", n_bins=10):
    """Plot calibration curves and return Brier scores."""
    fig, ax = plt.subplots(figsize=(8, 6))
    rows = []
    for model_name, pred in model_predictions.items():
        y_true = np.asarray(pred["y"])
        y_prob = np.asarray(pred["probs"])
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
        brier = brier_score_loss(y_true, y_prob)
        rows.append({"Model": model_name, "Brier": brier})
        ax.plot(mean_pred, frac_pos, marker="o", lw=2, label=f"{model_name} (Brier={brier:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration Curves")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_fig(filename, dpi=300)
    plt.show()
    return pd.DataFrame(rows).sort_values("Brier")


def bootstrap_metric_ci(y_true, y_score, metric_fn, n_boot=1000, seed=42):
    """Percentile 95% CI for a ranking metric, resampling scored rows."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    vals = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric_fn(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return lo, hi


def build_metric_ci_table(model_predictions, n_boot=1000, seed=42):
    """Bootstrap 95% CIs for pooled ROC-AUC and AP."""
    rows = []
    for model_name, pred in model_predictions.items():
        y_true = np.asarray(pred["y"])
        y_prob = np.asarray(pred["probs"])
        auc_lo, auc_hi = bootstrap_metric_ci(y_true, y_prob, roc_auc_score, n_boot=n_boot, seed=seed)
        ap_lo, ap_hi = bootstrap_metric_ci(y_true, y_prob, average_precision_score, n_boot=n_boot, seed=seed + 1)
        rows.append({
            "Model": model_name,
            "ROC_AUC": roc_auc_score(y_true, y_prob),
            "ROC_AUC_95CI": f"{auc_lo:.3f}-{auc_hi:.3f}",
            "Average_Precision": average_precision_score(y_true, y_prob),
            "AP_95CI": f"{ap_lo:.3f}-{ap_hi:.3f}",
        })
    return pd.DataFrame(rows).sort_values("Average_Precision", ascending=False)


def train_final_models_for_export(X_sparse, y, feature_names, export_dir="model_outputs/export", random_seed=42):
    """Train final RF/XGB/Lasso models on the full matrix and save scoring artifacts."""
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X_sparse)
    models, _ = build_fold_models(X_scaled, y, random_seed)
    for name, model in models.items():
        joblib.dump(model, export_dir / f"{name}_final.joblib")
    joblib.dump(scaler, export_dir / "scaler_maxabs.joblib")
    with open(export_dir / "feature_names.json", "w") as f:
        json.dump(list(feature_names), f, indent=2)
    return pd.DataFrame([
        {"Artifact": "Final models", "Value": ", ".join(models.keys())},
        {"Artifact": "Scaler", "Value": "scaler_maxabs.joblib"},
        {"Artifact": "Feature names", "Value": "feature_names.json"},
        {"Artifact": "Feature count", "Value": int(X_sparse.shape[1])},
    ])


def summarize_shap_by_feature_family(shap_df: pd.DataFrame, X_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize why dense numerical features can dominate mean |SHAP| rankings.

    ICD one-hot features are sparse, so their average contribution across all rows
    is often diluted by many zero-valued patients. Dense numeric/count features
    can contribute for nearly every patient.
    """
    if shap_df is None or X_df is None or shap_df.empty or X_df.empty:
        return pd.DataFrame()

    aligned_columns = [c for c in shap_df.columns if c in X_df.columns]
    rows = []
    for feature_name in aligned_columns:
        values = np.asarray(X_df[feature_name])
        nonzero_rate = float(np.mean(values != 0))
        if feature_name.startswith("DX_") or feature_name.startswith("T-"):
            feature_family = "Sparse ICD one-hot"
        elif feature_name in {"Age", "LOS_DAYS", "is_male"} or feature_name.endswith("_1y") or feature_name.endswith("_ever"):
            feature_family = "Dense/demographic/engineered"
        else:
            feature_family = "Other"

        rows.append({
            "Feature": feature_name,
            "Feature family": feature_family,
            "Mean_abs_SHAP_all_rows": float(np.mean(np.abs(shap_df[feature_name]))),
            "Nonzero_rate": nonzero_rate,
            "Mean_abs_SHAP_when_present": (
                float(np.mean(np.abs(shap_df.loc[np.asarray(values) != 0, feature_name])))
                if np.any(values != 0)
                else 0.0
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("Mean_abs_SHAP_all_rows", ascending=False)
        .reset_index(drop=True)
    )


def select_model_feature_space(
    candidate_feature_matrix: csr_matrix,
    candidate_feature_names: Sequence[str],
    reduced_features: pd.DataFrame,
    use_reduced_feature_set: bool = True,
    feature_column: str = "Feature",
) -> Tuple[csr_matrix, List[str], pd.DataFrame]:
    """Choose reduced CSV features or the full candidate matrix."""
    candidate_feature_names = list(candidate_feature_names)

    if not use_reduced_feature_set:
        summary = pd.DataFrame([
            {"Item": "Feature set", "Value": "Full candidate matrix"},
            {"Item": "Features used", "Value": len(candidate_feature_names)},
            {"Item": "Final matrix shape", "Value": str(candidate_feature_matrix.shape)},
        ])
        return candidate_feature_matrix, candidate_feature_names, summary

    if feature_column not in reduced_features.columns:
        feature_column = reduced_features.columns[0]

    requested_features = (
        reduced_features[feature_column]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    feature_name_to_column = {
        feature_name: column_index
        for column_index, feature_name in enumerate(candidate_feature_names)
    }

    selected_feature_names = [
        feature_name
        for feature_name in requested_features
        if feature_name in feature_name_to_column
    ]

    missing_features = [
        feature_name
        for feature_name in requested_features
        if feature_name not in feature_name_to_column
    ]

    if not selected_feature_names:
        raise ValueError(
            "No features from top_reduced_BASE_features.csv matched the constructed feature matrix."
        )

    selected_columns = [
        feature_name_to_column[feature_name]
        for feature_name in selected_feature_names
    ]

    selected_feature_matrix = candidate_feature_matrix[:, selected_columns].tocsr()

    summary = pd.DataFrame([
        {"Item": "Feature set", "Value": "Reduced CSV feature list"},
        {"Item": "Features requested by CSV", "Value": len(requested_features)},
        {"Item": "Features matched", "Value": len(selected_feature_names)},
        {"Item": "Features missing", "Value": len(missing_features)},
        {"Item": "Final matrix shape", "Value": str(selected_feature_matrix.shape)},
    ])

    return selected_feature_matrix, selected_feature_names, summary

