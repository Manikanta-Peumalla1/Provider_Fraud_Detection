# Import required libraries for data manipulation and preprocessing
import pandas as pd  
import numpy as np 

import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _resolve_input_path(path: str) -> Path:
    """
    Resolve input file path to an absolute Path, checking multiple locations.

    Tries the path as-is, then relative to workspace root, then normalized relative,
    then relative to current working directory.

    Parameters
    ----------
    path : str
        The input path string.

    Returns
    -------
    Path
        The resolved absolute path to the file.

    Raises
    ------
    FileNotFoundError
        If the file cannot be found in any of the attempted locations.
    """
    candidate = Path(path)
    if candidate.exists():
        return candidate

    workspace_root = Path(__file__).resolve().parents[1]
    candidate = (workspace_root / path).resolve()
    if candidate.exists():
        return candidate

    # If user passes '../Train_Test_Data/...', normalize to workspace root as intended
    p = Path(path)
    relative_parts = [part for part in p.parts if part not in ('.', '..')]
    if relative_parts:
        candidate = workspace_root.joinpath(*relative_parts)
        if candidate.exists():
            return candidate

    candidate = Path.cwd() / path
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Data file not found: '{path}'")


def read_provider_data(path:str,nrows:Optional[int]=None)->pd.DataFrame:
    """
    Read and validate provider data from a CSV file.

    Parameters
    ----------
    path : str
        Path to the provider CSV file.
    nrows : int, optional
        Number of rows to read. If None, reads all.

    Returns
    -------
    pd.DataFrame
        DataFrame with provider data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty.
    """
    # Read the Master Providers CSV file into a dataframe
    #df_provider = pd.read_csv(path)
    #return df_provider
    file_path = _resolve_input_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Provider data file not found: '{path}'")

    logger.info(f"Reading provider data from: {file_path}")
    df = pd.read_csv(file_path, nrows=nrows)

    if df.empty:
        raise ValueError(f"Provider data file is empty: '{path}'")

    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns.")
    return df





# Constants
DATE_COLUMNS = {
    'admission': ('AdmissionDt', 'DischargeDt', 'IP_Number_of_Days_in_Hospital'),
    'claim':     ('ClaimStartDt', 'ClaimEndDt', 'IP_Claim_Days'),
}

PHYSICIAN_COLS = ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']

DIAGNOSIS_COLS = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]
PROCEDURE_COLS = [f'ClmProcedureCode_{i}' for i in range(1, 7)]

DROP_AFTER_FEATURE_COLS = (
    ['ClmAdmitDiagnosisCode', 'DiagnosisGroupCode']
    + DIAGNOSIS_COLS
    + PROCEDURE_COLS
)

RENAME_MAP = {
    'ClaimID':               'IP_ClaimID',
    'InscClaimAmtReimbursed':'IP_InscClaimAmtReimbursed',
    'DeductibleAmtPaid':     'IP_DeductibleAmtPaid',
}

EXPECTED_COLUMNS = (
    ['ClaimID', 'AdmissionDt', 'DischargeDt', 'ClaimStartDt', 'ClaimEndDt']
    + PHYSICIAN_COLS
    + DIAGNOSIS_COLS
    + PROCEDURE_COLS
    + ['ClmAdmitDiagnosisCode', 'DiagnosisGroupCode',
       'InscClaimAmtReimbursed', 'DeductibleAmtPaid']
)


def _parse_dates(df: pd.DataFrame, col: str) -> pd.Series:
    """
    Parse a column in the DataFrame to datetime, coercing errors to NaT.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the column.
    col : str
        The column name to parse.

    Returns
    -------
    pd.Series
        The parsed datetime series.
    """
    parsed = pd.to_datetime(df[col], errors='coerce')
    n_failed = parsed.isna().sum() - df[col].isna().sum()
    if n_failed > 0:
        logger.warning(f"Column '{col}': {n_failed} value(s) could not be parsed as dates → set to NaT.")
    return parsed


def _validate_columns(df: pd.DataFrame, path: str) -> None:
    """
    Validate that expected columns are present in the DataFrame, logging warnings for missing ones.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    path : str
        The file path for logging purposes.
    """
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(f"File '{path}' is missing expected column(s): {missing}")


def _safe_drop(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Drop columns from DataFrame if they exist, logging warnings for absent ones.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to modify.
    columns : list[str]
        List of column names to drop.

    Returns
    -------
    pd.DataFrame
        The DataFrame with columns dropped.
    """
    to_drop = [c for c in columns if c in df.columns]
    skipped  = [c for c in columns if c not in df.columns]
    if skipped:
        logger.warning(f"Columns not found for drop (already absent?): {skipped}")
    return df.drop(columns=to_drop)


def read_ip_data(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Read and preprocess inpatient (IP) claims data from a CSV file.

    Steps
    -----
    1. Validate file existence and read CSV.
    2. Validate expected schema.
    3. Parse date columns and derive duration features.
    4. Count diagnosis and procedure codes per claim.
    5. Drop raw/intermediate columns and rename for clarity.

    Parameters
    ----------
    path  : str | Path — Location of the inpatient CSV file.
    nrows : int, optional — Read only the first N rows (useful for testing).

    Returns
    -------
    pd.DataFrame with engineered IP features, ready for downstream modelling.

    Raises
    ------
    FileNotFoundError : if the path does not exist.
    ValueError        : if the file is empty.
    """
    # ── 1. File validation & load ──────────────────────────────────────────
    file_path = _resolve_input_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Inpatient data file not found: '{path}'")

    logger.info(f"Reading inpatient data from: {file_path}")
    df = pd.read_csv(file_path, nrows=nrows)

    if df.empty:
        raise ValueError(f"Inpatient data file is empty: '{path}'")

    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns.")

    # ── 2. Schema validation ───────────────────────────────────────────────
    _validate_columns(df, path)

    # ── 3. Date parsing & duration features ───────────────────────────────
    for start_col, end_col, new_col in DATE_COLUMNS.values():
        if start_col in df.columns and end_col in df.columns:
            df[start_col] = _parse_dates(df, start_col)
            df[end_col]   = _parse_dates(df, end_col)
            df[new_col]   = (df[end_col] - df[start_col]).dt.days
            logger.info(f"Derived '{new_col}' from '{start_col}' → '{end_col}'.")

    date_cols_to_drop = [c for pair in DATE_COLUMNS.values() for c in pair[:2]]
    df = _safe_drop(df, date_cols_to_drop + PHYSICIAN_COLS)

    # ── 4. Diagnosis & procedure feature engineering ───────────────────────
    available_diag = [c for c in DIAGNOSIS_COLS if c in df.columns]
    available_proc = [c for c in PROCEDURE_COLS if c in df.columns]

    df['IP_Unique_Disease_Count']   = df[available_diag].count(axis=1)
    df['IP_Unique_Treatment_Count'] = df[available_proc].count(axis=1)
    logger.info("Computed IP_Unique_Disease_Count and IP_Unique_Treatment_Count.")

    # ── 5. Drop raw code columns & rename ─────────────────────────────────
    df = _safe_drop(df, DROP_AFTER_FEATURE_COLS)

    existing_rename = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    df.rename(columns=existing_rename, inplace=True)
    logger.info(f"Renamed columns: {existing_rename}")

    logger.info(f"Preprocessing complete. Output shape: {df.shape}")
    return df


DATE_COLUMNS_OP={'CLAIM':('ClaimStartDt','ClaimEndDt','OP_Claim_Days')}
PHYSICIAN_COLS = ['AttendingPhysician', 'OperatingPhysician', 'OtherPhysician']
DIAGNOSIS_COLS = [f'ClmDiagnosisCode_{i}' for i in range(1, 11)]
PROCEDURE_COLS = [f'ClmProcedureCode_{i}' for i in range(1, 7)]
OP_DROP_AFTER_FEATURE_COLS = (
    ['ClmAdmitDiagnosisCode']+
    DIAGNOSIS_COLS
   + PROCEDURE_COLS
)

EXPECTED_COLUMNS = (
    ['ClaimID','BeneID','Provider', 'ClaimStartDt', 'ClaimEndDt']
    + PHYSICIAN_COLS
    + DIAGNOSIS_COLS
    + PROCEDURE_COLS
    + ['OP_InscClaimAmtReimbursed', 'OP_DeductibleAmtPaid']
)

RENAME_MAP_OP = {
    'ClaimID':               'OP_ClaimID',
    'InscClaimAmtReimbursed':'OP_InscClaimAmtReimbursed',
    'DeductibleAmtPaid':     'OP_DeductibleAmtPaid',
}
def read_op_data(path:str,nrows:Optional[int]=None)->pd.DataFrame:
    """
    Read and preprocess outpatient (OP) claims data from a CSV file.

    Steps
    -----
    1. Validate file existence and read CSV.
    2. Validate expected schema.
    3. Parse date columns and derive duration features.
    4. Count diagnosis and procedure codes per claim.
    5. Drop raw/intermediate columns and rename for clarity.

    Parameters
    ----------
    path : str | Path
        Location of the outpatient CSV file.
    nrows : int, optional
        Read only the first N rows (useful for testing).

    Returns
    -------
    pd.DataFrame
        DataFrame with engineered OP features, ready for downstream modelling.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    ValueError
        If the file is empty.
    """
    file_path = _resolve_input_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Outpatient data file not found: '{path}'")
    logger.info(f"reading Outpatient data from {file_path}")
    df=pd.read_csv(file_path,nrows=nrows)
    if df.empty:
        raise ValueError("Loaded Data frame was empty")
    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns.")
        # ── 2. Schema validation ──────────────────────────────────────────────
    _validate_columns(df, path)
    
    # -------3. Parse date columns -----------
    for start,end,new in DATE_COLUMNS_OP.values():
        if start in df.columns and end in df.columns:
            df[start]=_parse_dates(df=df,col=start)
            df[end]=_parse_dates(df=df,col=end)
            df[new]=(df[end]-df[start]).dt.days
            logger.info("Date columns parsed successfully and derrived new col")
    date_cols_to_drop = [c for pair in DATE_COLUMNS_OP.values() for c in pair[:2]]
    df=_safe_drop(df=df,columns=date_cols_to_drop+PHYSICIAN_COLS)
    #---- 4. Count diagnosis and procedure codes per claim.
    available_daig_cols=[c for c in DIAGNOSIS_COLS if c in df.columns]
    available_proc_cols=[c for c in PROCEDURE_COLS if c in df.columns]
    df["OP_Unique_Disease_Count"]=df[available_daig_cols].count(axis=1)
    df["OP_Unique_Treatment_Count"]=df[available_proc_cols].count(axis=1)
    logger.info("Computed OP_Unique_Disease_Count and OP_Unique_Treatment_Count.")
    
    # ----5 Drop raw/intermediate columns and rename for clarity.
    df=_safe_drop(df=df,columns=OP_DROP_AFTER_FEATURE_COLS)
    existing_col_rename={k:v for k, v in RENAME_MAP_OP.items() if k in df.columns}
    df=df.rename(columns=existing_col_rename)
    logger.info(f"Renamed columns: {existing_col_rename}")

    logger.info(f"Preprocessing complete. Output shape: {df.shape}")
    return df


def _mode_or_nan(series: pd.Series):
    """
    Compute the mode of a series, returning NaN if no mode exists.

    Parameters
    ----------
    series : pd.Series
        The series to compute mode for.

    Returns
    -------
    scalar
        The mode value or np.nan.
    """
    mode = series.mode(dropna=True)
    return mode.iloc[0] if not mode.empty else np.nan


def aggregate_ip_data(df_ip: pd.DataFrame, df_provider: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Aggregate inpatient claims to provider-level features.

    Parameters
    ----------
    df_ip : pd.DataFrame
        Preprocessed inpatient data.
    df_provider : pd.DataFrame, optional
        Provider data to merge with.

    Returns
    -------
    pd.DataFrame
        Aggregated IP features per provider.
    """
    if df_provider is not None:
        df_ip = df_provider[['Provider']].merge(df_ip, on='Provider', how='left')

    grouped = df_ip.groupby('Provider', as_index=False).agg(
        IP_Claim_Count=('IP_ClaimID', 'nunique'),
        IP_Benf_Count=('BeneID', 'nunique'),
        Avg_IP_InscClaimAmtReimbursed=('IP_InscClaimAmtReimbursed', 'mean'),
        Avg_IP_DeductibleAmtPaid=('IP_DeductibleAmtPaid', 'mean'),
        Avg_IP_Number_of_Days_in_Hospital=('IP_Number_of_Days_in_Hospital', _mode_or_nan),
        Avg_IP_Claim_Days=('IP_Claim_Days', _mode_or_nan),
        Avg_IP_Unique_Disease_Count=('IP_Unique_Disease_Count', _mode_or_nan),
        Avg_IP_Unique_Treatment_Count=('IP_Unique_Treatment_Count', _mode_or_nan),
    )
    return grouped


def aggregate_op_data(df_op: pd.DataFrame, df_provider: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Aggregate outpatient claims to provider-level features.

    Parameters
    ----------
    df_op : pd.DataFrame
        Preprocessed outpatient data.
    df_provider : pd.DataFrame, optional
        Provider data to merge with.

    Returns
    -------
    pd.DataFrame
        Aggregated OP features per provider.
    """
    if df_provider is not None:
        df_op = df_provider[['Provider']].merge(df_op, on='Provider', how='left')

    grouped = df_op.groupby('Provider', as_index=False).agg(
        OP_Claim_Count=('OP_ClaimID', 'nunique'),
        OP_Benf_Count=('BeneID', 'nunique'),
        Avg_OP_InscClaimAmtReimbursed=('OP_InscClaimAmtReimbursed', 'mean'),
        Avg_OP_DeductibleAmtPaid=('OP_DeductibleAmtPaid', 'mean'),
        Avg_OP_Claim_Days=('OP_Claim_Days', _mode_or_nan),
        Avg_OP_Unique_Disease_Count=('OP_Unique_Disease_Count', _mode_or_nan),
        Avg_OP_Unique_Treatment_Count=('OP_Unique_Treatment_Count', _mode_or_nan),
    )
    return grouped


def read_beneficiary_data(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """
    Read and preprocess beneficiary data from a CSV file.

    Parameters
    ----------
    path : str
        Path to the beneficiary CSV file.
    nrows : int, optional
        Number of rows to read.

    Returns
    -------
    pd.DataFrame
        Preprocessed beneficiary data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty.
    """
    file_path = _resolve_input_path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Beneficiary data file not found: '{path}'")

    logger.info(f"Reading beneficiary data from: {file_path}")
    df = pd.read_csv(file_path, nrows=nrows)
    if df.empty:
        raise ValueError(f"Beneficiary data file is empty: '{path}'")

    # Keep columns used for modeling and buckets; drop raw demographics if available.
    drop_cols = [c for c in ['DOB', 'DOD', 'Gender', 'Race', 'State', 'County'] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    chronic_cols = [
        'ChronicCond_Alzheimer', 'ChronicCond_Heartfailure', 'ChronicCond_KidneyDisease',
        'ChronicCond_Cancer', 'ChronicCond_ObstrPulmonary', 'ChronicCond_Depression',
        'ChronicCond_Diabetes', 'ChronicCond_IschemicHeart', 'ChronicCond_Osteoporasis',
        'ChronicCond_rheumatoidarthritis', 'ChronicCond_stroke'
    ]
    for c in chronic_cols:
        if c in df.columns:
            df[c] = df[c].map({1: 0, 2: 1}).fillna(df[c])

    if 'RenalDiseaseIndicator' in df.columns:
        df['RenalDiseaseIndicator'] = df['RenalDiseaseIndicator'].map({'0': 0, 'Y': 1}).fillna(df['RenalDiseaseIndicator'])

    logger.info(f"Loaded {len(df):,} rows × {df.shape[1]} columns.")
    return df


def aggregate_beneficiary_data(df_benf: pd.DataFrame, df_ip: pd.DataFrame, df_op: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate beneficiary data to provider-level features.

    Parameters
    ----------
    df_benf : pd.DataFrame
        Preprocessed beneficiary data.
    df_ip : pd.DataFrame
        Inpatient claims data.
    df_op : pd.DataFrame
        Outpatient claims data.

    Returns
    -------
    pd.DataFrame
        Aggregated beneficiary features per provider.
    """
    df_ip_op_benf = pd.concat([
        df_ip[['Provider', 'BeneID']],
        df_op[['Provider', 'BeneID']]
    ], axis=0, ignore_index=True)

    df_ip_op_benf_unique = df_ip_op_benf.drop_duplicates().reset_index(drop=True)
    df_benf_provider = df_ip_op_benf_unique.merge(df_benf, on='BeneID', how='left')

    grouped = df_benf_provider.groupby('Provider', as_index=False).agg(
        Total_Beneficiaries=('BeneID', 'count'),
        RenalDisease_Count=('RenalDiseaseIndicator', 'sum'),
        Alzheimer_Count=('ChronicCond_Alzheimer', 'sum'),
        HeartFailure_Count=('ChronicCond_Heartfailure', 'sum'),
        KidneyDisease_Count=('ChronicCond_KidneyDisease', 'sum'),
        Cancer_Count=('ChronicCond_Cancer', 'sum'),
        Pulmonary_Count=('ChronicCond_ObstrPulmonary', 'sum'),
        Depression_Count=('ChronicCond_Depression', 'sum'),
        Diabetes_Count=('ChronicCond_Diabetes', 'sum'),
        IschemicHeart_Count=('ChronicCond_IschemicHeart', 'sum'),
        Osteoporosis_Count=('ChronicCond_Osteoporasis', 'sum'),
        RheumatoidArthritis_Count=('ChronicCond_rheumatoidarthritis', 'sum'),
        Stroke_Count=('ChronicCond_stroke', 'sum'),
        Avg_PartA_Months=('NoOfMonths_PartACov', _mode_or_nan),
        Avg_PartB_Months_Mode=('NoOfMonths_PartBCov', _mode_or_nan),
        Avg_IP_Reimbursement=('IPAnnualReimbursementAmt', 'mean'),
        Avg_IP_Deductible=('IPAnnualDeductibleAmt', 'mean'),
        Avg_OP_Reimbursement=('OPAnnualReimbursementAmt', 'mean'),
        Avg_OP_Deductible=('OPAnnualDeductibleAmt', 'mean'),
    )
    return grouped


def _resolve_output_path(path: str) -> Path:
    """
    Resolve output path relative to workspace root if relative.

    Parameters
    ----------
    path : str
        The output path string.

    Returns
    -------
    Path
        The resolved absolute path.
    """
    p = Path(path)
    if p.is_absolute():
        return p
    workspace_root = Path(__file__).resolve().parents[1]
    return workspace_root / path


def prepare_model_input(
    provider_path: str,
    ip_path: str,
    op_path: str,
    benf_path: str,
    output_path: str,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Prepare the complete model input dataset by reading, preprocessing, aggregating, and merging all data sources.

    Parameters
    ----------
    provider_path : str
        Path to provider data CSV.
    ip_path : str
        Path to inpatient data CSV.
    op_path : str
        Path to outpatient data CSV.
    benf_path : str
        Path to beneficiary data CSV.
    output_path : str
        Path to save the output CSV.
    nrows : int, optional
        Number of rows to read from each file.

    Returns
    -------
    pd.DataFrame
        The final merged dataset.
    """
    df_provider = read_provider_data(provider_path, nrows=nrows)
    df_ip = read_ip_data(ip_path, nrows=nrows)
    df_op = read_op_data(op_path, nrows=nrows)
    df_benf = read_beneficiary_data(benf_path, nrows=nrows)

    df_ip_final = aggregate_ip_data(df_ip, df_provider)
    df_op_final = aggregate_op_data(df_op, df_provider)
    df_benf_final = aggregate_beneficiary_data(df_benf, df_ip, df_op)

    df_final = df_provider.merge(df_ip_final, on='Provider', how='left')
    df_final = df_final.merge(df_op_final, on='Provider', how='left')
    df_final = df_final.merge(df_benf_final, on='Provider', how='left')

    df_final.fillna(0, inplace=True)

    output_file = _resolve_output_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    df_final.to_csv(output_file, index=False)
    logger.info(f"Saved merged model input data to: {output_file} (shape={df_final.shape})")

    return df_final


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare model input data from raw datasets.')
    parser.add_argument('--provider', default='../Train_Test_Data/Train_Master_Providers.csv', help='Provider data path')
    parser.add_argument('--ip', default='../Train_Test_Data/Train_Inpatientdata.csv', help='Inpatient data path')
    parser.add_argument('--op', default='../Train_Test_Data/Train_Outpatientdata.csv', help='Outpatient data path')
    parser.add_argument('--benf', default='../Train_Test_Data/Train_Beneficiarydata.csv', help='Beneficiary data path')
    parser.add_argument('--output', default='../Model_Input_Data/Model_Input_Data.csv', help='Output path for prepared data')
    parser.add_argument('--nrows', type=int, default=None, help='Optional number of rows to read for each input file')
    args = parser.parse_args()

    prepare_model_input(
        provider_path=args.provider,
        ip_path=args.ip,
        op_path=args.op,
        benf_path=args.benf,
        output_path=args.output,
        nrows=args.nrows,
    )
    