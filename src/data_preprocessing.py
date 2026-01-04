from pathlib import Path
import re
from datetime import datetime
import pandas as pd
import numpy as np
import argparse


def extract_number(val):
    """Extract the first numeric value from a string and return as float.

    Returns np.nan if no number found or value is NaN.
    """
    if pd.isna(val):
        return np.nan
    s = str(val)
    # Remove commas and look for a float/int pattern
    s_clean = s.replace(',', '')
    m = re.search(r"([0-9]+\.?[0-9]*)", s_clean)
    if not m:
        return np.nan
    try:
        return float(m.group(1))
    except Exception:
        return np.nan


def safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert a pandas Series with messy numeric strings into floats using extract_number."""
    return series.apply(extract_number)


def compute_age_from_year(series: pd.Series) -> pd.Series:
    """Compute approximate age from a `Year` column. Non-numeric years become NaN."""
    now = datetime.now().year
    def _age(x):
        try:
            if pd.isna(x):
                return np.nan
            y = int(float(x))
            return now - y if y > 0 and y <= now else np.nan
        except Exception:
            return np.nan
    return series.apply(_age)


def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading/trailing whitespace from object columns (inplace on a copy)."""
    out = df.copy()
    obj_cols = out.select_dtypes(include=['object']).columns
    for c in obj_cols:
        try:
            out[c] = out[c].astype(str).str.strip().replace({'nan': np.nan})
        except Exception:
            # If conversion fails, leave column as-is
            pass
    return out


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def ensure_target_numeric(df: pd.DataFrame, target: str = 'Selling_Price') -> pd.DataFrame:
    """Ensure target column is numeric (coerce errors to NaN)."""
    out = df.copy()
    if target in out.columns:
        out[target] = pd.to_numeric(out[target], errors='coerce')
    return out


def convert_columns_numeric(df: pd.DataFrame, cols: list, suffix: str = '_num') -> pd.DataFrame:
    """For each col in cols present in df, create a numeric column named col+suffix using safe_to_numeric."""
    out = df.copy()
    for col in cols:
        if col in out.columns:
            try:
                out[col + suffix] = safe_to_numeric(out[col])
            except Exception:
                out[col + suffix] = np.nan
    return out


def drop_missing_target(df: pd.DataFrame, target: str = 'Selling_Price') -> pd.DataFrame:
    """Drop rows where target is missing. If target not present, returns df unchanged."""
    if target in df.columns:
        return df[df[target].notna()].reset_index(drop=True)
    return df


def preprocess_dataframe(df: pd.DataFrame,
                         year_col: str = 'Year',
                         numeric_cols: list = None,
                         target: str = 'Selling_Price') -> pd.DataFrame:
    """Apply a sequence of safe cleaning steps and return cleaned DataFrame.

    Steps applied (only if relevant columns exist):
    - Strip whitespace from object columns
    - Compute `Age` from `year_col` if present
    - Convert provided numeric-like columns (e.g., Mileage, Engine, Power) to numeric with suffix `_num`
    - Ensure target numeric
    - Drop exact duplicates
    - Drop rows with missing target

    The function never mutates the input `df`.
    """
    if numeric_cols is None:
        numeric_cols = ['Mileage', 'Engine', 'Power']

    out = strip_whitespace(df)

    # Year -> Age
    if year_col in out.columns:
        try:
            out[year_col] = pd.to_numeric(out[year_col], errors='coerce')
            out['Age'] = compute_age_from_year(out[year_col])
        except Exception:
            out['Age'] = np.nan

    # Convert messy numeric columns
    out = convert_columns_numeric(out, numeric_cols, suffix='_num')

    # Ensure target numeric
    out = ensure_target_numeric(out, target=target)

    # Drop duplicates
    out = drop_exact_duplicates(out)

    # Drop rows missing target
    out = drop_missing_target(out, target=target)

    return out


def load_csv(path) -> pd.DataFrame:
    """Load CSV from path with basic error messaging."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'CSV file not found at {p}')
    return pd.read_csv(p, low_memory=False)


def save_csv(df: pd.DataFrame, path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def preprocess_and_save(raw_csv: str = None,
                        out_csv: str = None,
                        report_txt: str = None,
                        sample_out: str = None) -> dict:
    """Run full preprocessing pipeline and save outputs.

    Returns a small report dict with keys: rows_in, rows_out, messages.
    """
    project_root = Path('notebooks/data/car data.csv').resolve().parents[1]

    raw_csv = 'notebooks/data/car data.csv'
    out_csv = 'notebooks/data/processed/cleaned_car_data.csv'
    report_txt = 'notebooks/outputs/results/preprocessing_report.txt'
    sample_out = 'D:\\Livstream\\ Car Price Prediction with Machine Learning\\notebooks\\outputs\\results\\cleaned_sample.csv'

    report_lines = []
    try:
        df = load_csv(raw_csv)
    except Exception as e:
        raise

    report_lines.append(f'Raw shape: {df.shape}')

    try:
        df_clean = preprocess_dataframe(df)
        report_lines.append(f'Cleaned shape: {df_clean.shape}')
    except Exception as e:
        report_lines.append(f'Error during preprocess_dataframe: {e}')
        raise

    # Save cleaned data
    try:
        save_csv(df_clean, out_csv)
        report_lines.append(f'Wrote cleaned CSV to {out_csv}')
    except Exception as e:
        report_lines.append(f'Failed to write cleaned CSV: {e}')

    # Save report
    try:
        report_txt.parent.mkdir(parents=True, exist_ok=True)
        with open(report_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        report_lines.append(f'Wrote preprocessing report to {report_txt}')
    except Exception as e:
        report_lines.append(f'Could not write report: {e}')

    # Save a small sample
    try:
        sample = df_clean.sample(n=min(100, len(df_clean)), random_state=42)
        save_csv(sample, sample_out)
        report_lines.append(f'Wrote sample CSV to {sample_out}')
    except Exception as e:
        report_lines.append(f'Could not save sample: {e}')

    return {'rows_in': int(df.shape[0]), 'rows_out': int(df_clean.shape[0]), 'messages': report_lines}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess raw car data and save cleaned CSV.')
    parser.add_argument('--raw', help='Path to raw CSV (default: data/raw/car_data.csv)')
    parser.add_argument('--out', help='Path to write cleaned CSV (default: data/processed/cleaned_car_data.csv)')
    parser.add_argument('--report', help='Path to write preprocessing report')
    parser.add_argument('--sample', help='Path to write small sample CSV')
    args = parser.parse_args()

    try:
        result = preprocess_and_save(raw_csv=args.raw, out_csv=args.out, report_txt=args.report, sample_out=args.sample)
        print('Preprocessing complete:', result)
    except Exception as e:
        print('Preprocessing failed:', e)
