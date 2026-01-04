from pathlib import Path
import pickle
from typing import List, Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def extract_brand(df: pd.DataFrame, car_name_col: str = 'Car_Name') -> pd.DataFrame:
    """Extract brand from car name (first token) and drop original Car_Name column."""
    out = df.copy()
    if car_name_col in out.columns:
        try:
            out['Brand'] = out[car_name_col].astype(str).str.split().str[0].str.upper().replace({'NAN': np.nan})
            out.drop(columns=[car_name_col], inplace=True, errors='ignore')
        except Exception:
            out['Brand'] = np.nan
    return out


def choose_columns(df: pd.DataFrame,
                   preferred_numeric: Optional[List[str]] = None,
                   target: str = 'Selling_Price') -> (List[str], List[str]):
    """Return (numeric_cols, categorical_cols) to use for modeling.

    preferred_numeric: names like ['Mileage_num','Engine_num','Power_num','Age'] preferred if present.
    Categorical columns are object dtype (excluding target).
    """
    if preferred_numeric is None:
        preferred_numeric = ['Mileage_num', 'Engine_num', 'Power_num', 'Age']

    numeric_cols = [c for c in preferred_numeric if c in df.columns]
    # include any other numeric columns except the target
    other_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    other_numeric = [c for c in other_numeric if c not in numeric_cols and c != target]
    numeric_cols = numeric_cols + other_numeric
    # categorical columns - object dtype and Brand if present
    cat_cols = [c for c in df.select_dtypes(include=['object']).columns if c != target]
    if 'Brand' in df.columns and 'Brand' not in cat_cols:
        cat_cols.append('Brand')
    return numeric_cols, cat_cols


def fill_missing(X: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    out = X.copy()
    for c in numeric_cols:
        if c in out.columns:
            try:
                med = out[c].median()
                out[c] = out[c].fillna(med)
            except Exception:
                out[c] = out[c].fillna(0)
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].fillna('Unknown')
    return out


def encode_and_scale(X: pd.DataFrame, numeric_cols: List[str], cat_cols: List[str]):
    """Scale numeric columns and one-hot encode categoricals. Returns (X_transformed, scaler, final_numeric_cols).

    - Numeric columns are scaled with StandardScaler.
    - Categorical columns are one-hot encoded via pandas.get_dummies (drop_first=True).
    """
    out = X.copy()
    scaler = StandardScaler()
    if len(numeric_cols) > 0:
        # Fit scaler on numeric columns
        try:
            out[numeric_cols] = scaler.fit_transform(out[numeric_cols])
        except Exception:
            # Fallback: try to coerce to numeric then scale
            for c in numeric_cols:
                out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0)
            out[numeric_cols] = scaler.fit_transform(out[numeric_cols])
    # One-hot encode categoricals
    if len(cat_cols) > 0:
        out = pd.get_dummies(out, columns=[c for c in cat_cols if c in out.columns], drop_first=True)
    return out, scaler


def save_preprocessor(pickle_path: Path, scaler, features: List[str], numeric_features: List[str]):
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pickle_path, 'wb') as f:
        pickle.dump({'scaler': scaler, 'features': features, 'numeric_features': numeric_features}, f)


def process_and_save(cleaned_csv: str = None,
                     out_dir: str = None,
                     test_size: float = 0.2,
                     random_state: int = 42):
    """Full feature engineering pipeline: reads cleaned CSV, creates features, splits and saves outputs.

    Defaults:
      cleaned_csv: data/processed/cleaned_car_data.csv
      out_dir: project folders for outputs (data/processed, models, outputs/results)
    """
    project_root = Path.cwd()
    if cleaned_csv is None:
     cleaned_csv = Path(r'D:\\Livstream\\ Car Price Prediction with Machine Learning\\notebooks\\data\\processed\\cleaned_car_data.csv')
    else:
     cleaned_csv = Path(cleaned_csv)

    data_processed = project_root / 'data' / 'processed'
    models_dir = project_root / 'models'
    results_dir = project_root / 'outputs' / 'results'

    data_processed.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train_out = data_processed / 'X_train.csv'
    X_test_out = data_processed / 'X_test.csv'
    y_train_out = data_processed / 'y_train.csv'
    y_test_out = data_processed / 'y_test.csv'
    preproc_out = models_dir / 'preprocessor.pkl'
    features_out = results_dir / 'features_list.csv'

    # Load
    if not cleaned_csv.exists():
        raise FileNotFoundError(f'Cleaned CSV not found at {cleaned_csv}. Run preprocessing first.')
    df = pd.read_csv(cleaned_csv, low_memory=False)

    # Ensure target
    target = 'Selling_Price'
    if target not in df.columns:
        raise ValueError('Target column Selling_Price not found in cleaned data.')

    # Extract Brand and drop Car_Name if present
    df2 = extract_brand(df, car_name_col='Car_Name')

    # Choose columns
    numeric_cols, cat_cols = choose_columns(df2, preferred_numeric=['Mileage_num', 'Engine_num', 'Power_num', 'Age'], target=target)

    # Prepare X,y
    X = df2.drop(columns=[target])
    y = df2[target]

    # Fill missing
    X = fill_missing(X, numeric_cols, cat_cols)

    # Encode and scale
    X_transformed, scaler = encode_and_scale(X, numeric_cols, cat_cols)

    # Record feature list
    features = X_transformed.columns.tolist()
    pd.DataFrame({'feature': features}).to_csv(features_out, index=False)

    # Save preprocessor
    save_preprocessor(preproc_out, scaler, features, numeric_cols)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=test_size, random_state=random_state)

    # Save splits
    X_train.to_csv(X_train_out, index=False)
    X_test.to_csv(X_test_out, index=False)
    y_train.to_csv(y_train_out, index=False, header=True)
    y_test.to_csv(y_test_out, index=False, header=True)

    return {
        'X_train': str(X_train_out),
        'X_test': str(X_test_out),
        'y_train': str(y_train_out),
        'y_test': str(y_test_out),
        'preprocessor': str(preproc_out),
        'features_list': str(features_out)
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Feature engineering pipeline')
    parser.add_argument('--cleaned', help='Path to cleaned CSV (default: data/processed/cleaned_car_data.csv)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    args = parser.parse_args()

    try:
        out = process_and_save(cleaned_csv=args.cleaned, test_size=args.test_size, random_state=args.random_state)
        print('Feature engineering complete. Outputs:', out)
    except Exception as e:
        print('Feature engineering failed:', e)
