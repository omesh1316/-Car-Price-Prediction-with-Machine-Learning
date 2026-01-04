import sys
from pathlib import Path
import requests
import pandas as pd

# Ensure project root is on sys.path so `from src import ...` works when running this file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from src import data_preprocessing as dp
from src import feature_engineering as fe
from src.utils import resolve_path

# Resolve paths relative to project root; prefer project `models/` then `src/models/` then `notebooks/models/`
def first_existing(*paths):
    for p in paths:
        rp = resolve_path(p)
        if rp.exists():
            return rp
    return resolve_path(paths[0])

MODEL_PATH = first_existing('models/car_price_model.pkl', 'src/models/car_price_model.pkl', 'notebooks/models/car_price_model.pkl')
PREPROC_PATH = first_existing('models/preprocessor.pkl', 'src/models/preprocessor.pkl', 'notebooks/models/preprocessor.pkl')
FEATURES_LIST = first_existing('outputs/results/features_list.csv', 'src/outputs/results/features_list.csv', 'notebooks/outputs/results/features_list.csv')

st.set_page_config(page_title='Car Price Price Predictor', layout='centered')

st.title('Car Price Prediction')
st.write('Enter car details on the left and click Predict, or upload a CSV with the same columns.')

@st.cache_data
def load_model_and_preprocessor(model_path, preproc_path):
    model = None
    preproc = None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception:
        model = None
    try:
        with open(preproc_path, 'rb') as f:
            preproc = pickle.load(f)
    except Exception:
        preproc = None
    return model, preproc

model, preproc = load_model_and_preprocessor(MODEL_PATH, PREPROC_PATH)

if model is None or preproc is None:
    st.error('Model or preprocessor not found. Please ensure you have trained the model and have models/car_price_model.pkl and models/preprocessor.pkl')
    st.stop()

expected_features = preproc.get('features', [])
numeric_features = preproc.get('numeric_features', [])
scaler = preproc.get('scaler', None)

# Sidebar inputs
st.sidebar.header('Manual input')
car_name = st.sidebar.text_input('Car Name (e.g., Maruti Ciaz)')
year = st.sidebar.text_input('Year (e.g., 2017)')
mileage = st.sidebar.text_input('Mileage (e.g., 18.0)')
engine = st.sidebar.text_input('Engine (cc) (e.g., 1197)')
power = st.sidebar.text_input('Power (e.g., 88.5)')
fuel_type = st.sidebar.selectbox('Fuel Type', options=['Unknown','Petrol','Diesel','CNG','LPG','Electric'])
transmission = st.sidebar.selectbox('Transmission', options=['Unknown','Manual','Automatic'])
owner = st.sidebar.text_input('Owner (e.g., First Owner)')

uploaded_file = st.file_uploader('Or upload CSV (columns: Car_Name, Year, Mileage, Engine, Power, Fuel_Type, Transmission, Owner)', type=['csv'])

def prepare_row(row: pd.DataFrame):
    # Apply same preprocessing as training pipeline on a single-row dataframe
    df = row.copy()
    # Preprocessing: creates Mileage_num, Engine_num, Power_num, Age etc.
    df = dp.preprocess_dataframe(df)
    # Extract brand and drop Car_Name
    df = fe.extract_brand(df, car_name_col='Car_Name')
    # One-hot encode categoricals
    # Fill missing
    # Ensure categorical filling to avoid missing columns
    # For single row, fill missing numeric cols
    for c in numeric_features:
        if c not in df.columns:
            df[c] = np.nan
    # fill missing numeric with median-like fallback (0)
    for c in numeric_features:
        try:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        except Exception:
            df[c] = 0
    # For categoricals, fill with 'Unknown'
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in obj_cols:
        df[c] = df[c].fillna('Unknown')
    # Get dummies and then reindex to expected features
    df_enc = pd.get_dummies(df, drop_first=True)
    # Scale numeric features using saved scaler if available
    if scaler is not None and len(numeric_features) > 0:
        try:
            # Ensure we pass a DataFrame with the same column names the scaler was fitted on
            # Reindex to all numeric_features (missing filled with 0) and provide DataFrame to scaler
            df_num = df_enc.reindex(columns=numeric_features, fill_value=0)[numeric_features].astype(float)
            scaled = scaler.transform(df_num)
            # Assign scaled values back to df_enc for those numeric columns
            for i, col in enumerate(numeric_features):
                if col in df_enc.columns:
                    df_enc[col] = scaled[:, i]
        except Exception:
            pass
    # Reindex to expected features
    final = pd.DataFrame(columns=expected_features)
    for c in expected_features:
        if c in df_enc.columns:
            try:
                val = df_enc[c].iloc[0]
            except Exception:
                # fallback to scalar or 0
                _col = df_enc.get(c, 0)
                val = _col.iloc[0] if hasattr(_col, 'iloc') else _col
        else:
            val = 0
        final.loc[0, c] = val
    # Ensure numeric dtype
    final = final.fillna(0).astype(float)
    return final

if uploaded_file is not None:
    df_in = pd.read_csv(uploaded_file)
    preds = []
    final_rows = []
    for _, row in df_in.iterrows():
        row_df = pd.DataFrame([row])
        prepared = prepare_row(row_df)
        pred = model.predict(prepared)[0]
        preds.append(pred)
        final_rows.append(prepared.iloc[0].to_dict())
    res_df = pd.DataFrame(final_rows)
    res_df['prediction'] = preds
    st.subheader('Batch predictions')
    st.dataframe(res_df)
    csv = res_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download predictions CSV', csv, file_name='predictions.csv', mime='text/csv')
else:
    if st.sidebar.button('Predict'):
        input_df = pd.DataFrame([{
            'Car_Name': car_name,
            'Year': year,
            'Mileage': mileage,
            'Engine': engine,
            'Power': power,
            'Fuel_Type': fuel_type,
            'Transmission': transmission,
            'Owner': owner
        }])
        try:
            prepared = prepare_row(input_df)
            pred = model.predict(prepared)[0]
            st.subheader('Predicted Selling Price')
            st.write(f"{pred:.2f}")
        except Exception as e:
            st.error(f'Prediction failed: {e}')

st.markdown('---')
st.write('Note: This interface uses the trained model and preprocessor saved in `models/`. Ensure model and preprocessor are up-to-date.')

# ----------------------
# Currency conversion helper
# ----------------------

@st.cache_data(ttl=60*60)
def fetch_exchange_rates(base: str = 'INR') -> dict:
    """Fetch latest exchange rates (rates keyed by currency code) using exchangerate.host.
    Falls back to a small hardcoded set if the API fails.
    """
    url = f"https://api.exchangerate.host/latest?base={base}"
    try:
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        data = r.json()
        rates = data.get('rates', {})
        if not rates:
            raise RuntimeError('No rates returned')
        return rates
    except Exception:
        # Minimal fallback rates (approximate) if offline
        fallback = {
            'USD': 0.012,
            'EUR': 0.011,
            'GBP': 0.0095,
            'JPY': 1.75,
            'AUD': 0.018,
            'CAD': 0.016,
            'SGD': 0.016,
            'CHF': 0.011,
            'CNY': 0.087,
            'INR': 1.0
        }
        return fallback


def show_conversions(value: float, base: str = 'INR'):
    """Display conversion table for `value` expressed in `base` currency to many currencies."""
    if value is None:
        st.info('No prediction value provided to convert.')
        return
    rates = fetch_exchange_rates(base=base)
    keys = sorted(rates.keys())
    rows = []
    for k in keys:
        try:
            rows.append({'currency': k, 'rate': rates[k], 'value': float(value) * float(rates[k])})
        except Exception:
            pass
    df_rates = pd.DataFrame(rows)
    # Format values nicely
    df_rates['value'] = df_rates['value'].map(lambda x: f"{x:,.4f}")
    df_rates['rate'] = df_rates['rate'].map(lambda x: f"{x:,.6f}")
    st.subheader(f'Converted values (base={base})')
    st.dataframe(df_rates.set_index('currency'))


# Small UI: allow entering a numeric prediction to convert (or use last_prediction in session state)
st.sidebar.header('Convert a predicted value')
base_currency = st.sidebar.text_input('Base currency (3-letter code)', value='INR', max_chars=3)
pred_val_input = st.sidebar.text_input('Prediction value (in base currency)', value='')

pred_to_convert = None
if pred_val_input.strip() != '':
    try:
        pred_to_convert = float(pred_val_input)
    except Exception:
        st.sidebar.error('Enter a numeric prediction value, e.g. 0.78')
elif 'last_prediction' in st.session_state:
    try:
        pred_to_convert = float(st.session_state.get('last_prediction'))
    except Exception:
        pred_to_convert = None

if pred_to_convert is not None:
    show_conversions(pred_to_convert, base=base_currency.upper())
