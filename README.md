# Car Price Prediction with Machine Learning

Small project to predict used-car selling prices and expose a Streamlit UI for single-row or batch predictions.

## Project structure

- `app/` - Streamlit app (`app/app.py`).
- `data/raw/` - original CSV (`car data.csv`).
- `data/processed/` - cleaned data and train/test splits.
- `models/` - trained `car_price_model.pkl` and `preprocessor.pkl` (may also exist under `src/models` or `notebooks/models`).
- `notebooks/` - notebooks implementing EDA, preprocessing, feature engineering, training, evaluation (`01_..` → `05_..`).
- `outputs/` - saved graphs and CSV results.
- `src/` - reusable Python modules for preprocessing, feature engineering, training, evaluation, and utilities.

## Quickstart

1. Create a Python environment (recommended) and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run notebooks in order (optional but recommended) to reproduce preprocessing, feature engineering, training and evaluation:

- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_data_preprocessing.ipynb`
- `notebooks/03_feature_engineering.ipynb`
- `notebooks/04_model_training.ipynb`
- `notebooks/05_model_evaluation.ipynb`

3. Or run the `src` scripts/CLIs to preprocess, engineer features, train and evaluate (see `src/*.py` for details).

4. Start the Streamlit app for inference:

```bash
streamlit run app/app.py
```

## Streamlit app usage

- The app loads the trained model and preprocessor from the first-existing of `models/`, `src/models/`, or `notebooks/models/`.
- Use the left sidebar to input a single car's details or upload a CSV with columns similar to the training data (`Car_Name, Year, Mileage, Engine, Power, Fuel_Type, Transmission, Owner`).
- After predicting, the app exposes a **Convert a predicted value** panel in the sidebar where you can:
  - Type a base currency (3-letter code, e.g., `INR`, `USD`).
  - Enter a prediction value (e.g., `0.78`) to see conversions to many world currencies, or leave blank to use the app's last prediction if present in session state.
- Conversion rates are fetched from `https://api.exchangerate.host/latest`. A small offline fallback table is used if the API is unreachable.

## Reproducing training

- Ensure `data/raw/car data.csv` is present.
- Run preprocessing → feature engineering → training as in the notebooks or call the functions in `src/`.
- Trained model pickle: `models/car_price_model.pkl` and preprocessor pickle: `models/preprocessor.pkl`.

## Notes & troubleshooting

- If the app cannot find model artifacts, check `src/models/` and `notebooks/models/` — the app looks there as well.
- If you see a scaler/feature-name mismatch, re-run `03_feature_engineering.ipynb` and `04_model_training.ipynb` from the project root to regenerate `preprocessor.pkl` and `car_price_model.pkl`.
- To make currency conversions appear automatically after clicking Predict, the app uses `st.session_state['last_prediction']` if available. If you want this behavior guaranteed, I can patch the Predict button logic to always write the produced numeric prediction into session state.

## License & contact

This project is for educational/demo purposes. For questions or to request changes, open an issue in the repository or contact the maintainer.
