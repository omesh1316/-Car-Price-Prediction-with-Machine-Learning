from pathlib import Path
import pickle
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def load_splits(x_train_path: Path, y_train_path: Path, x_test_path: Path, y_test_path: Path):
    x_train = pd.read_csv(x_train_path)
    x_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()
    return x_train, y_train, x_test, y_test


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series,
                        n_estimators: int = 100, random_state: int = 42, n_jobs: int = -1) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=n_jobs)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = float(np.sqrt(mse))
    return {'r2': float(r2), 'mae': float(mae), 'mse': float(mse), 'rmse': rmse, 'preds': preds}


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{k: v for k, v in metrics.items() if k != 'preds'}])
    df.to_csv(path, index=False)


def save_predictions(y_true: pd.Series, y_pred: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred})
    out.to_csv(path, index=False)


def save_feature_importances(model, feature_names: list, path: Path) -> None:
    if not hasattr(model, 'feature_importances_'):
        return
    fi = model.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': fi}).sort_values('importance', ascending=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fi_df.to_csv(path, index=False)


def plot_pred_vs_actual(y_true: pd.Series, y_pred: np.ndarray, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    mn = min(np.min(y_true), np.min(y_pred))
    mx = max(np.max(y_true), np.max(y_pred))
    plt.plot([mn, mx], [mn, mx], 'r--')
    plt.xlabel('Actual Selling_Price')
    plt.ylabel('Predicted Selling_Price')
    plt.title('Predicted vs Actual (Test)')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_residuals(y_true: pd.Series, y_pred: np.ndarray, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    residuals = y_true - y_pred
    plt.figure(figsize=(7, 4))
    plt.hist(residuals, bins=40, edgecolor='k')
    plt.title('Residuals Distribution (Test)')
    plt.xlabel('Residual')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def train_and_save(x_train_path: str,
                   y_train_path: str,
                   x_test_path: str,
                   y_test_path: str,
                   model_out: Optional[str] = None,
                   metrics_out: Optional[str] = None,
                   predictions_out: Optional[str] = None,
                   importances_out: Optional[str] = None,
                   graph_dir: Optional[str] = None,
                   n_estimators: int = 100,
                   random_state: int = 42) -> Dict[str, str]:
    project_root = Path.cwd()
    X_train_p = Path(x_train_path)
    X_test_p = Path(x_test_path)
    y_train_p = Path(y_train_path)
    y_test_p = Path(y_test_path)

    model_out = Path(model_out) if model_out else project_root / 'models' / 'car_price_model.pkl'
    metrics_out = Path(metrics_out) if metrics_out else project_root / 'outputs' / 'results' / 'model_metrics.csv'
    predictions_out = Path(predictions_out) if predictions_out else project_root / 'outputs' / 'results' / 'predictions_test.csv'
    importances_out = Path(importances_out) if importances_out else project_root / 'outputs' / 'results' / 'feature_importances.csv'
    graph_dir = Path(graph_dir) if graph_dir else project_root / 'outputs' / 'graphs'
    graph_dir.mkdir(parents=True, exist_ok=True)

    # Load
    X_train, y_train, X_test, y_test = load_splits(X_train_p, y_train_p, X_test_p, y_test_p)

    # Train
    model = train_random_forest(X_train, y_train, n_estimators=n_estimators, random_state=random_state)

    # Save model
    save_model(model, model_out)

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test)

    # Save metrics and predictions
    save_metrics(metrics, metrics_out)
    save_predictions(y_test.reset_index(drop=True), metrics['preds'], predictions_out)

    # Save feature importances
    save_feature_importances(model, X_train.columns.tolist(), importances_out)

    # Plots
    plot_pred_vs_actual(y_test.reset_index(drop=True), metrics['preds'], graph_dir / 'predicted_vs_actual.png')
    plot_residuals(y_test.reset_index(drop=True), metrics['preds'], graph_dir / 'residuals_histogram.png')

    return {
        'model': str(model_out),
        'metrics': str(metrics_out),
        'predictions': str(predictions_out),
        'importances': str(importances_out),
        'graphs': str(graph_dir)
    }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train RandomForestRegressor and save artifacts')
    parser.add_argument('--x-train', required=True, help='Path to X_train.csv')
    parser.add_argument('--y-train', required=True, help='Path to y_train.csv')
    parser.add_argument('--x-test', required=True, help='Path to X_test.csv')
    parser.add_argument('--y-test', required=True, help='Path to y_test.csv')
    parser.add_argument('--model-out', help='Path to save trained model (pickle)')
    parser.add_argument('--metrics-out', help='Path to save metrics CSV')
    parser.add_argument('--predictions-out', help='Path to save predictions CSV')
    parser.add_argument('--importances-out', help='Path to save feature importances CSV')
    parser.add_argument('--graphs-dir', help='Directory to save graphs')
    parser.add_argument('--n-estimators', type=int, default=100)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    try:
        out = train_and_save(
            x_train_path=args.x_train,
            y_train_path=args.y_train,
            x_test_path=args.x_test,
            y_test_path=args.y_test,
            model_out=args.model_out,
            metrics_out=args.metrics_out,
            predictions_out=args.predictions_out,
            importances_out=args.importances_out,
            graph_dir=args.graphs_dir,
            n_estimators=args.n_estimators,
            random_state=args.random_state
        )
        print('Training complete. Artifacts saved:', out)
    except Exception as e:
        print('Training failed:', e)
