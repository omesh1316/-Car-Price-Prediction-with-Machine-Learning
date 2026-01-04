from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def load_model(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_test_data(x_test_path: Path, y_test_path: Path):
    X_test = pd.read_csv(x_test_path)
    y_test = pd.read_csv(y_test_path).squeeze()
    return X_test, y_test


def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-8, y_true))) * 100)
    return {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse, 'mape_pct': mape}


def save_metrics(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([metrics])
    df.to_csv(path, index=False)


def save_predictions(y_true, y_pred, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({'y_true': np.asarray(y_true).ravel(), 'y_pred': np.asarray(y_pred).ravel()})
    df.to_csv(path, index=False)


def plot_pred_vs_actual(y_true, y_pred, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7,5))
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


def plot_residuals(y_true, y_pred, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    residuals = np.asarray(y_true).ravel() - np.asarray(y_pred).ravel()
    plt.figure(figsize=(7,4))
    plt.hist(residuals, bins=40, edgecolor='k')
    plt.title('Residuals Distribution (Test)')
    plt.xlabel('Residual')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_calibration(y_true, y_pred, outpath: Path, n_bins: int = 10):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        bins = pd.qcut(np.asarray(y_pred).ravel(), q=n_bins, duplicates='drop')
        calib = pd.DataFrame({'y_true': np.asarray(y_true).ravel(), 'y_pred': np.asarray(y_pred).ravel(), 'bin': bins})
        calib_group = calib.groupby('bin').agg({'y_true': 'mean', 'y_pred': 'mean'}).reset_index()
        plt.figure(figsize=(6,5))
        plt.plot(calib_group['y_pred'], calib_group['y_true'], 'o-')
        plt.plot([calib_group['y_pred'].min(), calib_group['y_pred'].max()], [calib_group['y_pred'].min(), calib_group['y_pred'].max()], 'r--')
        plt.xlabel('Mean Predicted')
        plt.ylabel('Mean Actual')
        plt.title('Calibration by Predicted Quantile')
        plt.tight_layout()
        plt.savefig(outpath, dpi=150)
        plt.close()
    except Exception:
        # If calibration plot fails (e.g., not enough variability), skip quietly
        pass


def save_feature_importances(model, feature_names, outpath: Path):
    try:
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            df = pd.DataFrame({'feature': feature_names, 'importance': fi})
            df.sort_values('importance', ascending=False, inplace=True)
            outpath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(outpath, index=False)
    except Exception:
        pass


def write_report(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('Model Evaluation Report\n')
            f.write('====================\n')
            for k, v in metrics.items():
                f.write(f'{k}: {v:.4f}\n')
            f.write('\nNotes: All paths are relative. See metrics and predictions CSV for details.')
    except Exception:
        pass


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate trained model on test set')
    parser.add_argument('--model', help='Path to trained model pickle', default='models/car_price_model.pkl')
    parser.add_argument('--x-test', help='Path to X_test.csv', default='data/processed/X_test.csv')
    parser.add_argument('--y-test', help='Path to y_test.csv', default='data/processed/y_test.csv')
    parser.add_argument('--metrics-out', help='Path to save metrics CSV', default='outputs/results/evaluation_metrics.csv')
    parser.add_argument('--predictions-out', help='Path to save predictions CSV', default='outputs/results/evaluation_predictions.csv')
    parser.add_argument('--feature-importances-out', help='Path to save feature importances CSV', default='outputs/results/evaluation_feature_importances.csv')
    parser.add_argument('--plots-dir', help='Directory to save plots', default='outputs/graphs')
    parser.add_argument('--report-out', help='Path to write evaluation report', default='outputs/results/model_evaluation_report.txt')

    args = parser.parse_args()

    model_path = Path(args.model)
    x_test_path = Path(args.x_test)
    y_test_path = Path(args.y_test)
    metrics_out = Path(args.metrics_out)
    preds_out = Path(args.predictions_out)
    fi_out = Path(args.feature_importances_out)
    plots_dir = Path(args.plots_dir)
    report_out = Path(args.report_out)

    try:
        model = load_model(model_path)
        X_test, y_test = load_test_data(x_test_path, y_test_path)
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        save_metrics(metrics, metrics_out)
        save_predictions(y_test, preds, preds_out)
        save_feature_importances(model, X_test.columns.tolist(), fi_out)
        plot_pred_vs_actual(y_test, preds, plots_dir / 'eval_predicted_vs_actual.png')
        plot_residuals(y_test, preds, plots_dir / 'eval_residuals_hist.png')
        plot_calibration(y_test, preds, plots_dir / 'eval_calibration.png')
        write_report(metrics, report_out)
        print('Evaluation complete. Metrics saved to', metrics_out)
    except Exception as e:
        print('Evaluation failed:', e)
