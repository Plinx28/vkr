"""
Оценка обученных моделей на тестовой выборке из data/splits.
"""

import argparse
import logging
import time
from pathlib import Path
import pandas as pd

from utils import set_seed, get_dataset_splits, compute_metrics
from models.logistic_regression import LogisticRegressionModel
from models.xgboost_model import XGBoostModel
from models.mlp_model import MLPModel
from models.autoencoder_model import AutoencoderModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "logistic_regression": LogisticRegressionModel,
    "xgboost": XGBoostModel,
    "mlp": MLPModel,
    "autoencoder": AutoencoderModel,
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models on test split.")
    parser.add_argument("--model", type=str, default="all",
                        choices=list(MODEL_REGISTRY.keys()) + ["all"],
                        help="Model to evaluate (default: all).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit test samples (speeds up evaluation).")
    parser.add_argument("--output_dir", type=str, default="reports/metrics")
    return parser.parse_args()

def evaluate_model(model_name, model, X_test, y_test):
    """Вычисляет метрики и время инференса."""
    logger.info(f"Evaluating {model_name}...")
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
    else:
        y_proba = None
    elapsed = time.perf_counter() - start

    metrics = compute_metrics(y_test, y_pred, y_proba)
    metrics["inference_time_sec"] = round(elapsed, 4)
    metrics["model"] = model_name
    return metrics

def main():
    args = parse_args()
    set_seed(42)

    logger.info("Loading cached test split...")
    _, X_test, _, y_test = get_dataset_splits(force_reload=False,
                                              max_samples=args.max_samples)
    logger.info(f"Test set size: {X_test.shape[0]}")

    # Определяем модели для оценки
    models_dir = Path("models")
    if args.model != "all":
        model_names = [args.model]
    else:
        model_names = [d.name for d in models_dir.iterdir() if d.is_dir()]

    all_metrics = []
    for name in model_names:
        model_path = models_dir / name
        if not model_path.exists():
            logger.warning(f"{model_path} not found, skipping.")
            continue
        if name not in MODEL_REGISTRY:
            logger.warning(f"Unknown model type {name}, skipping.")
            continue

        model_cls = MODEL_REGISTRY[name]
        try:
            model = model_cls.load(model_path)
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")
            continue

        m = evaluate_model(name, model, X_test, y_test)
        all_metrics.append(m)

        print(f"\n{'='*50}")
        print(f"Model: {name}")
        for k, v in m.items():
            print(f"{k:20s}: {v}")
        print(f"{'='*50}")

    if all_metrics:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df_out = pd.DataFrame(all_metrics)
        cols = ["model"] + [c for c in df_out if c != "model"]
        df_out = df_out[cols]
        out_file = output_dir / "evaluation_summary.csv"
        df_out.to_csv(out_file, index=False)
        logger.info(f"Summary saved to {out_file}")
    else:
        logger.warning("No models evaluated.")

if __name__ == "__main__":
    main()
