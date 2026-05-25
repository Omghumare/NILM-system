from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError:
    PROJECT_ROOT = Path.cwd()
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
PLOT_DIR = ARTIFACT_DIR / "plots"
MODEL_DIR = ARTIFACT_DIR / "models"
REPORT_DIR = ARTIFACT_DIR / "reports"


APPLIANCES = ("fridge", "washing_machine", "microwave", "ac", "tv")


def ensure_artifact_dirs() -> None:
    for path in (ARTIFACT_DIR, PLOT_DIR, MODEL_DIR, REPORT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def safe_std(values: pd.Series | np.ndarray) -> float:
    std = float(np.nanstd(values))
    return std if std > 1e-12 else 1.0


def save_figure(fig: plt.Figure, name: str) -> Path:
    ensure_artifact_dirs()
    path = PLOT_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_model_key(model_key: str) -> tuple[str, str]:
    known_prefixes = ("cnn_lstm", "lstm", "gru", "xgb", "gbr", "rf", "svr")
    for prefix in known_prefixes:
        marker = f"{prefix}_"
        if model_key.startswith(marker):
            return prefix.upper().replace("_", "-"), model_key[len(marker) :]
    model, appliance = model_key.split("_", 1)
    return model.upper(), appliance


class NILMDataLoader:
    """Data loader and preprocessor for a synthetic or CSV-backed NILM dataset."""

    def __init__(self, data_path: str | os.PathLike[str] | None = None):
        self.data_path = Path(data_path) if data_path else None
        self.aggregate_data: pd.DataFrame | None = None
        self.appliance_data: dict[str, pd.DataFrame] = {}
        self.processed_data: dict[str, Any] | None = None
        self.agg_mean = 0.0
        self.agg_std = 1.0
        self.appliance_stats: dict[str, dict[str, float]] = {}

    def load_csv(self, data_path: str | os.PathLike[str] | None = None) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        path = Path(data_path or self.data_path or "")
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        df = pd.read_csv(path)
        required = {"timestamp", "aggregate"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
        df["aggregate"] = pd.to_numeric(df["aggregate"], errors="coerce")
        df = df.dropna(subset=["aggregate"])
        if df.empty:
            raise ValueError("CSV did not contain any valid aggregate readings.")

        self.aggregate_data = df[["aggregate"]]
        self.appliance_data = {}
        for appliance in APPLIANCES:
            if appliance in df.columns:
                series = pd.to_numeric(df[appliance], errors="coerce").fillna(0)
                self.appliance_data[appliance] = pd.DataFrame({"power": series}, index=df.index)

        if not self.appliance_data:
            self.appliance_data = self._estimate_appliances_from_aggregate(self.aggregate_data)

        print(f"Loaded CSV dataset with {len(self.aggregate_data)} samples from {path}")
        return self.aggregate_data, self.appliance_data

    def create_synthetic_data(self, days: int = 14, freq: str = "5min") -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
        if days < 1:
            raise ValueError("days must be at least 1")

        print("Creating synthetic NILM dataset...")
        rng = np.random.default_rng(42)
        start_date = datetime(2023, 1, 1)
        time_index = pd.date_range(start=start_date, end=start_date + timedelta(days=days), freq=freq, inclusive="left")
        n_samples = len(time_index)
        if n_samples < 100:
            raise ValueError("Generated dataset is too small. Increase days or use a finer frequency.")

        hour = time_index.hour.to_numpy()
        minute_step = max(pd.Timedelta(freq).total_seconds() / 60, 1)
        samples_per_hour = max(int(round(60 / minute_step)), 1)

        fridge = 120 + rng.normal(0, 8, n_samples)
        fridge += rng.choice([0, 160], n_samples, p=[0.82, 0.18])
        fridge = np.clip(fridge, 0, None)

        washing_machine = np.zeros(n_samples)
        cycle_count = max(1, int(days * 0.45))
        for start in rng.choice(n_samples, size=cycle_count, replace=False):
            end = min(start + 2 * samples_per_hour, n_samples)
            washing_machine[start:end] = np.clip(1200 + rng.normal(0, 150, end - start), 0, None)

        microwave = np.zeros(n_samples)
        for start in rng.choice(n_samples, size=max(1, days * 2), replace=False):
            end = min(start + max(1, int(round(3 / minute_step))), n_samples)
            microwave[start:end] = np.clip(900 + rng.normal(0, 60, end - start), 0, None)

        ac = np.clip((np.sin(2 * np.pi * (hour - 6) / 24) * 0.5 + 0.5) * 900 + rng.normal(0, 70, n_samples), 0, None)

        tv = np.zeros(n_samples)
        evening = (hour >= 18) & (hour <= 23)
        tv[evening] = np.clip(140 + rng.normal(0, 20, evening.sum()), 0, None)

        base_load = 250 + rng.normal(0, 35, n_samples)
        aggregate = np.clip(fridge + washing_machine + microwave + ac + tv + base_load, 0, None)

        self.aggregate_data = pd.DataFrame({"aggregate": aggregate}, index=time_index)
        self.appliance_data = {
            "fridge": pd.DataFrame({"power": fridge}, index=time_index),
            "washing_machine": pd.DataFrame({"power": washing_machine}, index=time_index),
            "microwave": pd.DataFrame({"power": microwave}, index=time_index),
            "ac": pd.DataFrame({"power": ac}, index=time_index),
            "tv": pd.DataFrame({"power": tv}, index=time_index),
        }
        print(f"Created synthetic dataset with {n_samples} samples over {days} days")
        return self.aggregate_data, self.appliance_data

    def preprocess_data(self, resample_freq: str = "5min", normalize: bool = True) -> dict[str, Any]:
        if self.aggregate_data is None:
            if self.data_path:
                self.load_csv(self.data_path)
            else:
                raise ValueError("No data loaded. Call create_synthetic_data() or load_csv() first.")

        aggregate = self.aggregate_data.resample(resample_freq).mean().ffill().bfill()
        appliances = {
            name: data.resample(resample_freq).mean().ffill().bfill()
            for name, data in self.appliance_data.items()
        }

        if normalize:
            self.agg_mean = float(aggregate["aggregate"].mean())
            self.agg_std = safe_std(aggregate["aggregate"])
            aggregate = aggregate.copy()
            aggregate["aggregate"] = (aggregate["aggregate"] - self.agg_mean) / self.agg_std

            normalized_appliances = {}
            self.appliance_stats = {}
            for name, data in appliances.items():
                mean_val = float(data["power"].mean())
                std_val = safe_std(data["power"])
                self.appliance_stats[name] = {"mean": mean_val, "std": std_val}
                normalized = data.copy()
                normalized["power"] = (normalized["power"] - mean_val) / std_val
                normalized_appliances[name] = normalized
            appliances = normalized_appliances

        self.processed_data = {"aggregate": aggregate, "appliances": appliances}
        print(f"Preprocessing complete. Aggregate shape: {aggregate.shape}")
        return self.processed_data

    def create_sequences(self, sequence_length: int = 24, target_appliance: str = "fridge") -> tuple[np.ndarray, np.ndarray]:
        if self.processed_data is None:
            raise ValueError("No processed data. Call preprocess_data() first.")
        if target_appliance not in self.processed_data["appliances"]:
            raise ValueError(f"Unknown appliance: {target_appliance}")

        aggregate = self.processed_data["aggregate"]["aggregate"].to_numpy()
        target = self.processed_data["appliances"][target_appliance]["power"].to_numpy()
        if len(aggregate) <= sequence_length:
            raise ValueError("Not enough samples to create sequences.")

        X = np.array([aggregate[i - sequence_length : i] for i in range(sequence_length, len(aggregate))])
        y = target[sequence_length:]
        return X, y

    def get_data_summary(self) -> dict[str, Any]:
        if self.aggregate_data is None:
            return {"status": "No data loaded"}
        return {
            "samples": len(self.aggregate_data),
            "start": str(self.aggregate_data.index.min()),
            "end": str(self.aggregate_data.index.max()),
            "appliances": list(self.appliance_data.keys()),
            "aggregate": self.aggregate_data["aggregate"].describe().to_dict(),
        }

    def _estimate_appliances_from_aggregate(self, aggregate: pd.DataFrame) -> dict[str, pd.DataFrame]:
        values = aggregate["aggregate"].to_numpy()
        idx = aggregate.index
        hour = idx.hour.to_numpy()
        estimates = {
            "fridge": np.clip(values * 0.18, 0, None),
            "washing_machine": np.where(values > np.percentile(values, 92), values * 0.25, 0),
            "microwave": np.where(values > np.percentile(values, 97), values * 0.10, 0),
            "ac": np.clip(values * (0.15 + 0.20 * ((hour >= 10) & (hour <= 18))), 0, None),
            "tv": np.where((hour >= 18) & (hour <= 23), values * 0.12, 0),
        }
        return {name: pd.DataFrame({"power": val}, index=idx) for name, val in estimates.items()}


class NILMExplorer:
    def __init__(self, loader: NILMDataLoader):
        if loader.aggregate_data is None or not loader.appliance_data:
            raise ValueError("Load data before creating NILMExplorer.")
        self.loader = loader
        self.aggregate_data = loader.aggregate_data
        self.appliance_data = loader.appliance_data

    def run_all(self) -> dict[str, Any]:
        self.plot_aggregate_consumption()
        self.plot_appliance_patterns()
        self.plot_consumption_distribution()
        self.plot_correlation_matrix()
        self.plot_daily_patterns()
        return self.generate_summary_statistics()

    def plot_aggregate_consumption(self, days_to_show: int = 7) -> Path:
        data = self.aggregate_data.head(days_to_show * 24 * 12)
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        axes[0].plot(data.index, data["aggregate"], linewidth=0.8)
        axes[0].set_title("Aggregate Power Consumption")
        axes[0].set_ylabel("Power (W)")
        hourly = data.groupby(data.index.hour)["aggregate"].mean()
        axes[1].bar(hourly.index, hourly.values, color="steelblue")
        axes[1].set_title("Average Hourly Consumption")
        axes[1].set_xlabel("Hour")
        axes[1].set_ylabel("Power (W)")
        return save_figure(fig, "aggregate_consumption.png")

    def plot_appliance_patterns(self, days_to_show: int = 3) -> Path:
        fig, axes = plt.subplots(len(self.appliance_data), 1, figsize=(12, 2.5 * len(self.appliance_data)))
        axes = np.atleast_1d(axes)
        for ax, (name, data) in zip(axes, self.appliance_data.items()):
            sample = data.head(days_to_show * 24 * 12)
            ax.plot(sample.index, sample["power"], linewidth=0.8)
            ax.set_title(name.replace("_", " ").title())
            ax.set_ylabel("Power (W)")
        axes[-1].set_xlabel("Time")
        return save_figure(fig, "appliance_patterns.png")

    def plot_consumption_distribution(self) -> Path:
        columns = 3
        rows = int(np.ceil((len(self.appliance_data) + 1) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(13, 4 * rows))
        axes = axes.flatten()
        axes[0].hist(self.aggregate_data["aggregate"], bins=40, color="black", alpha=0.75)
        axes[0].set_title("Aggregate")
        for ax, (name, data) in zip(axes[1:], self.appliance_data.items()):
            ax.hist(data["power"], bins=40, alpha=0.75)
            ax.set_title(name.replace("_", " ").title())
        for ax in axes[len(self.appliance_data) + 1 :]:
            ax.set_visible(False)
        return save_figure(fig, "consumption_distribution.png")

    def plot_correlation_matrix(self) -> Path:
        combined = self.aggregate_data.copy()
        for name, data in self.appliance_data.items():
            combined[name] = data["power"]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(combined.corr(), annot=True, cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Power Correlation Matrix")
        return save_figure(fig, "correlation_matrix.png")

    def plot_on_off_states(self, threshold_percentile: int = 20) -> dict[str, dict[str, float]]:
        stats = {}
        for name, data in self.appliance_data.items():
            threshold = np.percentile(data["power"], threshold_percentile)
            on_state = data["power"] > threshold
            stats[name] = {
                "on_time_percent": float(on_state.mean() * 100),
                "average_on_power": float(data.loc[on_state, "power"].mean()),
            }
        return stats

    def plot_daily_patterns(self) -> Path:
        fig, ax = plt.subplots(figsize=(10, 5))
        hourly = self.aggregate_data.groupby(self.aggregate_data.index.hour)["aggregate"].mean()
        ax.plot(hourly.index, hourly.values, marker="o", label="Aggregate", linewidth=2)
        for name, data in self.appliance_data.items():
            appliance_hourly = data.groupby(data.index.hour)["power"].mean()
            ax.plot(appliance_hourly.index, appliance_hourly.values, label=name.replace("_", " ").title(), alpha=0.8)
        ax.set_title("Average Daily Pattern")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Power (W)")
        ax.legend(ncol=2)
        return save_figure(fig, "daily_patterns.png")

    def generate_summary_statistics(self) -> dict[str, Any]:
        interval_hours = self._interval_hours()
        summary = {
            "dataset_info": {
                "total_samples": len(self.aggregate_data),
                "duration_days": max((self.aggregate_data.index[-1] - self.aggregate_data.index[0]).days, 1),
                "appliances": list(self.appliance_data.keys()),
            },
            "aggregate_stats": {
                "mean": float(self.aggregate_data["aggregate"].mean()),
                "std": float(self.aggregate_data["aggregate"].std()),
                "min": float(self.aggregate_data["aggregate"].min()),
                "max": float(self.aggregate_data["aggregate"].max()),
                "total_energy_kwh": float(self.aggregate_data["aggregate"].sum() * interval_hours / 1000),
            },
            "appliance_stats": {},
        }
        aggregate_sum = max(float(self.aggregate_data["aggregate"].sum()), 1e-8)
        for name, data in self.appliance_data.items():
            summary["appliance_stats"][name] = {
                "mean": float(data["power"].mean()),
                "total_energy_kwh": float(data["power"].sum() * interval_hours / 1000),
                "energy_contribution_percent": float(data["power"].sum() / aggregate_sum * 100),
            }
        return summary

    def _interval_hours(self) -> float:
        if len(self.aggregate_data.index) < 2:
            return 5 / 60
        return float((self.aggregate_data.index[1] - self.aggregate_data.index[0]).total_seconds() / 3600)


class BaselineModels:
    def __init__(self, loader: NILMDataLoader):
        self.loader = loader
        self.features: dict[str, dict[str, np.ndarray]] = {}
        self.models: dict[str, Any] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.feature_importance: dict[str, dict[str, float]] = {}
        self.results: dict[str, dict[str, Any]] = {}

    def create_features(self, window_size: int = 12, lag_features: int = 5) -> dict[str, dict[str, np.ndarray]]:
        if self.loader.processed_data is None:
            raise ValueError("No processed data available. Run preprocessing first.")
        aggregate = self.loader.processed_data["aggregate"]["aggregate"].to_numpy()
        start = max(window_size, lag_features)
        if len(aggregate) <= start + 5:
            raise ValueError("Not enough samples to create baseline features.")

        features_dict = {}
        for name, appliance_data in self.loader.processed_data["appliances"].items():
            target = appliance_data["power"].to_numpy()
            rows, y = [], []
            for i in range(start, len(aggregate)):
                window = aggregate[i - window_size : i]
                timestamp = self.loader.processed_data["aggregate"].index[i]
                row = [
                    aggregate[i],
                    float(np.mean(window)),
                    float(np.std(window)),
                    float(np.min(window)),
                    float(np.max(window)),
                    float(np.median(window)),
                    *[aggregate[i - lag] for lag in range(1, lag_features + 1)],
                    timestamp.hour,
                    timestamp.dayofweek,
                    timestamp.month,
                    np.sin(2 * np.pi * timestamp.hour / 24),
                    np.cos(2 * np.pi * timestamp.hour / 24),
                    np.sin(2 * np.pi * timestamp.dayofweek / 7),
                    np.cos(2 * np.pi * timestamp.dayofweek / 7),
                    aggregate[i] - aggregate[i - 1],
                ]
                rows.append(row)
                y.append(target[i])
            features_dict[name] = {"X": np.asarray(rows), "y": np.asarray(y)}
        self.features = features_dict
        print(f"Created baseline features for {len(features_dict)} appliances")
        return features_dict

    def train_random_forest(self, appliance_name: str, test_size: float = 0.2, n_estimators: int = 40):
        return self._train_model(
            appliance_name,
            "rf",
            RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1, min_samples_leaf=2),
            test_size,
        )

    def train_xgboost(self, appliance_name: str, test_size: float = 0.2):
        try:
            from xgboost import XGBRegressor

            model = XGBRegressor(n_estimators=60, max_depth=4, learning_rate=0.08, random_state=42, n_jobs=-1, verbosity=0)
            key = "xgb"
        except Exception:
            model = GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=42)
            key = "gbr"
        return self._train_model(appliance_name, key, model, test_size)

    def train_svr(self, appliance_name: str, test_size: float = 0.2):
        return self._train_model(appliance_name, "svr", SVR(kernel="rbf", C=1.0, gamma="scale"), test_size)

    def train_all_models(self, appliances: list[str] | None = None):
        if not self.features:
            self.create_features()
        appliances = appliances or list(self.features.keys())
        for appliance in appliances:
            self.train_random_forest(appliance)
            self.train_xgboost(appliance)
            self.train_svr(appliance)
        print(f"Trained {len(self.models)} baseline models")

    def compare_models(self) -> pd.DataFrame:
        rows = []
        for key, result in self.results.items():
            model, appliance = parse_model_key(key)
            rows.append(
                {
                    "Model": model,
                    "Appliance": appliance.replace("_", " ").title(),
                    "MAE": result["test_mae"],
                    "RMSE": result["test_rmse"],
                    "R2": result["test_r2"],
                }
            )
        return pd.DataFrame(rows)

    def plot_feature_importance(self, appliance_name: str, top_n: int = 12) -> Path | None:
        key = f"rf_{appliance_name}"
        if key not in self.feature_importance:
            return None
        items = sorted(self.feature_importance[key].items(), key=lambda item: item[1], reverse=True)[:top_n]
        labels, values = zip(*items)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels, values)
        ax.invert_yaxis()
        ax.set_title(f"Feature Importance - {appliance_name}")
        return save_figure(fig, f"feature_importance_{appliance_name}.png")

    def plot_predictions(self, appliance_name: str, days_to_show: int = 2) -> Path | None:
        keys = [key for key in self.results if key.endswith(f"_{appliance_name}")]
        if not keys:
            return None
        fig, axes = plt.subplots(len(keys), 1, figsize=(11, 3 * len(keys)))
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, keys):
            result = self.results[key]
            samples = min(days_to_show * 24 * 12, len(result["y_test"]))
            ax.plot(result["y_test"][:samples], label="Actual", color="black")
            ax.plot(result["y_pred"][:samples], label="Predicted", alpha=0.8)
            ax.set_title(f"{key} MAE={result['test_mae']:.3f}")
            ax.legend()
        return save_figure(fig, f"baseline_predictions_{appliance_name}.png")

    def save_models(self, save_path: str | os.PathLike[str] = MODEL_DIR) -> None:
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        for key, model in self.models.items():
            joblib.dump(model, path / f"{key}.joblib")
            joblib.dump(self.scalers[key], path / f"{key}_scaler.joblib")

    def _train_model(self, appliance_name: str, prefix: str, model: Any, test_size: float):
        if appliance_name not in self.features:
            raise ValueError(f"Features not available for {appliance_name}")
        X = self.features[appliance_name]["X"]
        y = self.features[appliance_name]["y"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_test = model.predict(X_test_scaled)

        key = f"{prefix}_{appliance_name}"
        self.models[key] = model
        self.scalers[key] = scaler
        if hasattr(model, "feature_importances_"):
            self.feature_importance[key] = dict(zip(self._get_feature_names(), model.feature_importances_))
        self.results[key] = self._result_dict(y_train, y_pred_train, y_test, y_pred_test)
        print(f"Trained {key}: MAE={self.results[key]['test_mae']:.4f}, R2={self.results[key]['test_r2']:.4f}")
        return model, self.results[key]

    def _result_dict(self, y_train, y_pred_train, y_test, y_pred_test) -> dict[str, Any]:
        return {
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "y_test": y_test,
            "y_pred": y_pred_test,
        }

    def _get_feature_names(self) -> list[str]:
        return [
            "current_aggregate",
            "window_mean",
            "window_std",
            "window_min",
            "window_max",
            "window_median",
            "lag_1",
            "lag_2",
            "lag_3",
            "lag_4",
            "lag_5",
            "hour",
            "dayofweek",
            "month",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "diff",
        ]


class DeepLearningModels:
    """Sequence models for NILM. Uses TensorFlow when available, with a fast sklearn fallback."""

    def __init__(self, loader: NILMDataLoader):
        self.loader = loader
        self.sequences: dict[str, dict[str, np.ndarray]] = {}
        self.models: dict[str, Any] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.history: dict[str, Any] = {}
        self.tensorflow_available = False

    def prepare_sequences(self, sequence_length: int = 24, test_size: float = 0.2) -> dict[str, dict[str, np.ndarray]]:
        if self.loader.processed_data is None:
            raise ValueError("No processed data available. Run preprocessing first.")
        sequences = {}
        for appliance in self.loader.processed_data["appliances"]:
            X, y = self.loader.create_sequences(sequence_length, appliance)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            sequences[appliance] = {
                "X_train": X_train.reshape((X_train.shape[0], X_train.shape[1], 1)),
                "X_test": X_test.reshape((X_test.shape[0], X_test.shape[1], 1)),
                "y_train": y_train,
                "y_test": y_test,
            }
        self.sequences = sequences
        print(f"Prepared sequences for {len(sequences)} appliances")
        return sequences

    def build_lstm_model(self, sequence_length: int = 24, lstm_units: int = 24, dropout_rate: float = 0.1):
        tf = self._try_tensorflow()
        if tf is None:
            return MLPRegressor(hidden_layer_sizes=(48, 24), max_iter=120, random_state=42, early_stopping=True)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(sequence_length, 1)),
                tf.keras.layers.LSTM(lstm_units, return_sequences=False),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
        return model

    def build_gru_model(self, sequence_length: int = 24, gru_units: int = 24, dropout_rate: float = 0.1):
        tf = self._try_tensorflow()
        if tf is None:
            return MLPRegressor(hidden_layer_sizes=(36, 18), max_iter=120, random_state=43, early_stopping=True)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(sequence_length, 1)),
                tf.keras.layers.GRU(gru_units, return_sequences=False),
                tf.keras.layers.Dropout(dropout_rate),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
        return model

    def build_cnn_lstm_model(self, sequence_length: int = 24, filters: int = 16, kernel_size: int = 3):
        tf = self._try_tensorflow()
        if tf is None:
            return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=120, random_state=44, early_stopping=True)
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(sequence_length, 1)),
                tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, activation="relu"),
                tf.keras.layers.MaxPooling1D(pool_size=2),
                tf.keras.layers.LSTM(16),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss="mse", metrics=["mae"])
        return model

    def train_model(self, model: Any, appliance_name: str, model_type: str, epochs: int = 3, batch_size: int = 32, **_: Any):
        if appliance_name not in self.sequences:
            raise ValueError(f"Sequences not available for {appliance_name}")
        data = self.sequences[appliance_name]
        X_train, X_test = data["X_train"], data["X_test"]
        y_train, y_test = data["y_train"], data["y_test"]

        if hasattr(model, "fit") and model.__class__.__module__.startswith("sklearn"):
            model.fit(X_train.reshape((X_train.shape[0], -1)), y_train)
            y_pred_train = model.predict(X_train.reshape((X_train.shape[0], -1)))
            y_pred_test = model.predict(X_test.reshape((X_test.shape[0], -1)))
            epochs_trained = getattr(model, "n_iter_", 1)
        else:
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=0)
            self.history[f"{model_type}_{appliance_name}"] = history
            y_pred_train = model.predict(X_train, verbose=0).ravel()
            y_pred_test = model.predict(X_test, verbose=0).ravel()
            epochs_trained = len(history.history.get("loss", []))

        key = f"{model_type}_{appliance_name}"
        self.models[key] = model
        self.results[key] = {
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_r2": r2_score(y_test, y_pred_test),
            "y_test": y_test,
            "y_pred": y_pred_test,
            "epochs_trained": epochs_trained,
        }
        print(f"Trained {key}: MAE={self.results[key]['test_mae']:.4f}, R2={self.results[key]['test_r2']:.4f}")
        return model, self.results[key]

    def train_all_models(self, appliances: list[str] | None = None, sequence_length: int = 24):
        if not self.sequences:
            self.prepare_sequences(sequence_length=sequence_length)
        appliances = appliances or list(self.sequences.keys())
        for appliance in appliances:
            self.train_model(self.build_lstm_model(sequence_length), appliance, "lstm")
            self.train_model(self.build_gru_model(sequence_length), appliance, "gru")
            self.train_model(self.build_cnn_lstm_model(sequence_length), appliance, "cnn_lstm")
        print(f"Trained {len(self.models)} sequence models")

    def compare_models(self) -> pd.DataFrame:
        rows = []
        for key, result in self.results.items():
            model, appliance = parse_model_key(key)
            rows.append(
                {
                    "Model": model,
                    "Appliance": appliance.replace("_", " ").title(),
                    "MAE": result["test_mae"],
                    "RMSE": result["test_rmse"],
                    "R2": result["test_r2"],
                    "Epochs": result["epochs_trained"],
                }
            )
        return pd.DataFrame(rows)

    def plot_training_history(self, appliance_name: str) -> Path | None:
        histories = {key: value for key, value in self.history.items() if key.endswith(f"_{appliance_name}")}
        if not histories:
            return None
        fig, ax = plt.subplots(figsize=(8, 4))
        for key, history in histories.items():
            ax.plot(history.history.get("loss", []), label=key)
        ax.set_title(f"Training History - {appliance_name}")
        ax.legend()
        return save_figure(fig, f"dl_history_{appliance_name}.png")

    def plot_predictions(self, appliance_name: str, days_to_show: int = 2) -> Path | None:
        keys = [key for key in self.results if key.endswith(f"_{appliance_name}")]
        if not keys:
            return None
        fig, axes = plt.subplots(len(keys), 1, figsize=(11, 3 * len(keys)))
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, keys):
            result = self.results[key]
            samples = min(days_to_show * 24 * 12, len(result["y_test"]))
            ax.plot(result["y_test"][:samples], label="Actual", color="black")
            ax.plot(result["y_pred"][:samples], label="Predicted")
            ax.set_title(f"{key} MAE={result['test_mae']:.3f}")
            ax.legend()
        return save_figure(fig, f"dl_predictions_{appliance_name}.png")

    def save_models(self, save_path: str | os.PathLike[str] = MODEL_DIR) -> None:
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        for key, model in self.models.items():
            if hasattr(model, "save") and not model.__class__.__module__.startswith("sklearn"):
                model.save(path / f"{key}.keras")
            else:
                joblib.dump(model, path / f"{key}.joblib")

    def _try_tensorflow(self):
        try:
            import tensorflow as tf

            tf.random.set_seed(42)
            self.tensorflow_available = True
            return tf
        except Exception:
            self.tensorflow_available = False
            return None


class NILMEvaluator:
    def __init__(self, baseline_models: BaselineModels | None = None, dl_models: DeepLearningModels | None = None, loader: NILMDataLoader | None = None):
        self.baseline_models = baseline_models
        self.dl_models = dl_models
        self.loader = loader
        self.evaluation_results: dict[str, dict[str, Any]] = {}

    def calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        nonzero = np.abs(y_true) > 1e-6
        mape = np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100 if np.any(nonzero) else 0.0
        return {
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": float(r2_score(y_true, y_pred)),
            "MAPE": float(mape),
            "Max_Error": float(np.max(np.abs(y_true - y_pred))),
            "Mean_Error": float(np.mean(y_true - y_pred)),
            "Std_Error": float(np.std(y_true - y_pred)),
        }

    def calculate_on_off_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, threshold_percentile: int = 70):
        threshold = np.percentile(y_true, threshold_percentile)
        y_true_binary = (y_true > threshold).astype(int)
        y_pred_binary = (y_pred > threshold).astype(int)
        metrics = {
            "F1_Score": float(f1_score(y_true_binary, y_pred_binary, zero_division=0)),
            "Precision": float(precision_score(y_true_binary, y_pred_binary, zero_division=0)),
            "Recall": float(recall_score(y_true_binary, y_pred_binary, zero_division=0)),
            "ON_Time_Accuracy": float(np.mean(y_true_binary == y_pred_binary)),
        }
        return metrics, y_true_binary, y_pred_binary

    def calculate_energy_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, time_interval_hours: float = 5 / 60) -> dict[str, float]:
        true_energy = float(np.sum(y_true) * time_interval_hours / 1000)
        pred_energy = float(np.sum(y_pred) * time_interval_hours / 1000)
        return {
            "True_Energy_kWh": true_energy,
            "Pred_Energy_kWh": pred_energy,
            "Energy_Error_kWh": abs(true_energy - pred_energy),
            "Energy_Error_Percent": abs(true_energy - pred_energy) / (abs(true_energy) + 1e-8) * 100,
            "Energy_Bias": (pred_energy - true_energy) / (abs(true_energy) + 1e-8) * 100,
        }

    def evaluate_all_models(self) -> dict[str, dict[str, Any]]:
        all_results = {}
        sources = []
        if self.baseline_models is not None:
            sources.append(self.baseline_models.results)
        if self.dl_models is not None:
            sources.append(self.dl_models.results)

        for results_by_model in sources:
            for key, result in results_by_model.items():
                _, appliance = parse_model_key(key)
                y_true = np.asarray(result["y_test"], dtype=float)
                y_pred = np.asarray(result["y_pred"], dtype=float)
                y_true_eval, y_pred_eval = self._denormalize_appliance_values(appliance, y_true, y_pred)
                reg = self.calculate_regression_metrics(y_true_eval, y_pred_eval)
                on_off, y_true_binary, y_pred_binary = self.calculate_on_off_metrics(y_true_eval, y_pred_eval)
                energy = self.calculate_energy_metrics(y_true_eval, y_pred_eval)
                all_results[key] = {
                    "metrics": {**reg, **on_off, **energy},
                    "predictions": {"y_true": y_true_eval, "y_pred": y_pred_eval},
                    "binary_states": {"y_true_binary": y_true_binary, "y_pred_binary": y_pred_binary},
                }
        self.evaluation_results = all_results
        print(f"Evaluated {len(all_results)} models")
        return all_results

    def _denormalize_appliance_values(self, appliance: str, y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.loader is None or appliance not in getattr(self.loader, "appliance_stats", {}):
            return y_true, y_pred
        stats = self.loader.appliance_stats[appliance]
        mean = stats["mean"]
        std = stats["std"]
        true_power = np.maximum(y_true * std + mean, 0)
        pred_power = np.maximum(y_pred * std + mean, 0)
        return true_power, pred_power

    def create_comprehensive_comparison(self) -> pd.DataFrame:
        if not self.evaluation_results:
            self.evaluate_all_models()
        rows = []
        for key, result in self.evaluation_results.items():
            model, appliance = parse_model_key(key)
            metrics = result["metrics"]
            rows.append(
                {
                    "Model": model,
                    "Appliance": appliance.replace("_", " ").title(),
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                    "R2": metrics["R2"],
                    "MAPE": metrics["MAPE"],
                    "F1_Score": metrics["F1_Score"],
                    "Precision": metrics["Precision"],
                    "Recall": metrics["Recall"],
                    "Energy_Error_Percent": metrics["Energy_Error_Percent"],
                    "Energy_Bias": metrics["Energy_Bias"],
                }
            )
        return pd.DataFrame(rows).sort_values(["Appliance", "MAE"]).reset_index(drop=True)

    def plot_prediction_samples(self, appliance_name: str, days_to_show: int = 2, models_to_compare: list[str] | None = None) -> Path | None:
        if not self.evaluation_results:
            self.evaluate_all_models()
        keys = models_to_compare or [key for key in self.evaluation_results if key.endswith(f"_{appliance_name}")]
        if not keys:
            return None
        fig, axes = plt.subplots(len(keys), 1, figsize=(11, 3 * len(keys)))
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, keys):
            result = self.evaluation_results[key]
            y_true = result["predictions"]["y_true"]
            y_pred = result["predictions"]["y_pred"]
            samples = min(days_to_show * 24 * 12, len(y_true))
            ax.plot(y_true[:samples], label="Actual", color="black")
            ax.plot(y_pred[:samples], label="Predicted")
            ax.set_title(key)
            ax.legend()
        return save_figure(fig, f"evaluation_predictions_{appliance_name}.png")

    def plot_confusion_matrices(self, appliance_name: str) -> Path | None:
        if not self.evaluation_results:
            self.evaluate_all_models()
        keys = [key for key in self.evaluation_results if key.endswith(f"_{appliance_name}")]
        if not keys:
            return None
        cols = min(3, len(keys))
        rows = int(np.ceil(len(keys) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
        axes = np.atleast_1d(axes).flatten()
        for ax, key in zip(axes, keys):
            states = self.evaluation_results[key]["binary_states"]
            cm = confusion_matrix(states["y_true_binary"], states["y_pred_binary"])
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
            ax.set_title(key)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
        for ax in axes[len(keys) :]:
            ax.set_visible(False)
        return save_figure(fig, f"confusion_matrices_{appliance_name}.png")

    def generate_evaluation_report(self, save_path: str | os.PathLike[str] = REPORT_DIR / "nilm_evaluation_report.txt") -> Path:
        ensure_artifact_dirs()
        path = Path(save_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        comparison = self.create_comprehensive_comparison()
        with path.open("w", encoding="utf-8") as handle:
            handle.write("AI-Powered Appliance Energy Disaggregation (NILM)\n")
            handle.write("Comprehensive Model Evaluation Report\n")
            handle.write("=" * 72 + "\n\n")
            handle.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            handle.write(f"Models evaluated: {len(self.evaluation_results)}\n\n")
            handle.write(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
            handle.write("\n")
        comparison.to_csv(path.with_suffix(".csv"), index=False)
        print(f"Evaluation report saved to {path}")
        return path


@dataclass
class PipelineResult:
    loader: NILMDataLoader
    baseline: BaselineModels
    deep_learning: DeepLearningModels
    evaluator: NILMEvaluator
    comparison: pd.DataFrame
    report_path: Path


def run_pipeline(days: int = 7, freq: str = "5min", selected_appliances: list[str] | None = None) -> PipelineResult:
    ensure_artifact_dirs()
    selected_appliances = selected_appliances or ["fridge", "washing_machine"]

    print("Step 1: dataset loading and preprocessing")
    loader = NILMDataLoader()
    loader.create_synthetic_data(days=days, freq=freq)
    loader.preprocess_data(resample_freq=freq, normalize=True)

    print("Step 2: exploratory analysis")
    explorer = NILMExplorer(loader)
    summary = explorer.run_all()
    (REPORT_DIR / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Step 3: baseline models")
    baseline = BaselineModels(loader)
    baseline.create_features(window_size=12, lag_features=5)
    baseline.train_all_models(appliances=list(loader.appliance_data.keys()))
    baseline.save_models()

    print("Step 4: sequence models")
    deep_learning = DeepLearningModels(loader)
    deep_learning.prepare_sequences(sequence_length=24, test_size=0.2)
    deep_learning.train_all_models(appliances=selected_appliances, sequence_length=24)
    deep_learning.save_models()

    print("Step 5: model evaluation")
    evaluator = NILMEvaluator(baseline_models=baseline, dl_models=deep_learning, loader=loader)
    evaluator.evaluate_all_models()
    comparison = evaluator.create_comprehensive_comparison()
    report_path = evaluator.generate_evaluation_report(REPORT_DIR / "nilm_comprehensive_report.txt")

    print("Pipeline completed successfully")
    return PipelineResult(loader, baseline, deep_learning, evaluator, comparison, report_path)
