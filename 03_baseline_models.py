from nilm_pipeline import BaselineModels, NILMDataLoader


def main() -> None:
    loader = NILMDataLoader()
    loader.create_synthetic_data(days=7, freq="5min")
    loader.preprocess_data(resample_freq="5min", normalize=True)

    baseline = BaselineModels(loader)
    baseline.create_features(window_size=12, lag_features=5)
    baseline.train_all_models()
    baseline.save_models()

    print("Baseline model training completed")
    print(baseline.compare_models().to_string(index=False))


if __name__ == "__main__":
    main()
