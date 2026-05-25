from nilm_pipeline import NILMDataLoader


def main() -> None:
    loader = NILMDataLoader()
    aggregate_data, appliance_data = loader.create_synthetic_data(days=7, freq="5min")
    loader.preprocess_data(resample_freq="5min", normalize=True)
    X_fridge, y_fridge = loader.create_sequences(sequence_length=24, target_appliance="fridge")

    print("Dataset loading completed")
    print(f"Aggregate samples: {len(aggregate_data)}")
    print(f"Appliances: {', '.join(appliance_data.keys())}")
    print(f"Fridge sequence shape: X={X_fridge.shape}, y={y_fridge.shape}")


if __name__ == "__main__":
    main()
