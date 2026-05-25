from nilm_pipeline import DeepLearningModels, NILMDataLoader


def main() -> None:
    loader = NILMDataLoader()
    loader.create_synthetic_data(days=7, freq="5min")
    loader.preprocess_data(resample_freq="5min", normalize=True)

    dl_models = DeepLearningModels(loader)
    dl_models.prepare_sequences(sequence_length=24, test_size=0.2)
    dl_models.train_all_models(appliances=["fridge", "washing_machine"], sequence_length=24)
    dl_models.save_models()

    print("Sequence model training completed")
    print(dl_models.compare_models().to_string(index=False))


if __name__ == "__main__":
    main()
