from nilm_pipeline import BaselineModels, DeepLearningModels, NILMDataLoader, NILMEvaluator


def main() -> None:
    loader = NILMDataLoader()
    loader.create_synthetic_data(days=7, freq="5min")
    loader.preprocess_data(resample_freq="5min", normalize=True)

    baseline = BaselineModels(loader)
    baseline.create_features()
    baseline.train_all_models(appliances=["fridge", "washing_machine"])

    dl_models = DeepLearningModels(loader)
    dl_models.prepare_sequences(sequence_length=24)
    dl_models.train_all_models(appliances=["fridge", "washing_machine"], sequence_length=24)

    evaluator = NILMEvaluator(baseline_models=baseline, dl_models=dl_models, loader=loader)
    evaluator.evaluate_all_models()
    comparison = evaluator.create_comprehensive_comparison()
    report = evaluator.generate_evaluation_report()

    print("Model evaluation completed")
    print(comparison.to_string(index=False))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
