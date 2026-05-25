from nilm_pipeline import NILMDataLoader, NILMExplorer, REPORT_DIR


def main() -> None:
    loader = NILMDataLoader()
    loader.create_synthetic_data(days=7, freq="5min")
    loader.preprocess_data(resample_freq="5min", normalize=False)

    explorer = NILMExplorer(loader)
    summary = explorer.run_all()

    print("Exploratory analysis completed")
    print(f"Samples: {summary['dataset_info']['total_samples']}")
    print(f"Plots saved under: {REPORT_DIR.parent / 'plots'}")


if __name__ == "__main__":
    main()
