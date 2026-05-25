from nilm_pipeline import run_pipeline


def main() -> None:
    result = run_pipeline(days=7, freq="5min", selected_appliances=["fridge", "washing_machine"])
    print("Complete pipeline finished")
    print(f"Models evaluated: {len(result.evaluator.evaluation_results)}")
    print(f"Report: {result.report_path}")


if __name__ == "__main__":
    main()
