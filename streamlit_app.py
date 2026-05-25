import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from nilm_pipeline import BaselineModels, DeepLearningModels, NILMDataLoader, NILMEvaluator


st.set_page_config(page_title="NILM Energy Dashboard", layout="wide")


def get_state(name: str, default=None):
    if name not in st.session_state:
        st.session_state[name] = default
    return st.session_state[name]


def generate_data(days: int, freq: str) -> None:
    loader = NILMDataLoader()
    loader.create_synthetic_data(days=days, freq=freq)
    loader.preprocess_data(resample_freq=freq, normalize=True)
    st.session_state.loader = loader
    st.session_state.baseline = None
    st.session_state.dl_models = None
    st.session_state.evaluator = None


def load_uploaded_data(uploaded_file, freq: str) -> None:
    loader = NILMDataLoader()
    df = pd.read_csv(uploaded_file)
    required = {"timestamp", "aggregate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Uploaded CSV is missing required columns: {sorted(missing)}")
    loader.aggregate_data = df.assign(timestamp=pd.to_datetime(df["timestamp"])).set_index("timestamp")[["aggregate"]]
    loader.appliance_data = loader._estimate_appliances_from_aggregate(loader.aggregate_data)
    loader.preprocess_data(resample_freq=freq, normalize=True)
    st.session_state.loader = loader
    st.session_state.baseline = None
    st.session_state.dl_models = None
    st.session_state.evaluator = None


def train_models(model_choices: list[str]) -> None:
    loader = st.session_state.loader
    if loader is None:
        st.warning("Load or generate data first.")
        return

    if any(choice in model_choices for choice in ["Random Forest", "XGBoost/GBR", "SVR"]):
        baseline = BaselineModels(loader)
        baseline.create_features()
        appliances = ["fridge", "washing_machine"]
        for appliance in appliances:
            if "Random Forest" in model_choices:
                baseline.train_random_forest(appliance)
            if "XGBoost/GBR" in model_choices:
                baseline.train_xgboost(appliance)
            if "SVR" in model_choices:
                baseline.train_svr(appliance)
        st.session_state.baseline = baseline

    if any(choice in model_choices for choice in ["LSTM", "GRU", "CNN-LSTM"]):
        dl_models = DeepLearningModels(loader)
        dl_models.prepare_sequences(sequence_length=24)
        appliances = ["fridge", "washing_machine"]
        for appliance in appliances:
            if "LSTM" in model_choices:
                dl_models.train_model(dl_models.build_lstm_model(sequence_length=24), appliance, "lstm")
            if "GRU" in model_choices:
                dl_models.train_model(dl_models.build_gru_model(sequence_length=24), appliance, "gru")
            if "CNN-LSTM" in model_choices:
                dl_models.train_model(dl_models.build_cnn_lstm_model(sequence_length=24), appliance, "cnn_lstm")
        st.session_state.dl_models = dl_models

    evaluator = NILMEvaluator(st.session_state.baseline, st.session_state.dl_models, loader)
    evaluator.evaluate_all_models()
    st.session_state.evaluator = evaluator


def render_data_page(loader: NILMDataLoader) -> None:
    aggregate = loader.aggregate_data
    st.subheader("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples", f"{len(aggregate):,}")
    col2.metric("Appliances", len(loader.appliance_data))
    col3.metric("Average W", f"{aggregate['aggregate'].mean():.0f}")
    col4.metric("Peak W", f"{aggregate['aggregate'].max():.0f}")

    fig = go.Figure()
    sample = aggregate.tail(min(len(aggregate), 7 * 24 * 12))
    fig.add_trace(go.Scatter(x=sample.index, y=sample["aggregate"], mode="lines", name="Aggregate"))
    fig.update_layout(title="Aggregate Consumption", xaxis_title="Time", yaxis_title="Watts", height=420)
    st.plotly_chart(fig, use_container_width=True)

    appliance_means = {name: data["power"].mean() for name, data in loader.appliance_data.items()}
    st.bar_chart(pd.Series(appliance_means, name="Average Power W"))


def render_results_page() -> None:
    baseline = get_state("baseline")
    dl_models = get_state("dl_models")
    evaluator = get_state("evaluator")

    rows = []
    if baseline is not None:
        rows.append(baseline.compare_models())
    if dl_models is not None:
        rows.append(dl_models.compare_models())
    if rows:
        st.subheader("Training Results")
        st.dataframe(pd.concat(rows, ignore_index=True), use_container_width=True)

    if evaluator is not None:
        st.subheader("Evaluation Results")
        comparison = evaluator.create_comprehensive_comparison()
        st.dataframe(comparison, use_container_width=True)

        appliance = st.selectbox("Prediction plot appliance", ["fridge", "washing_machine"])
        matching = [key for key in evaluator.evaluation_results if key.endswith(f"_{appliance}")]
        if matching:
            fig = go.Figure()
            first = evaluator.evaluation_results[matching[0]]["predictions"]
            samples = min(288, len(first["y_true"]))
            fig.add_trace(go.Scatter(y=first["y_true"][:samples], mode="lines", name="Actual"))
            fig.add_trace(go.Scatter(y=first["y_pred"][:samples], mode="lines", name=matching[0]))
            fig.update_layout(title=f"Prediction Sample: {matching[0]}", xaxis_title="Time step", yaxis_title="Normalized power")
            st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    get_state("loader")
    get_state("baseline")
    get_state("dl_models")
    get_state("evaluator")

    st.title("AI-Powered Appliance Energy Disaggregation")
    st.caption("Synthetic NILM pipeline with baseline ML, sequence models, and evaluation.")

    with st.sidebar:
        st.header("Data")
        freq = st.selectbox("Sampling frequency", ["5min", "15min", "1min"], index=0)
        days = st.slider("Synthetic days", 2, 30, 7)
        if st.button("Generate synthetic data"):
            with st.spinner("Generating data..."):
                generate_data(days, freq)
            st.success("Data generated.")

        uploaded_file = st.file_uploader("Or upload CSV", type=["csv"])
        if uploaded_file is not None and st.button("Load uploaded CSV"):
            try:
                load_uploaded_data(uploaded_file, freq)
                st.success("CSV loaded.")
            except Exception as exc:
                st.error(str(exc))

        st.header("Models")
        choices = st.multiselect(
            "Models to train",
            ["Random Forest", "XGBoost/GBR", "SVR", "LSTM", "GRU", "CNN-LSTM"],
            default=["Random Forest", "LSTM"],
        )
        if st.button("Train selected models"):
            with st.spinner("Training models..."):
                train_models(choices)
            st.success("Training complete.")

    loader = st.session_state.loader
    tab_data, tab_models = st.tabs(["Data Analysis", "Models and Evaluation"])
    with tab_data:
        if loader is None:
            st.info("Generate synthetic data or upload a CSV from the sidebar.")
        else:
            render_data_page(loader)
    with tab_models:
        if loader is None:
            st.info("Load data before training models.")
        else:
            render_results_page()


if __name__ == "__main__":
    main()
