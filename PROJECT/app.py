
import streamlit as st
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Custom Optimiser Performance Platform",
    page_icon="📊",
    layout="wide"
)

st.title("Custom Optimiser Development and Performance Comparison Platform")

st.write(
    "An interactive platform for comparing five optimisation "
    "algorithms using a CNN trained on the Fashion-MNIST dataset."
)

st.divider()

# --------------------------------------------------
# LOAD EXPERIMENT DATA
# --------------------------------------------------

@st.cache_data
def load_data():
    results = pd.read_csv("final_results.csv")
    history = pd.read_csv("training_history.csv")
    return results, history

try:
    results_df, history_df = load_data()
except FileNotFoundError:
    st.error(
        "CSV files not found. Place final_results.csv and "
        "training_history.csv in the same folder as app.py."
    )
    st.stop()

# --------------------------------------------------
# OVERVIEW METRICS
# --------------------------------------------------

best_row = results_df.loc[
    results_df["Test Accuracy (%)"].idxmax()
]

fastest_row = results_df.loc[
    results_df["Training Time (seconds)"].idxmin()
]

col1, col2, col3 = st.columns(3)

col1.metric(
    "Number of Optimisers",
    len(results_df)
)

col2.metric(
    "Best Test Accuracy",
    f'{best_row["Test Accuracy (%)"]:.2f}%'
)

col3.metric(
    "Fastest Optimiser",
    fastest_row["Optimizer"]
)

st.caption(
    f'Best test accuracy achieved by: {best_row["Optimizer"]}'
)

st.divider()

# --------------------------------------------------
# OPTIMISER COMPARISON TABLE
# --------------------------------------------------

st.header("1. Overall Performance Comparison")

st.dataframe(
    results_df.round(4),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# TRAINING TIME CHART
# --------------------------------------------------

st.header("2. Training Time Comparison")

time_fig = px.bar(
    results_df,
    x="Optimizer",
    y="Training Time (seconds)",
    color="Optimizer",
    title="Training Time of Different Optimisers"
)

time_fig.update_layout(showlegend=False)

st.plotly_chart(time_fig, use_container_width=True)

# --------------------------------------------------
# TEST ACCURACY CHART
# --------------------------------------------------

st.header("3. Test Accuracy Comparison")

accuracy_fig = px.bar(
    results_df,
    x="Optimizer",
    y="Test Accuracy (%)",
    color="Optimizer",
    title="Test Accuracy of Different Optimisers"
)

accuracy_fig.update_layout(showlegend=False)

st.plotly_chart(accuracy_fig, use_container_width=True)

# --------------------------------------------------
# INDIVIDUAL OPTIMISER ANALYSIS
# --------------------------------------------------

st.header("4. Individual Optimiser Analysis")

selected_optimizer = st.selectbox(
    "Select an optimiser",
    results_df["Optimizer"].tolist()
)

selected_history = history_df[
    history_df["Optimizer"] == selected_optimizer
]

selected_result = results_df[
    results_df["Optimizer"] == selected_optimizer
].iloc[0]

metric1, metric2, metric3 = st.columns(3)

metric1.metric(
    "Test Accuracy",
    f'{selected_result["Test Accuracy (%)"]:.2f}%'
)

metric2.metric(
    "Test Loss",
    f'{selected_result["Test Loss"]:.4f}'
)

metric3.metric(
    "Training Time",
    f'{selected_result["Training Time (seconds)"]:.2f} sec'
)

# Training loss curve
loss_fig = px.line(
    selected_history,
    x="Epoch",
    y=["Train Loss", "Validation Loss"],
    markers=True,
    title=f"{selected_optimizer} - Loss Comparison"
)

st.plotly_chart(loss_fig, use_container_width=True)

# Accuracy curve
accuracy_history_fig = px.line(
    selected_history,
    x="Epoch",
    y=["Train Accuracy (%)", "Validation Accuracy (%)"],
    markers=True,
    title=f"{selected_optimizer} - Accuracy Comparison"
)

st.plotly_chart(
    accuracy_history_fig,
    use_container_width=True
)

# --------------------------------------------------
# TRAINING HISTORY TABLE
# --------------------------------------------------

st.header("5. Training History")

st.dataframe(
    history_df.round(4),
    use_container_width=True,
    hide_index=True
)

# --------------------------------------------------
# DOWNLOAD RESULTS
# --------------------------------------------------

st.header("6. Download Experiment Results")

col1, col2 = st.columns(2)

with col1:
    st.download_button(
        label="Download Final Results CSV",
        data=results_df.to_csv(index=False),
        file_name="final_results.csv",
        mime="text/csv"
    )

with col2:
    st.download_button(
        label="Download Training History CSV",
        data=history_df.to_csv(index=False),
        file_name="training_history.csv",
        mime="text/csv"
    )

st.divider()

st.caption(
    "Project: Custom Optimiser Development and Performance Comparison Platform | "
    "Dataset: Fashion-MNIST | Model: Convolutional Neural Network"
)
