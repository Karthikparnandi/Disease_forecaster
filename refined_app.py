import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import folium_static
from folium.plugins import HeatMap
from sklearn.cluster import DBSCAN
import warnings

# Suppress deprecation warnings for a cleaner presentation
warnings.filterwarnings("ignore", category=UserWarning, module='folium')

# ======================================================================================
# Page Configuration & Title
# ======================================================================================
st.set_page_config(page_title="Disease Intelligence Dashboard", layout="wide")
st.title("🦠 Disease Forecasting & Risk Classification Dashboard")

# ======================================================================================
# Helper Functions (The Analytics Engine)
# ======================================================================================
@st.cache_data
def load_data(filepath):
    """Loads a CSV from a filepath or an uploaded file object."""
    try:
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error(f"Default data file '{filepath}' not found. Please upload a file.")
        st.stop()

@st.cache_resource
def load_model(filepath):
    """Loads the pre-trained XGBoost model."""
    try:
        model = joblib.load(filepath)
        return model
    except FileNotFoundError:
        return None

def get_risk_label_and_color(score, high_thresh, mod_thresh):
    """Assigns a text label and color based on the risk score."""
    if score >= high_thresh: return "High Risk", "red"
    if score >= mod_thresh: return "Moderate Risk", "orange"
    return "Low Risk", "green"

def find_hotspot_clusters(latest_data, risk_threshold, eps_km=25):
    """Identifies clusters of high-risk regions using DBSCAN."""
    high_risk_points = latest_data[latest_data['risk_score'] >= risk_threshold]
    if len(high_risk_points) < 2:
        return latest_data.assign(cluster=-1)

    coords = high_risk_points[['lat', 'lon']].values
    kms_per_radian = 6371.0088
    epsilon = eps_km / kms_per_radian
    
    db = DBSCAN(eps=epsilon, min_samples=2, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    high_risk_points = high_risk_points.copy()
    high_risk_points['cluster'] = db.labels_
    
    latest_data = latest_data.merge(high_risk_points[['region_id', 'cluster']], on='region_id', how='left').fillna({'cluster': -1})
    return latest_data

def generate_forecast(_model, region_df, selected_region, intervention_factor=1.0):
    """Generates a 14-day forecast, with an optional intervention factor."""
    last_known_date = region_df['date'].iloc[-1]
    future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=14, freq='D')
    future_df = pd.DataFrame({'date': future_dates, 'region_id': selected_region})

    future_df['month'] = future_df['date'].dt.month
    future_df['week_of_year'] = future_df['date'].dt.isocalendar().week.astype(int)

    region_df['rainfall_lag_7d'] = region_df['rainfall_mm'].shift(7)
    region_df['vector_index_lag_7d'] = region_df['vector_index'].shift(7)
    latest_lags = region_df.dropna().iloc[-1]

    future_df['rainfall_lag_7d'] = latest_lags['rainfall_lag_7d']
    future_df['vector_index_lag_7d'] = latest_lags['vector_index_lag_7d'] * intervention_factor

    features = ['rainfall_lag_7d', 'vector_index_lag_7d', 'month', 'week_of_year']
    predictions = _model.predict(future_df[features])
    predictions = np.maximum(0, predictions).round().astype(int)
    
    return pd.DataFrame({'date': future_dates, 'predicted_cases': predictions}).set_index('date')

# ======================================================================================
# Data Loading & Main App Logic
# ======================================================================================

# --- File Uploader ---
uploaded_file = st.file_uploader("Upload your own disease dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.success("Successfully loaded uploaded dataset.")
else:
    data = load_data('mock_data.csv')
    st.info("No file uploaded. Loading default `mock_data.csv` for demo purposes.")

model = load_model('disease_forecaster.joblib')
if model is None:
    st.error("FATAL: Forecasting model 'disease_forecaster.joblib' not found. Please ensure it's in the same folder.")
    st.stop()

# --- Data Pre-computation ---
latest_data = data.groupby('region_id').last().reset_index()
high_thresh = latest_data['daily_cases'].quantile(0.90) # Top 10% are high risk
mod_thresh = latest_data['daily_cases'].quantile(0.60) # Top 40% are moderate risk
latest_data['risk_score'] = (0.5 * latest_data['daily_cases']) + (0.3 * latest_data['vector_index'])
latest_data[['risk_label', 'risk_color']] = latest_data['risk_score'].apply(lambda x: pd.Series(get_risk_label_and_color(x, high_thresh, mod_thresh)))
latest_data_clustered = find_hotspot_clusters(latest_data, risk_threshold=mod_thresh)

# ======================================================================================
# Sidebar for Controls
# ======================================================================================
st.sidebar.title("Controls & Filters")
region_ids = sorted(data['region_id'].unique().tolist())
selected_region = st.sidebar.selectbox('Select a Region for Detailed Analysis:', region_ids)

# ======================================================================================
# Main Dashboard Layout
# ======================================================================================

# --- Main Analysis Section ---
st.markdown("---")
st.subheader(f"📈 Detailed Analysis for: **{selected_region}**")

col1, col2, col3 = st.columns(3)
selected_region_data = latest_data[latest_data['region_id'] == selected_region].iloc[0]
region_df = data[data['region_id'] == selected_region].copy()

# --- Key Metrics ---
with col1:
    st.metric("Current Risk Level", selected_region_data['risk_label'])
with col2:
    st.metric("Latest Daily Cases", f"{int(selected_region_data['daily_cases'])}")
with col3:
    st.metric("Vector Index", f"{selected_region_data['vector_index']:.2f}")

# --- Forecasting Chart ---
st.subheader("🔮 14-Day Case Forecast")
try:
    forecast_df = generate_forecast(model, region_df, selected_region)
    st.line_chart(forecast_df, color="#ffaa00", height=300)
except Exception:
    st.warning("Not enough historical data for this region to generate a forecast.")


# --- Geospatial Analysis Section ---
st.markdown("---")
st.subheader("🗺️ Geospatial Hotspot Analysis")

map_view = st.radio(
    "Select Map Visualization:",
    ('Intensity Heatmap', 'Risk Cluster View'),
    horizontal=True
)

map_center = [data['lat'].mean(), data['lon'].mean()]
m = folium.Map(location=map_center, zoom_start=10, tiles="CartoDB positron")

if map_view == 'Intensity Heatmap':
    heat_data = [[row['lat'], row['lon'], row['daily_cases']] for _, row in latest_data.iterrows()]
    HeatMap(heat_data, radius=25, blur=20).add_to(m)
else: # Risk Cluster View
    cluster_colors = ['#FF00FF', '#00FFFF', '#FFFF00', '#FF0000', '#00FF00']
    for _, row in latest_data_clustered.iterrows():
        color = cluster_colors[int(row['cluster']) % len(cluster_colors)] if row['cluster'] != -1 else row['risk_color']
        folium.Circle(
            location=[row['lat'], row['lon']],
            radius=float(row['risk_score'] * 50 + 100), # Adjusted radius for better visibility
            color=color, fill=True, fill_color=color,
            tooltip=f"<strong>Region:</strong> {row['region_id']}<br>"
                    f"<strong>Risk Label:</strong> {row['risk_label']}<br>"
                    f"<strong>Cluster ID:</strong> {'N/A' if row['cluster'] == -1 else int(row['cluster'])}"
        ).add_to(m)

folium_static(m, height=500)


# --- Intervention Simulator Section ---
st.markdown("---")
with st.expander("🔬 **Click here to run the 'What-If' Intervention Simulator**"):
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        st.subheader("Simulation Controls")
        intervention_type = st.selectbox(
            "Select Intervention Type:",
            ["None", "Vector Control (Fogging)", "Public Awareness Campaign"],
            help="Simulate the impact of an intervention on the forecast for the selected region."
        )
        
        intervention_factor = 1.0
        if intervention_type == "Vector Control (Fogging)": intervention_factor = 0.6
        elif intervention_type == "Public Awareness Campaign": intervention_factor = 0.8

    with sim_col2:
        st.subheader(f"Forecast Comparison for {selected_region}")
        base_forecast = forecast_df
        intervention_forecast = generate_forecast(model, region_df, selected_region, intervention_factor=intervention_factor)
        
        comparison_df = pd.DataFrame({
            'Baseline Forecast': base_forecast['predicted_cases'],
            'With Intervention': intervention_forecast['predicted_cases']
        })
        st.line_chart(comparison_df)
        
        base_cases, inter_cases = int(base_forecast['predicted_cases'].sum()), int(intervention_forecast['predicted_cases'].sum())
        cases_averted = base_cases - inter_cases
        
        st.metric(
            "Estimated Cases Averted Over 14 Days",
            value=cases_averted,
            delta=f"{((inter_cases - base_cases) / base_cases) * 100:.1f}% Change",
            delta_color="inverse" if cases_averted > 0 else "off"
        )
```

---

### **Step 2: How to Set Up and Run**

1.  **File Organization:**
    Make sure these four files are in the same folder in your GitHub Codespaces environment:
    * `final_hackathon_app.py` (The code above)
    * `mock_data.csv`
    * `disease_forecaster.joblib`
    * `requirements.txt`

2.  **Update `requirements.txt`:**
    Ensure your `requirements.txt` file contains all the necessary libraries:
    ```txt
    streamlit
    pandas
    scikit-learn
    numpy
    folium
    streamlit-folium
    xgboost
    ```

3.  **Install/Update Libraries:**
    Open the terminal in Codespaces and run this command to make sure everything is installed:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Final App:**
    In the terminal, execute:
    ```bash
    streamlit run final_hackathon_app.py
    
