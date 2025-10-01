import streamlit as st
import pandas as pd
import joblib
import numpy as np
import folium
from streamlit_folium import folium_static

# ======================================================================================
# Page Configuration & Title
# ======================================================================================
# Set the page layout to wide for a modern, spacious feel. This must be the first Streamlit command.
st.set_page_config(layout="wide")

# Use a clear and compelling title. Emojis can add a nice touch.
st.title('Aegis 🛡️: Proactive Disease Intelligence')

# ======================================================================================
# Data Loading and Caching
# ======================================================================================
# Use a function with st.cache_data to load data efficiently.
# This prevents reloading the data every time a user interacts with the app, making it faster.
@st.cache_data
def load_data(filepath):
    """
    Loads and preprocesses the main dataset from a CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        st.error(f"Data file '{filepath}' not found. Please ensure it's in the correct directory.")
        st.stop() # Stop the app if data isn't available

# Load your main dataset
df_data = load_data('mock_data.csv')

@st.cache_resource
def load_model(filepath):
    """
    Loads the trained machine learning model.
    Using st.cache_resource is ideal for ML models.
    """
    try:
        model = joblib.load(filepath)
        return model
    except FileNotFoundError:
        return None # Return None if the model isn't found

# Load your trained model
model = load_model('disease_forecaster.joblib')

# ======================================================================================
# Sidebar Controls
# ======================================================================================
# Group all user controls into the sidebar for a clean interface.
st.sidebar.header('Dashboard Controls ⚙️')
region_ids = df_data['region_id'].unique().tolist()
selected_region = st.sidebar.selectbox('Select a Region:', region_ids, help="Choose a region to view its specific forecast and metrics.")

# ======================================================================================
# Main Dashboard Layout
# ======================================================================================
# Create a two-column layout for the map and the analysis.
col1, col2 = st.columns((6, 4)) # Give the map column more space

# --- Column 1: Map and Actionable Insights ---
with col1:
    st.subheader('Live Disease Hotspot Map 🗺️')

    # --- Hotspot Map Generation ---
    # Centering the map on the average coordinates of your data for a good default view.
    map_center = [df_data['lat'].mean(), df_data['lon'].mean()]
    hotspot_map = folium.Map(location=map_center, zoom_start=10, tiles="CartoDB positron")

    # Get the latest data point for each region to display on the map
    latest_data = df_data.groupby('region_id').last().reset_index()

    # Define risk thresholds for color-coding
    high_risk_threshold = 15.0
    moderate_risk_threshold = 10.0

    # Simplified risk score calculation (can be replaced with model output later)
    latest_data['risk_score'] = (0.5 * latest_data['daily_cases']) + (0.3 * latest_data['vector_index'])

    # Function to determine color based on risk
    def get_color(risk_score):
        if risk_score >= high_risk_threshold:
            return 'red'
        elif risk_score >= moderate_risk_threshold:
            return 'orange'
        else:
            return 'green'

    # Add circles to the map for each region
    for _, row in latest_data.iterrows():
        folium.Circle(
            location=[row['lat'], row['lon']],
            radius=float(row['risk_score'] * 100), # Radius based on risk
            color=get_color(row['risk_score']),
            fill=True,
            fill_color=get_color(row['risk_score']),
            tooltip=f"<strong>Region:</strong> {row['region_id']}<br>"
                    f"<strong>Risk Score:</strong> {row['risk_score']:.2f}<br>"
                    f"<strong>Daily Cases:</strong> {row['daily_cases']}"
        ).add_to(hotspot_map)

    # Display the map in Streamlit
    folium_static(hotspot_map, width=700, height=500)

    # --- Actionable Insight Section ---
    st.subheader('Actionable Insight & Recommendations 💡')
    selected_region_data = latest_data[latest_data['region_id'] == selected_region]
    if not selected_region_data.empty:
        risk_score = selected_region_data['risk_score'].iloc[0]
        if risk_score >= high_risk_threshold:
            st.error(f"**Region {selected_region} is at HIGH risk (Score: {risk_score:.2f}).** \n**Recommendations:** Implement enhanced surveillance, mobilize public health teams, and issue public advisories immediately.")
        elif risk_score >= moderate_risk_threshold:
            st.warning(f"**Region {selected_region} is at MODERATE risk (Score: {risk_score:.2f}).** \n**Recommendations:** Increase awareness campaigns, prepare healthcare facilities, and monitor key indicators closely.")
        else:
            st.success(f"**Region {selected_region} is at LOW risk (Score: {risk_score:.2f}).** \n**Recommendations:** Continue routine surveillance and community engagement.")
    else:
        st.info("Select a region to see insights.")


# --- Column 2: Regional Analysis & Forecast ---
with col2:
    st.subheader(f'Analysis for {selected_region} 📊')

    if model is None:
        st.error("Forecasting model not found. Please ensure 'disease_forecaster.joblib' is available.")
    else:
        # Filter data for the selected region
        region_df = df_data[df_data['region_id'] == selected_region].copy()

        if not region_df.empty:
            # --- Key Metrics Display ---
            latest_region_data = region_df.iloc[-1]
            total_predicted_cases = "N/A" # Placeholder for forecast result
            delta_label = "N/A"

            # Display key metrics using st.metric
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(
                    label="Predicted Cases (Next 14 Days)",
                    value=total_predicted_cases
                )
            with metric_col2:
                 st.metric(
                    label="Current Vector Index",
                    value=f"{latest_region_data['vector_index']:.2f}"
                )

            # --- 14-Day Case Forecast ---
            st.subheader('14-Day Case Forecast')
            try:
                # --- Feature Engineering for Prediction ---
                last_known_date = region_df['date'].iloc[-1]
                future_dates = pd.date_range(start=last_known_date + pd.Timedelta(days=1), periods=14, freq='D')
                future_df = pd.DataFrame({'date': future_dates, 'region_id': selected_region})

                # Create time-based features
                future_df['month'] = future_df['date'].dt.month
                future_df['week_of_year'] = future_df['date'].dt.isocalendar().week.astype(int)

                # Use the last known lagged values for forecasting
                # Note: A more advanced model might try to forecast these values too.
                region_df['rainfall_lag_7d'] = region_df['rainfall_mm'].shift(7)
                region_df['vector_index_lag_7d'] = region_df['vector_index'].shift(7)
                latest_lags = region_df.dropna().iloc[-1]

                future_df['rainfall_lag_7d'] = latest_lags['rainfall_lag_7d']
                future_df['vector_index_lag_7d'] = latest_lags['vector_index_lag_7d']

                # --- Prediction ---
                features = ['rainfall_lag_7d', 'vector_index_lag_7d', 'month', 'week_of_year']
                X_future = future_df[features]
                predictions = model.predict(X_future)
                predictions = np.maximum(0, predictions).round().astype(int) # Ensure non-negative cases

                forecast_df = pd.DataFrame({
                    'date': future_dates,
                    'predicted_cases': predictions
                }).set_index('date')

                # Display the forecast chart
                st.line_chart(forecast_df, color="#ffaa00") # Use a distinct color

                # Update the metric placeholder with the actual prediction sum
                total_predicted_cases = forecast_df['predicted_cases'].sum()
                metric_col1.metric(
                    label="Predicted Cases (Next 14 Days)",
                    value=int(total_predicted_cases)
                )

            except Exception as e:
                st.warning(f"Could not generate a forecast for {selected_region}. Error: {e}")

        else:
            st.warning(f"No data available for region {selected_region}.")