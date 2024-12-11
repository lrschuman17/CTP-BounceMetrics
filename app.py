import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

# Custom CSS for vibrant NBA styling
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to bottom, #0033a0, #ed174c); /* NBA team colors gradient */
        font-family: 'Trebuchet MS', sans-serif;
        margin: 0;
        padding: 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(to bottom, #ed174c, #ffcc00); /* Red to yellow gradient */
        border-radius: 10px;
        padding: 10px;
        color: #ffffff;
    }
    .stButton > button {
        background-color: #ffcc00; /* Bold yellow */
        color: #0033a0;
        border: none;
        border-radius: 5px;
        padding: 10px 15px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #ffc107; /* Brighter yellow */
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    .stMarkdown h1 {
        color: #ffffff;
        font-size: 3em;
        font-weight: bold;
        text-shadow: 2px 2px #000000;
    }
    .stMarkdown h2, .stMarkdown h3 {
        color: #f4f4f4;
    }
    .block-container {
        border-radius: 10px;
        padding: 20px;
        background-color: rgba(0, 0, 0, 0.8); /* Dark semi-transparent background */
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.3);
    }
    .dataframe {
        background-color: rgba(255, 255, 255, 0.1); /* Transparent table background */
        color: #ffffff;
        border-radius: 10px;
    }
    .stPlotlyChart {
        background-color: rgba(0, 0, 0, 0.8); /* Match dark theme */
        padding: 10px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.5);
    }
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        margin: 25px 0;
        font-size: 18px;
        text-align: left;
        border-radius: 5px 5px 0 0;
        overflow: hidden;
    }
    .styled-table th, .styled-table td {
        padding: 12px 15px;
    }
    .styled-table thead tr {
        background-color: #009879;
        color: #ffffff;
        text-align: center;
        font-weight: bold;
    }
    .styled-table tbody tr {
        border-bottom: 1px solid #dddddd;
    }
    .styled-table tbody tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    .styled-table tbody tr:last-of-type {
        border-bottom: 2px solid #009879;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Mapping for position to numeric values
position_mapping = {
    "PG": 1.0,  # Point Guard
    "SG": 2.0,  # Shooting Guard
    "SF": 3.0,  # Small Forward
    "PF": 4.0,  # Power Forward
    "C": 5.0,   # Center
}

# Predefined injury types
injury_types = [
    "foot fracture injury",
    "hip flexor surgery injury",
    "calf strain injury",
    "quad injury injury",
    "shoulder sprain injury",
    "foot sprain injury",
    "torn rotator cuff injury injury",
    "torn mcl injury",
    "hip flexor strain injury",
    "fractured leg injury",
    "sprained mcl injury",
    "ankle sprain injury",
    "hamstring injury injury",
    "meniscus tear injury",
    "torn hamstring injury",
    "dislocated shoulder injury",
    "ankle fracture injury",
    "fractured hand injury",
    "bone spurs injury",
    "acl tear injury",
    "hip labrum injury",
    "back surgery injury",
    "arm injury injury",
    "torn shoulder labrum injury",
    "lower back spasm injury",
]

# Injury average days dictionary
average_days_injured = {
    "foot fracture injury": 207.666667,
    "hip flexor surgery injury": 256.000000,
    "calf strain injury": 236.000000,
    "quad injury injury": 283.000000,
    "shoulder sprain injury": 259.500000,
    "foot sprain injury": 294.000000,
    "torn rotator cuff injury injury": 251.500000,
    "torn mcl injury": 271.000000,
    "hip flexor strain injury": 253.000000,
    "fractured leg injury": 250.250000,
    "sprained mcl injury": 228.666667,
    "ankle sprain injury": 231.333333,
    "hamstring injury injury": 220.000000,
    "meniscus tear injury": 201.250000,
    "torn hamstring injury": 187.666667,
    "dislocated shoulder injury": 269.000000,
    "ankle fracture injury": 114.500000,
    "fractured hand injury": 169.142857,
    "bone spurs injury": 151.500000,
    "acl tear injury": 268.000000,
    "hip labrum injury": 247.500000,
    "back surgery injury": 215.800000,
    "arm injury injury": 303.666667,
    "torn shoulder labrum injury": 195.666667,
    "lower back spasm injury": 234.000000,
}

# Load player dataset
@st.cache_resource
def load_player_data():
    return pd.read_csv("player_data.csv")

# Load Random Forest model
@st.cache_resource
def load_rf_model():
    return joblib.load("rf_injury_change_model.pkl")

# Main Streamlit app
def main():
    st.title("NBA Player Performance Predictor 🏀")
    st.write(
        """
        Welcome to the **NBA Player Performance Predictor**! This app helps predict changes in a player's performance metrics
        after experiencing a hypothetical injury. Simply input the details and see the magic happen!
        """
    )

    # Load player data
    player_data = load_player_data()

    # Sidebar inputs
    st.sidebar.header("Player and Injury Input")
    
    st.sidebar.markdown(
        """
        <div style="padding: 10px; background: linear-gradient(to right, #6a11cb, #2575fc); color: white; border-radius: 10px;">
        <h3>Player Details</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Dropdown for player selection
    player_list = sorted(player_data['player_name'].dropna().unique())
    player_name = st.sidebar.selectbox("Select Player", player_list)

    if player_name:
        # Retrieve player details
        player_row = player_data[player_data['player_name'] == player_name]

        if not player_row.empty:
            position = player_row.iloc[0]['position']
            position_numeric = position_mapping.get(position, 0)

            st.sidebar.write(f"**Position**: {position} (Numeric: {position_numeric})")

            # Default values for features
            stats_columns = ['age', 'player_height', 'player_weight']
            default_stats = {
                stat: player_row.iloc[0][stat] if stat in player_row.columns else 0
                for stat in stats_columns
            }

            # Allow manual adjustment of stats
            for stat in default_stats.keys():
                default_stats[stat] = st.sidebar.number_input(f"{stat}", value=float(default_stats[stat]))

            # Injury details
            injury_type = st.sidebar.selectbox("Select Hypothetical Injury", injury_types)
            # Replace slider with default average based on injury type
            default_days_injured = average_days_injured[injury_type]
            days_injured = st.sidebar.slider(
                "Estimated Days Injured",
                0,
                365,
                int(default_days_injured),
                help=f"Default days for {injury_type}: {int(default_days_injured)}"
            )

            injury_occurrences = st.sidebar.number_input("Injury Occurrences", min_value=0, value=1)

            # Prepare input data
            input_data = pd.DataFrame([{
                "days_injured": days_injured,
                "injury_occurrences": injury_occurrences,
                "position": position_numeric,
                "injury_type": injury_type,  # Include the selected injury type
                **default_stats
            }])

            # Encode injury type
            input_data["injury_type"] = pd.factorize(input_data["injury_type"])[0]

            # Load Random Forest model
            rf_model = load_rf_model()

            try:
                # Align input data with the model's feature names
                expected_features = rf_model.feature_names_in_
                input_data = input_data.reindex(columns=expected_features, fill_value=0)

                # Predict and display results
                if st.sidebar.button("Predict 🔮"):
                    predictions = rf_model.predict(input_data)
                    prediction_columns = ["Predicted Change in PTS", "Predicted Change in REB", "Predicted Change in AST"]
                    st.subheader("Predicted Post-Injury Performance")
                    st.write("Based on the inputs, here are the predicted metrics:")
                    styled_table = pd.DataFrame(predictions, columns=prediction_columns).style.set_table_attributes('class="styled-table"')
                    st.write(styled_table.to_html(), unsafe_allow_html=True)

                    # Plot predictions
                    prediction_df = pd.DataFrame(predictions, columns=prediction_columns)
                    fig = go.Figure()

                    for col in prediction_columns:
                        fig.add_trace(go.Bar(
                            x=[col],
                            y=prediction_df[col],
                            name=col,
                            marker=dict(color=px.colors.qualitative.Plotly[prediction_columns.index(col)])
                        ))

                    fig.update_layout(
                        title="Predicted Performance Changes",
                        xaxis_title="Metrics",
                        yaxis_title="Change Value",
                        template="plotly_dark",
                        showlegend=True
                    )

                    st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error: {e}")

if __name__ == "__main__":
    main()
