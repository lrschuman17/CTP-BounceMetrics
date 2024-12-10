import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

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
    "lower back spasm injury"
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
    st.title("NBA Player Performance Predictor")
    st.write(
        """
        Predict how a player's performance metrics (e.g., points, rebounds, assists) might change
        if a hypothetical injury occurs, based on their position and other factors.
        """
    )

    # Load player data
    player_data = load_player_data()
    rf_model = load_rf_model()

    # Sidebar inputs
    st.sidebar.header("Player and Injury Input")

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
                default_stats[stat] = st.sidebar.number_input(f"{stat}", value=default_stats[stat])

            # Injury details
            injury_type = st.sidebar.selectbox("Select Hypothetical Injury", injury_types)
            # Replace slider with default average based on injury type
            default_days_injured = average_days_injured[injury_type] or 30  # Use 30 if `None`
            days_injured = st.sidebar.slider(
                "Estimated Days Injured",
                0,
                365,
                int(default_days_injured),
                help=f"Default days for {injury_type}: {int(default_days_injured) if default_days_injured else 'N/A'}"
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
            try:
                rf_model = load_rf_model()

                # Align input data with the model's feature names
                expected_features = rf_model.feature_names_in_
                input_data = input_data.reindex(columns=rf_model.feature_names_in_, fill_value=0)

                # Predict and display results
                if st.sidebar.button("Predict"):
                    predictions = rf_model.predict(input_data)
                    prediction_columns = ["Predicted Change in PTS", "Predicted Change in REB", "Predicted Change inAST"]
                    st.subheader("Predicted Post-Injury Performance")
                    st.write("Based on the inputs, here are the predicted metrics:")
                    st.table(pd.DataFrame(predictions, columns=prediction_columns))
            except FileNotFoundError:
                st.error("Model file not found.")
            except ValueError as e:
                st.error(f"Error during prediction: {e}")

        else:
            st.sidebar.error("Player details not found in the dataset.")
    else:
        st.sidebar.error("Please select a player to view details.")

if __name__ == "__main__":
    main()
