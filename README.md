# ML Solar Wastage
Daily Solar Energy Wastage Prediction using Machine Learning

1. Project Overview

With the rapid increase in household rooftop solar installations, many households export excess solar energy to the electricity grid during sunny days, especially around midday. However, due to grid absorption limitations and low demand during these hours, not all exported solar energy can be utilized. This results in energy wastage (curtailment) at the grid level.

The objective of this Machine Learning (ML) project is to predict the daily amount of solar energy wasted after export to the grid, using historical solar generation and household load patterns from real German household data (DE_KN region).

2. Problem Definition

Goal:
Predict the daily solar energy wastage (in kWh) that occurs after households export excess solar energy to the grid.

ML Task Type:
Regression

Target Variable (Dependent Variable):
Daily_Wasted_Energy_kWh

3. Data Sources

3.1 German Household Energy Dataset

This dataset provides real household-level time-series data for solar PV generation and electricity consumption from the DE_KN (Germany, Konstanz) region.

Dataset: `household_data_60min_singleindex.csv`

Time Period: 2014-2016 (hourly data)

The dataset includes 6 residential households with the following features:

**Available Households with Solar PV:**
- DE_KN_residential1: PV generation, heat pump, grid import
- DE_KN_residential3: PV generation, grid export/import (solar surplus household)
- DE_KN_residential4: PV generation, grid export/import, EV, heat pump
- DE_KN_residential6: PV generation, grid export/import

**Data Fields per Household:**
- Timestamp (hourly)
- PV generation (kWh)
- Grid import/export (kWh)
- Individual appliance consumption (dishwasher, washing machine, freezer, heat pump, EV, etc.)

3.2 Weather Data (Optional Enhancement)

Weather data for the DE_KN region can be obtained from:
- Open-Meteo API (free historical weather data)
- DWD (German Weather Service) open data

4. Independent Variables (X)

The following features are used as inputs to the ML models:

**Time-based Features**
- Day of week (0-6)
- Month (1-12)
- Is weekend (1/0)
- Season (Winter/Spring/Summer/Fall)
- Hour of day (for hourly aggregation patterns)

**Household Identifier**
- Household ID (residential1/3/4/6)

**Energy Behaviour Features**
- Daily solar PV generation (kWh)
- Daily household total consumption (kWh)
- Daily grid import (kWh)
- Previous day solar generation (kWh)
- Previous day consumption (kWh)
- Previous day grid export (kWh)
- Rolling average PV generation (7-day window)
- Rolling average consumption (7-day window)

**Weather & Solar Conditions (Optional)**
- Daily solar irradiance estimate
- Average temperature
- Cloud cover

5. Dependent Variable (Y)

Daily Wasted Energy Calculation

The daily wasted energy is computed using the actual grid export data:

**Method 1: Using Actual Grid Export Data**
- The dataset contains `grid_export` columns for households with solar
- Daily_Grid_Export = sum of hourly grid_export values
- This represents actual energy sent to the grid

**Method 2: Calculated Surplus**
- Net_Production = Daily_PV_Generation - Daily_Household_Consumption
- Exportable_Energy = max(0, Net_Production)

**Wastage Calculation:**
- Grid Export Limit (Assumption): 5 kWh per day per household
- (Represents grid absorption limitations during high solar penetration)
- Daily_Wasted_Energy = max(0, Daily_Grid_Export - Export_Limit)

**Scenarios:**
- If Daily_Grid_Export ≤ Export_Limit → Wasted_Energy = 0 (grid accepts all)
- If Daily_Grid_Export > Export_Limit → Wasted_Energy = Daily_Grid_Export - Export_Limit

This calculated value becomes the dependent variable:
Daily_Wasted_Energy_kWh

6. Data Preparation Steps

1. **Load and inspect data**
   - Load household_data_60min_singleindex.csv
   - Examine structure, missing values, data types
   - Filter households with PV: residential1, 3, 4, 6

2. **Data cleaning**
   - Handle missing values (forward fill or interpolation)
   - Remove or impute outliers
   - Convert timestamps to datetime

3. **Feature calculation**
   - Calculate total household consumption per hour (sum of all appliances + grid_import)
   - Aggregate hourly → daily totals for PV, consumption, grid_export
   - Extract time-based features (day, month, weekday, weekend, season)

4. **Target variable creation**
   - Calculate Daily_Grid_Export
   - Apply export limit threshold (5 kWh)
   - Calculate Daily_Wasted_Energy_kWh

5. **Feature engineering**
   - Create lag features (previous 1, 3, 7 days)
   - Create rolling averages (7-day, 14-day windows)
   - Encode categorical variables (household ID, season)

6. **Data splitting and scaling**
   - Train/test split (80/20 or temporal split)
   - Normalize/standardize numeric features

7. Machine Learning Models Used

The following regression algorithms will be trained and compared:

Linear Regression

Ridge Regression

Decision Tree Regressor

Random Forest Regressor

K-Nearest Neighbors (KNN) Regressor

Gradient Boosting Regressor

Support Vector Regressor (Optional)

8. Model Evaluation Metrics

Each model is evaluated using:

Mean Absolute Error (MAE)

Root Mean Squared Error (RMSE)

R² Score

The best-performing model is selected based on the lowest RMSE.

9. Web Dashboard (ML Coursework Demonstration)

A simple web dashboard will be developed using Streamlit to present results.

Dashboard Features:

- Dataset overview and statistics
- Model comparison table (MAE, RMSE, R²)
- Actual vs predicted wasted energy chart
- Household selection (Residential 1/3/4/6)
- Time series visualization of PV generation and wastage
- Interactive prediction form (input daily PV, consumption, date)
- Feature importance visualization
- Energy wastage patterns by season/month

10. Conclusion

This project demonstrates how machine learning can be used to analyze and predict solar energy wastage due to grid management limitations. By leveraging publicly available datasets and Sri Lanka–specific solar data, the system provides insights that can support better solar energy utilization planning.

11. Tools & Technologies

Python

Pandas, NumPy

Scikit-learn

Streamlit

Kaggle Datasets

NASA POWER Data

