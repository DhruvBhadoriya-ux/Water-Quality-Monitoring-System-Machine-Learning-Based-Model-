import pandas as pd
# LOAD DATASETS
water = pd.read_csv("water_potability.csv")
leak = pd.read_csv("water_leakage.csv")
# FIX COLUMN NAMES (optional but recommended)
leak.columns = leak.columns.str.strip().str.replace(" ", "_")
#CONVERT TIMESTAMP (Leak data already has it)
leak['Timestamp'] = pd.to_datetime(leak['Timestamp'])
#CREATE SYNTHETIC TIMESTAMP FOR WATER DATA
water['Timestamp'] = pd.date_range(
    start=leak['Timestamp'].min(),
    periods=len(water),
    freq='5T'   # 5 minutes interval
)
water['Timestamp'] = pd.to_datetime(water['Timestamp'])
#SORT BOTH DATASETS (IMPORTANT for merge_asof)
water.sort_values('Timestamp', inplace=True)
leak.sort_values('Timestamp', inplace=True)
#SMART MERGE (NO DATA LOSS)
final_data = pd.merge_asof(
    water,
    leak,
    on='Timestamp',
    direction='nearest'
)
# HANDLE MISSING VALUES
final_data.fillna(final_data.mean(numeric_only=True), inplace=True)
# OPTIONAL FEATURE ENGINEERING (BONUS)
final_data['Pressure_Flow_Ratio'] = (
    final_data['Pressure_(bar)'] / final_data['Flow_Rate_(L/s)']
)
# SAVE FINAL DATASET

final_data.to_csv("final_combined_dataset.csv", index=False)
# CHECK RESULT
