import pandas as pd
final_data = pd.read_csv("final_combined_dataset.csv")
X = final_data.drop(['Potability', 'Leak_Status', 'Timestamp'], axis=1)

y_water = final_data['Potability']      # Water Quality
y_leak = final_data['Leak_Status']      # Leak Detection
print(X)
from sklearn.model_selection import train_test_split
X_train, X_test, yw_train, yw_test = train_test_split(
    X, y_water, test_size=0.2, random_state=42)
_, _, yl_train, yl_test = train_test_split(
    X, y_leak, test_size=0.2, random_state=42
)
from sklearn.ensemble import RandomForestClassifier

water_model = RandomForestClassifier(n_estimators=200)
water_model.fit(X_train, yw_train)
leak_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced'
)
leak_model.fit(X_train, yl_train)
from sklearn.metrics import classification_report

print("WATER QUALITY MODEL")
print(classification_report(yw_test, water_model.predict(X_test)))

print("LEAK DETECTION MODEL")
print(classification_report(yl_test, leak_model.predict(X_test)))
import joblib

joblib.dump(water_model, "water_quality_model.pkl")
joblib.dump(leak_model, "leak_detection_model.pkl")