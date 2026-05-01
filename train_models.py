import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv("final_combined_dataset.csv")

features = [
    'ph','Hardness','Solids','Conductivity','Turbidity',
    'Pressure_(bar)','Flow_Rate_(L/s)','Temperature_(°C)',
    'Pressure_Flow_Ratio'
]

X = data[features]

# 🔹 Leak model
y_leak = data['Leak_Status']
leak_model = RandomForestClassifier(n_estimators=100, random_state=42)
leak_model.fit(X, y_leak)
joblib.dump(leak_model, "leak_model.pkl")

# 🔹 Water quality model
y_water = data['Potability']
water_model = RandomForestClassifier(n_estimators=100, random_state=42)
water_model.fit(X, y_water)
joblib.dump(water_model, "water_quality_model.pkl")

print("✅ Models trained & saved successfully")

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ---------------- LEAK MODEL ACCURACY ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_leak, test_size=0.2, random_state=42
)

leak_model.fit(X_train, y_train)
leak_pred_test = leak_model.predict(X_test)

leak_accuracy = accuracy_score(y_test, leak_pred_test)
print(f"Leak Detection Accuracy: {leak_accuracy*100:.2f}%")

# ---------------- WATER QUALITY ACCURACY ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_water, test_size=0.2, random_state=42
)

water_model.fit(X_train, y_train)
water_pred_test = water_model.predict(X_test)

water_accuracy = accuracy_score(y_test, water_pred_test)
print(f"Water Quality Accuracy: {water_accuracy*100:.2f}%")
