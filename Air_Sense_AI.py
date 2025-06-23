import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# load data 
df = pd.read_csv("Air_Quality.csv")

# Drop rows with missing values and Feature selection
df.dropna(inplace=True)
feature_cols = ['CO', 'NO2', 'SO2', 'O3', 'PM2.5', 'PM10']
X = df[feature_cols].values

# labels (1 = polluted, 0 = not polluted) based on AQI threshold 
df['Pollution_Status'] = df['AQI'].apply(lambda x: 1 if x >= 50 else 0)
y = df['Pollution_Status'].values

#  makesome noise 
X += np.random.normal(0, 0.1, X.shape)

#Standardize features 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#Train the classifier 
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

#Save the model and scaler
joblib.dump(clf, "Air_Quality_model.pkl")
joblib.dump(scaler, "scaler.pkl")

#Predict on latest data row
latest_row = df.iloc[-1][feature_cols].values.reshape(1, -1)
latest_row_scaled = scaler.transform(latest_row)
prediction = clf.predict(latest_row_scaled)
proba = clf.predict_proba(latest_row_scaled)[0]

print("\n🔍 Latest Data Row Prediction:")
print("Prediction:", "Polluted" if prediction[0] == 1 else "Not Polluted")
print(f"Confidence: {proba[prediction[0]]:.2%}")

#Predict on custom input 
def predict_pollution(co, no2, so2, o3, pm25, pm10):
    model = joblib.load("Air_Quality_model.pkl")
    scaler = joblib.load("scaler.pkl")
    data = np.array([[co, no2, so2, o3, pm25, pm10]])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]
    proba = model.predict_proba(data_scaled)[0]
    return {
        "prediction": "Polluted" if prediction == 1 else "Not Polluted",
        "confidence": f"{proba[prediction]*100:.2f}%"
    }

#Example usage of the prediction function 
result = predict_pollution(170.0, 26.0, 1.2, 35.0, 17.2, 22.3)
print("\n🔮 Custom Input Prediction:")
print(result)
