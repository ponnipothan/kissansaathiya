import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("crop_ai.csv")

# Encode soil
le = LabelEncoder()
data["soil"] = le.fit_transform(data["soil"])

# Features and target
X = data[["soil", "temp", "humidity"]]
y = data["crop"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("ai_model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("AI model trained!")