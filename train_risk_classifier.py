import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -------------------------
# Load dataset
# -------------------------

print("Dataset loaded")

data = pd.read_csv("archive/CVEFixes.csv")
# Remove empty rows
data = data.dropna()

# Features and labels
X = data["code"]
y = data["safety"]

# -------------------------
# Convert code to numbers
# -------------------------

vectorizer = TfidfVectorizer()

X_vec = vectorizer.fit_transform(X)

# -------------------------
# Split dataset
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", X_train.shape[0])

# -------------------------
# Train model
# -------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Risk classifier trained")

# -------------------------
# Evaluate model
# -------------------------

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# -------------------------
# Save model
# -------------------------

os.makedirs("models", exist_ok=True)

pickle.dump(model, open("models/risk_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/risk_vectorizer.pkl", "wb"))

print("Model saved successfully")