import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
data = pd.read_csv("archive/CVEFixes.csv")

# Remove missing code
data = data.dropna(subset=["code"])

# Input and label
X = data["code"].astype(str)
y = data["language"]

# Convert text to vectors
vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Language model trained")

# Save model
pickle.dump(model, open("models/language_model.pkl", "wb"))
pickle.dump(vectorizer, open("models/language_vectorizer.pkl", "wb"))

print("Model saved")  