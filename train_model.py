import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

# 1 Load dataset
data = pd.read_csv("archive/CVEFixes.csv")

print("Dataset loaded")

# 2 Remove rows with missing code
data = data.dropna(subset=["code"])

# 3 Input and output
X = data["code"].astype(str)
y = data["safety"]

# 4 Convert code to numeric vectors
vectorizer = TfidfVectorizer(max_features=5000)

X_vec = vectorizer.fit_transform(X)

# 5 Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# 6 Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained")

# 7 Save model
pickle.dump(model, open("vulnerability_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully")