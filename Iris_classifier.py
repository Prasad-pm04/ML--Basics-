# iris_classifier.py

# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data     # Features
y = iris.target   # Target labels

# Step 2: Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 4: Make predictions
predictions = model.predict(X_test)

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print("✅ Model Accuracy:", round(accuracy * 100, 2), "%")
