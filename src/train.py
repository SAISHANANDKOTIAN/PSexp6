import os
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Train a simple model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Ensure the 'model' directory exists
os.makedirs('model', exist_ok=True)

# Save the trained model to the 'model' directory
with open('model/iris_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved to 'model/iris_model.pkl'")
