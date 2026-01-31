from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
from datetime import datetime

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Same split as training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Load trained model
model = joblib.load('models/latest_model.pkl')

# Make predictions
predictions = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')

os.makedirs('metrics', exist_ok=True)

# Save metrics 
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f'metrics/evaluation_{timestamp}.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"F1 Score: {f1}\n")

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
