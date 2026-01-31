from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


os.makedirs('models', exist_ok=True)

# Save model 
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
joblib.dump(model, f'models/model_{timestamp}.pkl')
joblib.dump(model, 'models/latest_model.pkl')

print(f"Model trained and saved at models/model_{timestamp}.pkl")
