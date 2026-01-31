from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
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

# Calibrate model
calibrated_model = CalibratedClassifierCV(
    model, method='sigmoid', cv='prefit'
)
calibrated_model.fit(X_train, y_train)

# Save calibrated model
os.makedirs('models', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
joblib.dump(calibrated_model, f'models/calibrated_model_{timestamp}.pkl')
joblib.dump(calibrated_model, 'models/latest_calibrated_model.pkl')

print(f"Calibrated model saved at models/calibrated_model_{timestamp}.pkl")
