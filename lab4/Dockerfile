# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/

# Copy trained model
COPY trained_models/ ./trained_models/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to verify model loads correctly
CMD ["python", "-c", "import joblib; model = joblib.load('trained_models/model.joblib'); print('Model loaded successfully!'); print(f'Model type: {type(model).__name__}')"]
