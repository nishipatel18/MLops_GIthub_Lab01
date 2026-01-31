# Mocking in Tests

## What is Mocking?

Mocking is a technique used in testing to replace real objects with fake ones. This is useful when:

- Testing code that interacts with external services (like GCS)
- You don't want to make actual API calls during tests
- You want to test specific scenarios (like errors)

## How We Use Mocking in Lab 4

We use `unittest.mock` to mock Google Cloud Storage:
```python
from unittest.mock import MagicMock, patch

@patch('src.train_and_save_model.storage.Client')
def test_save_model_to_gcs(self, mock_storage_client):
    # Setup mock
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_storage_client.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    
    # Now when code calls storage.Client(), it gets our mock
```

## Why Mock GCS?

1. **Speed**: No network calls needed
2. **Cost**: No GCS charges during testing
3. **Reliability**: Tests don't fail due to network issues
4. **Control**: Can simulate errors and edge cases
