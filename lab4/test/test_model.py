import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from sklearn.linear_model import LogisticRegression

import sys
sys.path.insert(0, '..')

from src.train_and_save_model import (
    download_data,
    preprocess_data,
    train_model,
    evaluate_model
)


class TestDataLoading:
    """Tests for data loading function"""
    
    def test_download_data_shape(self):
        """Test that data has correct shape"""
        X, y = download_data()
        assert X.shape[0] == 150, "Should have 150 samples"
        assert X.shape[1] == 4, "Should have 4 features"
    
    def test_download_data_labels(self):
        """Test that labels are correct"""
        X, y = download_data()
        assert len(y) == 150, "Should have 150 labels"
        assert set(y.unique()) == {0, 1, 2}, "Should have 3 classes"


class TestPreprocessing:
    """Tests for data preprocessing"""
    
    def test_preprocess_split_size(self):
        """Test train/test split sizes"""
        X, y = download_data()
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        
        assert len(X_train) == 120, "Train set should have 120 samples"
        assert len(X_test) == 30, "Test set should have 30 samples"
    
    def test_preprocess_reproducibility(self):
        """Test that split is reproducible"""
        X, y = download_data()
        
        X_train1, X_test1, _, _ = preprocess_data(X, y)
        X_train2, X_test2, _, _ = preprocess_data(X, y)
        
        assert X_train1.equals(X_train2), "Split should be reproducible"


class TestModelTraining:
    """Tests for model training"""
    
    def test_train_model_type(self):
        """Test that model is correct type"""
        X, y = download_data()
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        
        model = train_model(X_train, y_train)
        
        assert isinstance(model, LogisticRegression)
    
    def test_train_model_can_predict(self):
        """Test that model can make predictions"""
        X, y = download_data()
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        
        assert len(predictions) == 30


class TestModelEvaluation:
    """Tests for model evaluation"""
    
    def test_evaluate_model_metrics_range(self):
        """Test that metrics are in valid range"""
        X, y = download_data()
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        model = train_model(X_train, y_train)
        
        accuracy, f1 = evaluate_model(model, X_test, y_test)
        
        assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
        assert 0 <= f1 <= 1, "F1 score should be between 0 and 1"
    
    def test_evaluate_model_good_performance(self):
        """Test that model performs reasonably well"""
        X, y = download_data()
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        model = train_model(X_train, y_train)
        
        accuracy, f1 = evaluate_model(model, X_test, y_test)
        
        assert accuracy > 0.8, "Accuracy should be above 80%"
        assert f1 > 0.8, "F1 score should be above 80%"


class TestGCSFunctions:
    """Tests for GCS functions using mocks"""
    
    @patch('src.train_and_save_model.storage.Client')
    def test_save_model_to_gcs(self, mock_storage_client):
        """Test GCS upload with mock"""
        # Setup mock
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Train a model
        X, y = download_data()
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        model = train_model(X_train, y_train)
        
        # Test upload
        from src.train_and_save_model import save_model_to_gcs
        save_model_to_gcs(model, "test-bucket", "test-model.joblib")
        
        # Verify mock was called
        mock_storage_client.return_value.bucket.assert_called_once_with("test-bucket")
        mock_bucket.blob.assert_called_once_with("test-model.joblib")
        mock_blob.upload_from_file.assert_called_once()
    
    @patch('src.train_and_save_model.storage.Client')
    def test_get_model_version_exists(self, mock_storage_client):
        """Test getting version when file exists"""
        # Setup mock
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = True
        mock_blob.download_as_text.return_value = "5"
        
        # Test
        from src.train_and_save_model import get_model_version
        version = get_model_version("test-bucket", "version.txt")
        
        assert version == 5
    
    @patch('src.train_and_save_model.storage.Client')
    def test_get_model_version_not_exists(self, mock_storage_client):
        """Test getting version when file doesn't exist"""
        # Setup mock
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        mock_blob.exists.return_value = False
        
        # Test
        from src.train_and_save_model import get_model_version
        version = get_model_version("test-bucket", "version.txt")
        
        assert version == 0
