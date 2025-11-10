"""
Integration tests for client synchronization module.

Tests the client/sync.py module including upload, download, and status checks
with mocked server responses and retry logic.
"""

import unittest
import tempfile
import os
import shutil
import torch
from unittest.mock import patch, Mock, MagicMock
import requests

from client.sync import upload_weights, download_global_model, check_server_status
from client.model import FederatedModel


class TestSyncModule(unittest.TestCase):
    """Test cases for the client synchronization module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test model and save weights
        self.model = FederatedModel()
        self.test_weights_path = os.path.join(self.temp_dir, 'test_weights.pth')
        self.model.save_weights(self.test_weights_path)
        
        self.server_url = "http://localhost:8000"
        self.client_id = "test_client_001"
        self.n_samples = 1000
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('client.sync.requests.get')
    def test_check_server_status_success(self, mock_get):
        """Test successful server status check."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "server_status": "running",
            "timestamp": "2025-11-10T14:30:00",
            "total_clients": 2,
            "total_samples": 2500,
            "global_model_exists": True
        }
        mock_get.return_value = mock_response
        
        # Call function
        result = check_server_status(self.server_url)
        
        # Verify
        self.assertEqual(result["server_status"], "running")
        self.assertEqual(result["total_clients"], 2)
        mock_get.assert_called_once()
    
    @patch('client.sync.requests.get')
    def test_check_server_status_timeout(self, mock_get):
        """Test server status check with timeout."""
        # Mock timeout
        mock_get.side_effect = requests.exceptions.Timeout()
        
        # Call function and expect exception
        with self.assertRaises(requests.exceptions.Timeout):
            check_server_status(self.server_url)
    
    @patch('client.sync.requests.get')
    def test_check_server_status_connection_error(self, mock_get):
        """Test server status check with connection error."""
        # Mock connection error
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        # Call function and expect exception
        with self.assertRaises(requests.exceptions.ConnectionError):
            check_server_status(self.server_url)
    
    @patch('client.sync.requests.post')
    @patch('client.sync.check_server_status')
    def test_upload_weights_success(self, mock_status, mock_post):
        """Test successful weight upload."""
        # Mock server status check
        mock_status.return_value = {"server_status": "running"}
        
        # Mock successful upload response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "success",
            "client_id": self.client_id,
            "n_samples": self.n_samples,
            "timestamp": "2025-11-10T14:30:00"
        }
        mock_post.return_value = mock_response
        
        # Call function
        result = upload_weights(
            self.server_url,
            self.client_id,
            self.test_weights_path,
            self.n_samples
        )
        
        # Verify
        self.assertTrue(result)
        mock_status.assert_called_once()
        mock_post.assert_called_once()
    
    @patch('client.sync.check_server_status')
    def test_upload_weights_invalid_client_id(self, mock_status):
        """Test upload with invalid client_id."""
        mock_status.return_value = {"server_status": "running"}
        
        # Call with empty client_id
        with self.assertRaises(ValueError):
            upload_weights(
                self.server_url,
                "",
                self.test_weights_path,
                self.n_samples
            )
    
    @patch('client.sync.check_server_status')
    def test_upload_weights_invalid_sample_count(self, mock_status):
        """Test upload with invalid sample count."""
        mock_status.return_value = {"server_status": "running"}
        
        # Call with zero samples
        with self.assertRaises(ValueError):
            upload_weights(
                self.server_url,
                self.client_id,
                self.test_weights_path,
                0
            )
    
    @patch('client.sync.check_server_status')
    def test_upload_weights_file_not_found(self, mock_status):
        """Test upload with non-existent file."""
        mock_status.return_value = {"server_status": "running"}
        
        # Call with non-existent file
        with self.assertRaises(FileNotFoundError):
            upload_weights(
                self.server_url,
                self.client_id,
                "/nonexistent/path.pth",
                self.n_samples
            )
    
    @patch('client.sync.check_server_status')
    def test_upload_weights_invalid_file_extension(self, mock_status):
        """Test upload with invalid file extension."""
        mock_status.return_value = {"server_status": "running"}
        
        # Create a file with wrong extension
        wrong_file = os.path.join(self.temp_dir, 'weights.txt')
        with open(wrong_file, 'w') as f:
            f.write("test")
        
        # Call with wrong extension
        with self.assertRaises(ValueError):
            upload_weights(
                self.server_url,
                self.client_id,
                wrong_file,
                self.n_samples
            )
    
    @patch('client.sync.check_server_status')
    def test_upload_weights_server_unreachable(self, mock_status):
        """Test upload when server is unreachable."""
        # Mock server status check failure
        mock_status.side_effect = requests.exceptions.ConnectionError()
        
        # Call function
        result = upload_weights(
            self.server_url,
            self.client_id,
            self.test_weights_path,
            self.n_samples
        )
        
        # Verify
        self.assertFalse(result)
    
    @patch('client.sync.requests.post')
    @patch('client.sync.check_server_status')
    @patch('client.sync.time.sleep')
    def test_upload_weights_retry_on_timeout(self, mock_sleep, mock_status, mock_post):
        """Test upload retry logic on timeout."""
        # Mock server status check
        mock_status.return_value = {"server_status": "running"}
        
        # Mock timeout on first two attempts, success on third
        mock_post.side_effect = [
            requests.exceptions.Timeout(),
            requests.exceptions.Timeout(),
            Mock(status_code=200, json=lambda: {"status": "success"})
        ]
        
        # Call function
        result = upload_weights(
            self.server_url,
            self.client_id,
            self.test_weights_path,
            self.n_samples,
            max_retries=3
        )
        
        # Verify
        self.assertTrue(result)
        self.assertEqual(mock_post.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)  # Sleep between retries
    
    @patch('client.sync.requests.post')
    @patch('client.sync.check_server_status')
    @patch('client.sync.time.sleep')
    def test_upload_weights_max_retries_exceeded(self, mock_sleep, mock_status, mock_post):
        """Test upload fails after max retries."""
        # Mock server status check
        mock_status.return_value = {"server_status": "running"}
        
        # Mock timeout on all attempts
        mock_post.side_effect = requests.exceptions.Timeout()
        
        # Call function
        result = upload_weights(
            self.server_url,
            self.client_id,
            self.test_weights_path,
            self.n_samples,
            max_retries=3
        )
        
        # Verify
        self.assertFalse(result)
        self.assertEqual(mock_post.call_count, 3)
    
    @patch('client.sync.requests.post')
    @patch('client.sync.check_server_status')
    def test_upload_weights_client_error_no_retry(self, mock_status, mock_post):
        """Test upload doesn't retry on client errors (4xx)."""
        # Mock server status check
        mock_status.return_value = {"server_status": "running"}
        
        # Mock 400 error
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)
        mock_post.return_value = mock_response
        
        # Call function
        result = upload_weights(
            self.server_url,
            self.client_id,
            self.test_weights_path,
            self.n_samples,
            max_retries=3
        )
        
        # Verify - should not retry on client error
        self.assertFalse(result)
        self.assertEqual(mock_post.call_count, 1)
    
    @patch('client.sync.requests.get')
    @patch('client.sync.check_server_status')
    def test_download_global_model_success(self, mock_status, mock_get):
        """Test successful global model download."""
        # Mock server status check
        mock_status.return_value = {
            "server_status": "running",
            "global_model_exists": True
        }
        
        # Mock successful download response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = lambda chunk_size: [b"model_data_chunk"]
        mock_get.return_value = mock_response
        
        # Call function
        save_path = os.path.join(self.temp_dir, 'downloaded_model.pth')
        result = download_global_model(self.server_url, save_path)
        
        # Verify
        self.assertTrue(result)
        self.assertTrue(os.path.exists(save_path))
        mock_status.assert_called_once()
        mock_get.assert_called_once()
    
    @patch('client.sync.check_server_status')
    def test_download_global_model_not_exists(self, mock_status):
        """Test download when global model doesn't exist."""
        # Mock server status indicating no global model
        mock_status.return_value = {
            "server_status": "running",
            "global_model_exists": False
        }
        
        # Call function
        save_path = os.path.join(self.temp_dir, 'downloaded_model.pth')
        result = download_global_model(self.server_url, save_path)
        
        # Verify
        self.assertFalse(result)
        self.assertFalse(os.path.exists(save_path))
    
    @patch('client.sync.check_server_status')
    def test_download_global_model_server_unreachable(self, mock_status):
        """Test download when server is unreachable."""
        # Mock server status check failure
        mock_status.side_effect = requests.exceptions.ConnectionError()
        
        # Call function
        save_path = os.path.join(self.temp_dir, 'downloaded_model.pth')
        result = download_global_model(self.server_url, save_path)
        
        # Verify
        self.assertFalse(result)
        self.assertFalse(os.path.exists(save_path))
    
    @patch('client.sync.requests.get')
    @patch('client.sync.check_server_status')
    @patch('client.sync.time.sleep')
    def test_download_global_model_retry_on_timeout(self, mock_sleep, mock_status, mock_get):
        """Test download retry logic on timeout."""
        # Mock server status check
        mock_status.return_value = {
            "server_status": "running",
            "global_model_exists": True
        }
        
        # Mock timeout on first attempt, success on second
        success_response = Mock()
        success_response.status_code = 200
        success_response.iter_content = lambda chunk_size: [b"model_data"]
        
        mock_get.side_effect = [
            requests.exceptions.Timeout(),
            success_response
        ]
        
        # Call function
        save_path = os.path.join(self.temp_dir, 'downloaded_model.pth')
        result = download_global_model(self.server_url, save_path, max_retries=3)
        
        # Verify
        self.assertTrue(result)
        self.assertEqual(mock_get.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)
    
    @patch('client.sync.requests.get')
    @patch('client.sync.check_server_status')
    @patch('client.sync.time.sleep')
    def test_download_global_model_max_retries_exceeded(self, mock_sleep, mock_status, mock_get):
        """Test download fails after max retries."""
        # Mock server status check
        mock_status.return_value = {
            "server_status": "running",
            "global_model_exists": True
        }
        
        # Mock connection error on all attempts
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        # Call function
        save_path = os.path.join(self.temp_dir, 'downloaded_model.pth')
        result = download_global_model(self.server_url, save_path, max_retries=3)
        
        # Verify
        self.assertFalse(result)
        self.assertEqual(mock_get.call_count, 3)
    
    @patch('client.sync.requests.get')
    @patch('client.sync.check_server_status')
    def test_download_global_model_creates_directory(self, mock_status, mock_get):
        """Test download creates directory if it doesn't exist."""
        # Mock server status check
        mock_status.return_value = {
            "server_status": "running",
            "global_model_exists": True
        }
        
        # Mock successful download
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content = lambda chunk_size: [b"model_data"]
        mock_get.return_value = mock_response
        
        # Call function with nested directory
        save_path = os.path.join(self.temp_dir, 'subdir', 'model.pth')
        result = download_global_model(self.server_url, save_path)
        
        # Verify
        self.assertTrue(result)
        self.assertTrue(os.path.exists(save_path))
        self.assertTrue(os.path.exists(os.path.dirname(save_path)))


if __name__ == '__main__':
    unittest.main()
