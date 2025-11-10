"""
Integration tests for FastAPI server endpoints.

Tests the REST API endpoints including weight upload, aggregation,
global model download, and status queries.
"""

import unittest
import tempfile
import os
import shutil
import torch
from fastapi.testclient import TestClient
from io import BytesIO

from central.server import app
from central.model import FederatedModel
from central.state import clear_client_registry


class TestServerEndpoints(unittest.TestCase):
    """Test cases for the FastAPI server endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()
        
        # Clear client registry before each test
        clear_client_registry()
        
        # Create a test model and save weights
        self.model = FederatedModel()
        self.test_weights_path = os.path.join(self.temp_dir, 'test_weights.pth')
        self.model.save_weights(self.test_weights_path)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        
        # Clean up model store
        model_store = "./central/model_store"
        if os.path.exists(model_store):
            for file in os.listdir(model_store):
                file_path = os.path.join(model_store, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception:
                    pass
    
    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = self.client.get("/")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("endpoints", data)
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
    
    def test_upload_weights_success(self):
        """Test successful weight upload."""
        # Read test weights file
        with open(self.test_weights_path, 'rb') as f:
            weights_content = f.read()
        
        # Upload weights
        response = self.client.post(
            "/upload_weights/client_001?n_samples=1000",
            files={"file": ("weights.pth", BytesIO(weights_content), "application/octet-stream")}
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["client_id"], "client_001")
        self.assertEqual(data["n_samples"], 1000)
        self.assertIn("timestamp", data)
    
    def test_upload_weights_invalid_client_id(self):
        """Test upload with invalid client_id."""
        with open(self.test_weights_path, 'rb') as f:
            weights_content = f.read()
        
        response = self.client.post(
            "/upload_weights/ ?n_samples=1000",
            files={"file": ("weights.pth", BytesIO(weights_content), "application/octet-stream")}
        )
        
        self.assertEqual(response.status_code, 400)
    
    def test_upload_weights_invalid_sample_count(self):
        """Test upload with invalid sample count."""
        with open(self.test_weights_path, 'rb') as f:
            weights_content = f.read()
        
        response = self.client.post(
            "/upload_weights/client_001?n_samples=0",
            files={"file": ("weights.pth", BytesIO(weights_content), "application/octet-stream")}
        )
        
        self.assertEqual(response.status_code, 400)
    
    def test_upload_weights_invalid_file_format(self):
        """Test upload with invalid file format."""
        response = self.client.post(
            "/upload_weights/client_001?n_samples=1000",
            files={"file": ("weights.txt", BytesIO(b"invalid"), "text/plain")}
        )
        
        self.assertEqual(response.status_code, 400)
    
    def test_status_endpoint(self):
        """Test status endpoint returns server information."""
        response = self.client.get("/status")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["server_status"], "running")
        self.assertIn("timestamp", data)
        self.assertIn("total_clients", data)
        self.assertIn("clients", data)
    
    def test_status_after_upload(self):
        """Test status endpoint shows uploaded clients."""
        # Upload weights from a client
        with open(self.test_weights_path, 'rb') as f:
            weights_content = f.read()
        
        self.client.post(
            "/upload_weights/client_001?n_samples=1000",
            files={"file": ("weights.pth", BytesIO(weights_content), "application/octet-stream")}
        )
        
        # Check status
        response = self.client.get("/status")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_clients"], 1)
        self.assertEqual(data["total_samples"], 1000)
        self.assertEqual(len(data["clients"]), 1)
        self.assertEqual(data["clients"][0]["client_id"], "client_001")
    
    def test_aggregate_no_clients(self):
        """Test aggregation with no uploaded weights."""
        response = self.client.post("/aggregate")
        
        self.assertEqual(response.status_code, 400)
    
    def test_aggregate_success(self):
        """Test successful aggregation."""
        # Upload weights from two clients
        with open(self.test_weights_path, 'rb') as f:
            weights_content = f.read()
        
        self.client.post(
            "/upload_weights/client_001?n_samples=1000",
            files={"file": ("weights.pth", BytesIO(weights_content), "application/octet-stream")}
        )
        
        self.client.post(
            "/upload_weights/client_002?n_samples=1500",
            files={"file": ("weights.pth", BytesIO(weights_content), "application/octet-stream")}
        )
        
        # Trigger aggregation
        response = self.client.post("/aggregate")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["num_clients"], 2)
        self.assertEqual(data["total_samples"], 2500)
        self.assertIn("timestamp", data)
    
    def test_global_model_not_found(self):
        """Test downloading global model when it doesn't exist."""
        response = self.client.get("/global_model")
        
        self.assertEqual(response.status_code, 404)
    
    def test_global_model_download_after_aggregation(self):
        """Test downloading global model after aggregation."""
        # Upload weights and aggregate
        with open(self.test_weights_path, 'rb') as f:
            weights_content = f.read()
        
        self.client.post(
            "/upload_weights/client_001?n_samples=1000",
            files={"file": ("weights.pth", BytesIO(weights_content), "application/octet-stream")}
        )
        
        self.client.post("/aggregate")
        
        # Download global model
        response = self.client.get("/global_model")
        
        self.assertEqual(response.status_code, 200)
        self.assertGreater(len(response.content), 0)
    
    def test_concurrent_uploads(self):
        """Test concurrent uploads from multiple clients."""
        with open(self.test_weights_path, 'rb') as f:
            weights_content = f.read()
        
        # Upload from 3 clients
        clients = ["client_001", "client_002", "client_003"]
        sample_counts = [1000, 1500, 2000]
        
        for client_id, n_samples in zip(clients, sample_counts):
            response = self.client.post(
                f"/upload_weights/{client_id}?n_samples={n_samples}",
                files={"file": ("weights.pth", BytesIO(weights_content), "application/octet-stream")}
            )
            self.assertEqual(response.status_code, 200)
        
        # Check status shows all clients
        response = self.client.get("/status")
        data = response.json()
        
        self.assertEqual(data["total_clients"], 3)
        self.assertEqual(data["total_samples"], 4500)
        
        # Verify aggregation includes all clients
        response = self.client.post("/aggregate")
        data = response.json()
        
        self.assertEqual(data["num_clients"], 3)
        self.assertEqual(data["total_samples"], 4500)


if __name__ == '__main__':
    unittest.main()
