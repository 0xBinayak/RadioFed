# Federated Learning for Analog Modulation Classification

A federated learning system for automatic modulation classification (AMC) using K-Nearest Neighbors on RadioML 2016.10a dataset. Supports distributed training across multiple clients with automatic aggregation and real-time monitoring.

## Features

- **KNN-Only Architecture**: Simplified ML approach using scikit-learn's KNeighborsClassifier
- **Feature Extraction**: 8 instantaneous features (amplitude + frequency statistics) from I/Q signals
- **Federated Aggregation**: Merges training data from multiple clients to create global model
- **Auto-Aggregation**: Automatic model aggregation when threshold is met (default: 2 clients)
- **Real-Time Dashboard**: Live monitoring with accuracy trends, confusion matrices, and performance metrics
- **Per-SNR Analysis**: Accuracy breakdown from -20 to 18 dB

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# or
uv sync
```

### 2. Partition Dataset

```bash
python partition_dataset.py
```

Creates balanced partitions for federated learning from RadioML 2016.10a dataset.

### 3. Start Central Server

```bash
python central/main.py
```

Server runs on `http://localhost:8000` with dashboard at `http://localhost:7860`

### 4. Start Clients

```bash
# Terminal 1
python client/main.py --port 7861 --auto-id

# Terminal 2
python client/main.py --port 7862 --auto-id

# Terminal 3
python client/main.py --port 7863 --auto-id
```

### 5. Train on Each Client

1. Open client UI (e.g., `http://localhost:7861`)
2. Click "Load Partition"
3. Click "Extract Features" (wait 1-3 minutes)
4. Click "Train Model" (auto-uploads when complete)

### 6. View Results

Open dashboard at `http://localhost:7860` to see:
- Real-time accuracy trends
- Confusion matrices
- Per-SNR performance
- Client status and metrics

## Architecture

### Client Side

**Feature Extraction** (`client/train.py`):
```python
def extract_analog_features(signal, fs=128):
    # Extracts 8 features from I/Q signals:
    # - Amplitude: mean, variance, skewness, kurtosis
    # - Frequency: mean, variance, skewness, kurtosis
```

**KNN Training**:
- Uses `sklearn.neighbors.KNeighborsClassifier`
- Default: `n_neighbors=5`, `test_split=0.3`
- Measures training time and inference time
- Generates confusion matrix

### Server Side

**Aggregation** (`central/aggregator.py`):
- Merges training data from all clients
- Retrains global KNN on combined dataset
- Evaluates with per-SNR accuracy breakdown

**API Endpoints**:
- `POST /register/{client_id}` - Register client
- `POST /upload_model/{client_id}` - Upload KNN model + features + labels
- `POST /aggregate` - Trigger aggregation
- `GET /global_model` - Download global model
- `GET /status` - Server status

## Configuration

Edit `central/config.json`:

```json
{
  "model_save_path": "./central/model_store/global_knn_model.pkl",
  "host": "0.0.0.0",
  "port": 8000,
  "log_level": "INFO",
  "auto_aggregation_enabled": true,
  "auto_aggregation_threshold": 2
}
```

## Dataset

**RadioML 2016.10a** - Analog modulations only:
- **AM**: AM-DSB + AM-SSB (combined)
- **FM**: WBFM
- **SNR Range**: -20 to 18 dB (2 dB steps)
- **Samples**: 128 I/Q samples per signal

## Performance

**KNN Model**:
- Training time: ~1-2 seconds
- Inference time: ~1-2 ms/sample
- Accuracy (typical):
  - High SNR (>10 dB): 90-95%
  - Medium SNR (0-10 dB): 70-85%
  - Low SNR (<0 dB): 50-70%

## Project Structure

```
.
├── client/
│   ├── main.py              # Client Gradio UI
│   ├── train.py             # KNN training + feature extraction
│   ├── dataset_loader.py    # Dataset loading
│   ├── feature_extract.py   # Feature processing
│   └── sync.py              # Server communication
├── central/
│   ├── main.py              # Server entry point
│   ├── server.py            # FastAPI endpoints
│   ├── aggregator.py        # KNN aggregation logic
│   ├── dashboard.py         # Gradio dashboard
│   ├── visualization.py     # Plotting functions
│   └── state.py             # State management
├── partition_dataset.py     # Dataset partitioning
└── pyproject.toml          # Dependencies
```

## API Usage

### Upload Model

```bash
curl -X POST "http://localhost:8000/upload_model/client1?n_samples=1000" \
  -F "model_file=@model.pkl" \
  -F "features_file=@features.pkl" \
  -F "labels_file=@labels.pkl"
```

### Trigger Aggregation

```bash
curl -X POST "http://localhost:8000/aggregate"
```

### Download Global Model

```bash
curl -O "http://localhost:8000/global_model"
```

## Dashboard Metrics

- **System Status**: Server health, connected clients, current round
- **Accuracy Trends**: Historical before/after aggregation performance
- **Confusion Matrix**: KNN classification results
- **Accuracy vs SNR**: Performance across signal-to-noise ratios
- **Complexity**: Training and inference time metrics

## Troubleshooting

**Issue**: Client can't connect to server  
**Solution**: Ensure server is running on port 8000

**Issue**: Feature extraction is slow  
**Solution**: Normal for large partitions (1-3 minutes), be patient

**Issue**: Low accuracy  
**Solution**: Check SNR levels, ensure enough training data, verify aggregation completed

**Issue**: Aggregation not triggering  
**Solution**: Check threshold in config, manually trigger with `/aggregate` endpoint

## Development

### Run Tests

```bash
pytest tests/
```

### Check Server Status

```bash
python diagnose_server.py
```

### View Logs

Server logs are in console output. Adjust log level in `central/config.json`.

## Technical Details

### Feature Extraction

From each 128-sample I/Q signal:
1. Compute instantaneous amplitude: `|I + jQ|`
2. Compute instantaneous phase: `unwrap(angle(I + jQ))`
3. Compute instantaneous frequency: `diff(phase) / (2π) * fs`
4. Extract statistics: mean, variance, skewness, kurtosis

### Aggregation Strategy

1. Collect training data (features + labels) from all clients
2. Merge into single dataset
3. Train global KNN on combined data
4. Evaluate on test split (20%)
5. Compute per-SNR accuracy
6. Distribute global model to clients

### Model Format

- **Model**: Pickled `sklearn.neighbors.KNeighborsClassifier`
- **Features**: Pickled numpy array (n_samples, 8)
- **Labels**: Pickled numpy array (n_samples,)

## License

MIT License - See LICENSE file for details

## Citation

Based on RadioML 2016.10a dataset and federated learning principles. ML approach adapted from `amc-rml2016a-updated.ipynb`.

## Support

For issues or questions, check:
1. Server logs for errors
2. Dashboard for system status
3. `diagnose_server.py` for diagnostics


## How Auto-Aggregation Works

The system uses an automatic aggregation mechanism that triggers model aggregation when a configurable threshold of client uploads is reached.

### Auto-Aggregation Process

**Configuration**:
- `auto_aggregation_enabled`: Enable/disable auto-aggregation (default: `true`)
- `auto_aggregation_threshold`: Number of client uploads required to trigger aggregation (default: `2`)

**Workflow**:

1. **Client Upload Tracking**
   - When a client uploads their model, the server tracks it in `clients_uploaded_this_round`
   - The `pending_uploads` counter increments
   - Example: Client 1 uploads → `pending_uploads = 1`

2. **Threshold Check**
   - After each upload, the server checks: `pending_uploads >= threshold`
   - If threshold is met, auto-aggregation triggers immediately
   - Example: Client 2 uploads → `pending_uploads = 2` → **Aggregation starts**

3. **Aggregation Execution** (runs in background thread)
   - Captures before-aggregation metrics
   - Collects training data from **all clients** in the registry
   - Merges features and labels from all clients
   - Trains global KNN model on combined dataset
   - Evaluates global model and captures after-aggregation metrics
   - Stores metrics history for dashboard
   - Saves global model to `./central/model_store/global_knn_model.pkl`

4. **State Reset**
   - After successful aggregation:
     - `clients_uploaded_this_round` is cleared to `[]`
     - `pending_uploads` is reset to `0`
     - `current_round` increments
     - `last_aggregation_time` is updated

5. **Continuous Cycle**
   - The process repeats for subsequent uploads
   - Each time the threshold is reached, aggregation triggers again

### Example Scenario: 5 Clients, Threshold = 2

```
Round 1:
  Client 1 uploads → pending_uploads = 1
  Client 2 uploads → pending_uploads = 2 → AGGREGATION TRIGGERED
  → Aggregates data from Client 1 + Client 2
  → State resets: pending_uploads = 0, round = 1

Round 2:
  Client 3 uploads → pending_uploads = 1
  Client 4 uploads → pending_uploads = 2 → AGGREGATION TRIGGERED
  → Aggregates data from Client 3 + Client 4
  → State resets: pending_uploads = 0, round = 2

Round 3:
  Client 5 uploads → pending_uploads = 1
  → Waits for one more client to reach threshold
```

### Key Behaviors

**Multiple Clients**:
- The system supports unlimited clients
- Aggregation uses **all available client data** from the registry, not just the ones that triggered it
- More clients = more training data = potentially better global model

**Threshold Flexibility**:
- Set threshold to `1` for immediate aggregation after each upload
- Set threshold to `N` to wait for N clients before aggregating
- Threshold can be changed in `central/config.json` without restarting

**Manual Aggregation**:
- You can still manually trigger aggregation via `POST /aggregate` endpoint
- Manual aggregation bypasses the threshold check
- Useful for testing or forcing aggregation with fewer clients

**Thread Safety**:
- Upload tracking uses thread locks to prevent race conditions
- Only one aggregation can run at a time
- Concurrent uploads are queued and processed safely

### Monitoring Auto-Aggregation

Check aggregation status via the dashboard or API:

```bash
# Get server status
curl http://localhost:8000/status

# Response includes:
{
  "upload_status": {
    "pending_uploads": 1,
    "threshold": 2,
    "ready_for_aggregation": false
  }
}
```

Dashboard displays:
- Current round number
- Number of clients uploaded this round
- Threshold setting
- Last aggregation timestamp
