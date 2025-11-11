# Federated Learning for Analog Modulation Classification

A federated learning system for automatic modulation classification (AMC) using K-Nearest Neighbors on RadioML 2016.10a dataset. Supports distributed training across multiple clients with automatic aggregation and real-time monitoring..

## Features

- **KNN-Only Architecture**: Simplified ML approach using scikit-learn's KNeighborsClassifier
- **Feature Extraction**: 8 instantaneous features (amplitude + frequency statistics) from I/Q signals
- **Federated Aggregation**: Merges training data from multiple clients to create global model
- **Auto-Aggregation**: Automatic model aggregation when threshold is met (default: 2 clients)
- **Real-Time Dashboard**: Live monitoring with accuracy trends, confusion matrices, and performance metrics
- **Per-SNR Analysis**: Accuracy breakdown from -20 to 18 dB

## Architecture

### Client Side

**Feature Extraction** (`client/train.py`):
```python
def extract_analog_features(signal, fs=128):
    """ Extracts 8 features from I/Q signals:
        Amplitude: mean, variance, skewness, kurtosis
        Frequency: mean, variance, skewness, kurtosis"""
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


## Quick Start

### Install Dependencies

```bash
pip install uv
```
```bash
uv sync
```

###  Partition Dataset 
(for simulation)

#### Download unzip Dataset 
```bash
cd data
```
```bash
curl -L -o ./radioml2016-deepsigcom.zip \
  https://www.kaggle.com/api/v1/datasets/download/nolasthitnotomorrow/radioml2016-deepsigcom
````
```bash
unzip ./radioml2016-deepsigcom.zip -d ./
````


```bash
uv run  data/partition_dataset.py
```

Creates balanced partitions for federated learning from RadioML 2016.10a dataset.

###  Run Tests
```bash
uv run pytest tests
```

### Start Central Server

```bash
uv run  central/main.py
```

Server runs on `http://localhost:8000` with dashboard at `http://localhost:7860`

###  Start Clients

```bash
uv run client/main.py --port 7861 --auto-id
```
```bash
uv run client/main.py --port 7862 --auto-id
```
```bash
uv run client/main.py --port 7863 --auto-id


```

###  Train on Each Client

1. Open client UI (e.g., `http://localhost:7861`)
2. Click "Load Partition"
3. Click "Extract Features" (wait 1-3 minutes)
4. Click "Train Model" (auto-uploads when complete)

###  View Results

Open dashboard at `http://localhost:7860` to see:
- Real-time accuracy trends
- Confusion matrices
- Per-SNR performance
- Client status and metrics






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



## License

MIT License - See LICENSE file for details

## Citation

Based on RadioML 2016.10a dataset and federated learning principles. ML approach adapted from `amc-rml2016a-updated.