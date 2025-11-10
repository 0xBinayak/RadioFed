# Federated Learning AMC System

A comprehensive federated learning system for Automatic Modulation Classification (AMC) using traditional machine learning models (KNN and Decision Tree) with real-time dashboard monitoring and automatic weight upload.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [What's New](#whats-new)
4. [System Architecture](#system-architecture)
5. [Dataset Preparation](#dataset-preparation)
6. [Central Server Setup](#central-server-setup)
7. [Client Configuration](#client-configuration)
8. [Running Multi-Client Simulations](#running-multi-client-simulations)
9. [Dashboard Guide](#dashboard-guide)
10. [Troubleshooting](#troubleshooting)
11. [API Reference](#api-reference)

## Overview

This system implements federated learning for radio signal modulation classification using the RadioML 2016.10a dataset. The implementation is based on the `amc-rml2016a-updated.ipynb` notebook and provides a distributed training environment with automatic aggregation.

### Key Features

-**Auto-Upload**: Weights automatically upload after training (no manual button click needed)
-**Auto-Aggregation**: Automatic model aggregation when threshold is met (default: 2 clients)
-**Real-Time Dashboard**: Live updates every 2 seconds with current round number
-**Traditional ML Models**: Support for K-Nearest Neighbors (KNN) and Decision Tree classifiers
-**Historical Metrics**: Track model performance across training rounds
-**Per-SNR Analysis**: Accuracy breakdown from -20 to 18 dB
-**Confusion Matrices**: Visual representation of classification performance
-**Computation Metrics**: Training and inference time benchmarking

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, Linux, or macOS
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space for dataset and models

## What's New

### Recent Updates

**Auto-Upload Feature**
- Training now automatically uploads weights to the server
- No need to manually click "Upload Weights" button
- Visual feedback shows upload progress (X/Y clients)
- Status indicates when auto-aggregation will trigger

**Dashboard Improvements** 
- Current training round now updates correctly
- Historical trends show last 10 rounds
- Before/after aggregation comparison
- Real-time metrics refresh every 2 seconds

**How It Works**:
1. Train model → Weights auto-upload → Server tracks uploads
2. When threshold met (default: 2 clients) → Auto-aggregation triggers
3. Dashboard updates → Round increments → Metrics displayed


## Quick Start

### Installation
```bash
cd data
curl -L -o radioml2016-deepsigcom.zip https://www.kaggle.com/api/v1/datasets/download/nolasthitnotomorrow/radioml2016-deepsigcom
unzip radioml2016-deepsigcom.zip
cd ..

```

```bash
# Install dependencies using uv
uv sync

# Or using pip
pip install -r requirements.txt
```

### Running the System (3 Simple Steps)

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Partition Dataset                                  │
│  python data/partition_dataset.py --num-clients 3           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Start Central Server                               │
│  python central/main.py                                      │
│  → Dashboard opens at http://localhost:7860                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Start Clients & Train                              │
│  launch_3_clients.ps1  (or manually start 3 clients)        │
│  → Load partition → Extract features → Train model          │
│  → Weights auto-upload  → Aggregation triggers          │
└─────────────────────────────────────────────────────────────┘
```

#### 1. Prepare Dataset Partitions

Create balanced partitions for 3 clients:

```bash
uv run  python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 3
```

This creates:
- `data/partitions/client_0.pkl`
- `data/partitions/client_1.pkl`
- `data/partitions/client_2.pkl`

#### 2. Start Central Server

```bash
uv run python central/main.py
```

The server automatically:
- Starts FastAPI backend on port 8000
- Launches Gradio dashboard on port 7860
- Opens dashboard in your browser

Access at: **http://localhost:7860**

#### 3. Start Clients and Train

**Windows (Easy Way):**
```bash
launch_3_clients.ps1
```

**Manual (Any OS):**
```bash
# Terminal 1
uv run python client/main.py --partition-id 0 --port 7861 --auto-id

# Terminal 2
uv run python client/main.py --partition-id 1 --port 7862 --auto-id

# Terminal 3
uv run python client/main.py --partition-id 2 --port 7863 --auto-id
```

**For each client:**
1. Enter partition ID (0, 1, or 2)
2. Click "Load Partition"
3. Click "Extract Features" (wait ~1-3 minutes)
4. Select model type (KNN or Decision Tree)
5. **Ensure "Auto-upload after training" is checked ✓** (enabled by default)
6. Click "Train Model"
7. **Watch weights auto-upload!** 

**What You'll See**:
```
 KNN model uploaded successfully!
 Upload progress: 2/2 clients
 Auto-aggregation will trigger automatically!
```

**Aggregation happens automatically** when threshold is met (default: 2 clients)!

### What Happens Automatically

With auto-upload enabled (default):
1. Client trains model
2. **Weights automatically upload to server** (no button click needed)
3. Server tracks uploads and shows progress (X/Y clients)
4. Server triggers aggregation when threshold met
5. Global model is updated
6. Dashboard displays metrics and increments round number
7. Clients can download global model for next round
uv run python client/main.py --partition-id 1 --port 7862

# Terminal 3
uv run python client/main.py --partition-id 2 --port 7863
```

**For each client:**
1. Enter partition ID (0, 1, or 2)
2. Click "Load Partition"
3. Click "Extract Features" (wait ~1-3 minutes)
4. Select model type (KNN or Decision Tree)
5. Click "Train Model" (auto-uploads when complete)

**Aggregation happens automatically** when threshold is met (default: 2 clients)!

### What Happens Automatically

With auto-upload enabled (default):
1. Client trains model
2. Weights automatically upload to server
3. Server triggers aggregation when threshold met
4. Global model is updated
5. Dashboard displays metrics
6. Clients can download global model


## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Central Server Layer                      │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  central/server.py                                     │ │
│  │  - FastAPI REST API (port 8000)                        │ │
│  │  - Upload endpoint with auto-aggregation trigger       │ │
│  │  - Client registration and tracking                    │ │
│  │  - Model aggregation coordination                      │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  central/dashboard.py                                  │ │
│  │  - Gradio web interface (port 7860)                    │ │
│  │  - Real-time metrics visualization                     │ │
│  │  - Historical trends tracking                          │ │
│  │  - Client monitoring                                   │ │
│  └────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  central/aggregator.py                                 │ │
│  │  - FedAvg aggregation algorithm                        │ │
│  │  - Model evaluation and metrics collection             │ │
│  │  - Per-SNR accuracy analysis                           │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ↕
┌─────────────────────────────────────────────────────────────┐
│                      Client Nodes                            │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  client/main.py                                        │ │
│  │  - Gradio interface for local training                 │ │
│  │  - Dataset loading and partitioning                    │ │
│  │  - Feature extraction                                  │ │
│  │  - Model training (KNN/Decision Tree)                  │ │
│  │  - Weight upload to server                             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Auto-Aggregation Flow

```
Client uploads weights
        ↓
Track client upload
        ↓
Increment pending_uploads counter
        ↓
Check: pending_uploads >= threshold?
        ↓
    Yes → Trigger aggregation
        ↓
    Capture before-aggregation metrics
        ↓
    Perform FedAvg aggregation
        ↓
    Evaluate global model
        ↓
    Compute after-aggregation metrics
        ↓
    Store metrics in history
        ↓
    Update dashboard state
        ↓
    Reset pending_uploads = 0
        ↓
    Broadcast global model availability
```

### Project Structure

```
project/
├── central/                    # Central server components
│   ├── main.py                # Server entry point with auto-start
│   ├── server.py              # FastAPI REST API
│   ├── dashboard.py           # Gradio dashboard interface
│   ├── aggregator.py          # FedAvg aggregation logic
│   ├── state.py               # State management and metrics storage
│   ├── model.py               # Model definitions
│   ├── utils.py               # Utility functions
│   ├── config.json            # Server configuration
│   └── model_store/           # Stored models and weights
├── client/                     # Client node components
│   ├── main.py                # Client entry point
│   ├── train.py               # Local training logic
│   ├── sync.py                # Server communication
│   ├── dataset_loader.py      # Dataset loading and partitioning
│   ├── feature_extract.py     # Feature extraction
│   ├── model.py               # Model definitions
│   ├── config.json            # Client configuration
│   └── local/                 # Local models and metrics
├── data/                       # Dataset and partitions
│   ├── partition_dataset.py   # Dataset partitioning script
│   ├── RML2016.10a_dict.pkl   # RadioML dataset
│   └── partitions/            # Client partition files
├── tests/                      # Test suite
│   ├── test_aggregator.py
│   ├── test_dashboard_metrics.py
│   ├── test_multi_client_simulation.py
│   └── test_e2e_workflow.py
└── README.md                   # This file
```


## Dataset Preparation

### Overview

The dataset partitioning script prepares the RadioML 2016.10a dataset for federated learning by splitting it into balanced, non-overlapping partitions. This is a **one-time operation** performed before training.

### Why External Partitioning?

- **Efficiency**: Partition once, reuse across multiple experiments
- **Reproducibility**: Same partitions for consistent results
- **Simplicity**: Clients just load their assigned partition
- **Validation**: Pre-validate data splits before training

### Basic Usage

Create 3 partitions for 3 clients:

```bash
python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 3
```

Expected output:
```
✓ Dataset loaded successfully
✓ Filtered for analog modulations: ['AM', 'FM']
✓ Created 3 balanced partitions
✓ Validation passed: 3 valid partitions

======================================================================
PARTITION STATISTICS
======================================================================

Total Partitions: 3
Total Samples: 24,000
Output Directory: data/partitions

Partition    Samples      Modulations          SNR Range
----------------------------------------------------------------------
client_0     8,000        AM, FM               -20 to 18 dB
client_1     8,000        AM, FM               -20 to 18 dB
client_2     8,000        AM, FM               -20 to 18 dB
```

### Command-Line Parameters

**Required:**
- `--input`: Path to RadioML 2016.10a pickle file
- `--num-clients`: Number of client partitions to create

**Optional:**
- `--output`: Output directory (default: `data/partitions`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--no-balance`: Disable class balancing (not recommended)

### Examples

**Custom output directory:**
```bash
python data/partition_dataset.py \
  --input data/RML2016.10a_dict.pkl \
  --num-clients 5 \
  --output custom_partitions
```

**Reproducible partitions:**
```bash
python data/partition_dataset.py \
  --input data/RML2016.10a_dict.pkl \
  --num-clients 3 \
  --seed 123
```

### What the Script Does

1. **Load Dataset**: Loads RadioML 2016.10a pickle file
2. **Filter Modulations**: Extracts analog modulations (AM-DSB, AM-SSB → AM; WBFM → FM)
3. **Create Partitions**: Splits into equal, non-overlapping partitions
4. **Validate**: Ensures balanced distribution and no overlap
5. **Save Files**: Saves each partition as `client_N.pkl`
6. **Display Statistics**: Shows detailed partition information

### Partition Balance

The script ensures:
- **Equal samples**: Each partition gets ~same number of samples
- **Class distribution**: AM and FM samples distributed evenly
- **SNR coverage**: All SNR levels (-20 to 18 dB) in each partition
- **Non-overlapping**: No sample appears in multiple partitions

Balance metrics:
- **99%+**: Excellent balance (typical)
- **95-99%**: Good balance
- **<95%**: May indicate issues


## Central Server Setup

### Starting the Server

```bash
python central/main.py
```

The server automatically:
1. Starts FastAPI backend on `http://localhost:8000`
2. Launches Gradio dashboard on `http://localhost:7860`
3. Opens dashboard in your browser

Expected output:
```
======================================================================
Federated Learning Central Server - Auto-Start Mode
======================================================================
Starting FastAPI server on 127.0.0.1:8000...
✓ FastAPI server is ready at http://localhost:8000
======================================================================
✓ FastAPI Server: http://localhost:8000
  - API Documentation: http://localhost:8000/docs
  - Health Check: http://localhost:8000/health
======================================================================
✓ Dashboard: http://localhost:7860
======================================================================
Server is ready! Dashboard will open automatically.
Press Ctrl+C to stop the server.
```

### Server Configuration

Location: `central/config.json`

```json
{
  "model_save_path": "./central/model_store/global_model.pth",
  "host": "127.0.0.1",
  "port": 8000,
  "log_level": "INFO",
  "auto_aggregation_enabled": true,
  "auto_aggregation_threshold": 2
}
```

**Parameters:**
- `model_save_path`: Where to save global models
- `host`: Server host address (use `0.0.0.0` for remote access)
- `port`: FastAPI server port
- `log_level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `auto_aggregation_enabled`: Enable/disable auto-aggregation
- `auto_aggregation_threshold`: Minimum clients before auto-aggregation

### Stopping the Server

Press `Ctrl+C` in the terminal:
```
^C
Shutdown requested by user
Stopping server...
```

### Health Check

Verify server is running:
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-11-10T14:32:15.123456"
}
```


## Client Configuration

### Starting a Client

**Basic usage:**
```bash
python client/main.py --partition-id 0 --port 7861
```

**With auto-generated ID:**
```bash
python client/main.py --partition-id 0 --auto-id
```

### Command-Line Arguments

| Option | Description | Example |
|--------|-------------|---------|
| `--partition-id` | Partition ID to load (0, 1, 2, ...) | `--partition-id 0` |
| `--port` | Gradio UI port | `--port 7862` |
| `--client-id` | Custom client ID | `--client-id my_client` |
| `--auto-id` | Auto-generate random client ID | `--auto-id` |
| `--server-url` | Central server URL | `--server-url http://192.168.1.100:8000` |

### Client Workflow

1. **Load Partition**
   - Enter partition ID (0, 1, 2, etc.)
   - Click "Load Partition"
   - Verify dataset info appears

2. **Extract Features**
   - Click "Extract Features"
   - Wait for progress bar (~1-3 minutes for 8,000 samples)
   - Features: 8-dimensional vectors (amplitude + frequency statistics)

3. **Select Model**
   - Choose "KNN" or "Decision Tree"
   - Adjust training parameters if needed

4. **Train Model**
   - Click "Train Model"
   - Monitor training progress
   - Review training metrics

5. **Upload Weights** (Automatic if auto-upload enabled)
   - Weights sent to central server
   - Server may trigger aggregation

6. **Download Global Model** (Optional)
   - Click "Download Global Model"
   - Use as starting point for next round

### Feature Extraction

The client extracts **8-dimensional feature vectors** from IQ samples:

**Amplitude Features (4 dimensions):**
1. Mean instantaneous amplitude
2. Variance of amplitude
3. Skewness of amplitude
4. Kurtosis of amplitude

**Frequency Features (4 dimensions):**
5. Mean instantaneous frequency
6. Variance of frequency
7. Skewness of frequency
8. Kurtosis of frequency

**Typical Duration**: 1-3 minutes for 8,000 samples

### Model Selection

**K-Nearest Neighbors (KNN):**
- Instance-based learning
- Fast training (~1-3 seconds)
- Slower inference (distance calculations)
- Good for small to medium datasets

**Decision Tree (DT):**
- Tree-based classification
- Fast training (~0.5-2 seconds)
- Very fast inference
- Interpretable decisions

### Client Configuration File

Location: `client/config.json`

```json
{
  "client_id": "client_001",
  "server_url": "http://localhost:8000",
  "local_model_path": "./client/local/local_model.pth",
  "training": {
    "epochs": 10,
    "batch_size": 32,
    "learning_rate": 0.001
  }
}
```


## Running Multi-Client Simulations

### Quick Launch (Windows)

```powershell
launch_3_clients.ps1
```

This automatically launches 3 clients with random unique IDs.

### Manual Launch (Any OS)

**Terminal 1 - Client 0:**
```bash
python client/main.py --partition-id 0 --port 7861 --auto-id
```

**Terminal 2 - Client 1:**
```bash
python client/main.py --partition-id 1 --port 7862 --auto-id
```

**Terminal 3 - Client 2:**
```bash
python client/main.py --partition-id 2 --port 7863 --auto-id
```

### Complete Workflow Example

#### Step 1: Start Central Server
```bash
python central/main.py
```
Access at: http://localhost:7860

#### Step 2: Start Multiple Clients

Use the launch script or start manually (see above).

#### Step 3: Train on Each Client

For each client interface:

1. **Load Dataset**
   - Enter partition ID (0, 1, or 2)
   - Click "Load Partition"
   - Verify dataset info displays

2. **Extract Features**
   - Click "Extract Features"
   - Wait for completion (~30 seconds - 3 minutes)

3. **Train Model**
   - Select Model Type: "KNN" or "Decision Tree"
   - Enable "Auto-upload after training" ✓
   - Click "Train Model"
   - Wait for training and upload (~1-2 minutes)

#### Step 4: Monitor Aggregation

In the central server dashboard (http://localhost:7860):

1. **Check System Status**:
   - Connected clients should show 3
   - Training round increments after aggregation

2. **Monitor Client Status**:
   - All 3 clients should appear
   - Status shows "✓ Uploaded" when ready

3. **View Metrics**:
   - Training metrics update automatically
   - Confusion matrices display
   - Accuracy vs SNR plot shows curves

### Automated Simulation Script

For testing, use the automated simulation:

```bash
# Run with 3 clients using KNN
python tests/test_multi_client_simulation.py --num-clients 3 --model-type knn

# Run with Decision Tree
python tests/test_multi_client_simulation.py --num-clients 3 --model-type dt

# Run with 5 clients
python tests/test_multi_client_simulation.py --num-clients 5 --model-type knn
```

The script automatically:
1. Verifies dataset partitions exist
2. Checks central server is running
3. Trains models on all clients
4. Uploads weights to server
5. Verifies aggregation works
6. Checks dashboard metrics

### Performance Benchmarks

**Typical Timings (3 clients, 8,000 samples each):**

| Phase | Duration | Notes |
|-------|----------|-------|
| Dataset Partitioning | 30-60s | One-time operation |
| Server Startup | 5-10s | Auto-start sequence |
| Client Startup | 5s per client | Can run in parallel |
| Partition Loading | 2-5s per client | Fast pickle load |
| Feature Extraction | 1-3 min per client | Most time-consuming |
| Model Training (KNN) | 30-60s per client | Fast training |
| Model Training (DT) | 20-40s per client | Faster than KNN |
| Weight Upload | 1-2s per client | Network dependent |
| Aggregation | 2-5s | Automatic |
| **Total (first round)** | **10-15 minutes** | Including all clients |


## Dashboard Guide

### Accessing the Dashboard

Open your browser to: **http://localhost:7860**

The dashboard automatically opens when you start the central server.

### Dashboard Sections

#### 1. System Status

**Location**: Top of dashboard

**Displays**:
- Server status (🟢 Running / 🔴 Stopped)
- Connected clients count and IDs
- Current training round number
- Last aggregation timestamp
- Auto-aggregation threshold

**Auto-Refresh**: Updates every 2 seconds

**Example**:
```
🟢 Server is running
Connected Clients: 3 (client_0, client_1, client_2)
Training Round: 5
Last Aggregation: 2024-11-10 14:32:15
Auto-aggregation: Enabled (threshold: 2)
```

#### 2. Client Monitoring

**Displays**: Table with client information

| Column | Description |
|--------|-------------|
| Client ID | Unique identifier |
| Status | ✓ Uploaded, ⏳ Training, Idle |
| Last Upload | Timestamp of last weight upload |
| Sample Count | Number of training samples |

**Auto-Refresh**: Updates every 2 seconds

#### 3. Training Progress (Historical Trends)

**Displays**: Line plot showing accuracy trends over last 10 rounds

**Features**:
- Before/after aggregation accuracy for KNN and DT
- Round number on x-axis
- Accuracy percentage on y-axis
- Legend for model identification

**Use Case**: Track model improvement across federated learning rounds

#### 4. Latest Aggregation Results

**Displays**: Before/after comparison table

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| KNN Accuracy | 72.00% | 85.00% | +13.00% |
| DT Accuracy | 68.00% | 82.00% | +14.00% |

**Use Case**: Evaluate impact of federated aggregation

#### 5. Confusion Matrices

**Displays**: Side-by-side heatmaps for KNN and Decision Tree

**Features**:
- 2x2 matrix for AM/FM classification
- Blue color scale (darker = more predictions)
- Annotations showing actual counts
- True Label (vertical) vs Predicted Label (horizontal)

**Interpretation**:
- Diagonal: Correct predictions
- Off-diagonal: Misclassifications

#### 6. Accuracy vs SNR

**Displays**: Line plot comparing model performance across SNR levels

**Three Curves**:
1. Baseline (black dashed): Random guess performance
2. Decision Tree (blue solid): DT model accuracy
3. KNN (red solid): KNN model accuracy

**Features**:
- X-axis: SNR in dB (-20 to 18)
- Y-axis: Accuracy percentage (0-105%)
- Grid for easy reading

**Expected Pattern**:
- Low SNR: All models near baseline (noisy signals)
- High SNR: ML models significantly outperform baseline

#### 7. Computation Complexity

**Displays**: Comparison table of computational efficiency

| Method | Training Time (s) | Inference Time (ms/sample) |
|--------|------------------|---------------------------|
| Decision Tree | 2.345 | 0.456 |
| KNN | 1.234 | 1.234 |

**Use Case**: Compare model efficiency for deployment decisions

### Auto-Refresh

- Dashboard updates every 2 seconds
- Fetches latest aggregation results
- Updates all visualizations automatically
- No manual refresh needed


## Troubleshooting

### Common Issues and Solutions

#### Auto-Upload Not Working

**Symptoms**: Weights don't upload automatically after training

**Solutions**:
1. **Check the checkbox**: Ensure "Auto-upload after training" is ✓ checked in client UI
2. **Verify server is running**: Test with `curl http://localhost:8000/health`
3. **Check client logs**: Look for upload errors in terminal
4. **Verify model trained**: Ensure training completed successfully

**Expected Behavior**:
``` 
KNN model uploaded successfully!
Upload progress: 1/2 clients
Waiting for more clients...
```

#### Dashboard Not Updating

**Symptoms**: Dashboard shows "Training Round: 0" or old metrics

**Solutions**:
1. **Hard refresh browser**: Windows: `Ctrl + F5`, Mac: `Cmd + Shift + R`
2. **Wait for aggregation**: Dashboard updates after aggregation completes
3. **Check server logs**: Look for "Auto-aggregation workflow completed"
4. **Verify clients uploaded**: Check server status shows uploaded clients

**Expected Behavior**:
- Training Round increments after each aggregation (0 → 1 → 2...)
- Dashboard refreshes every 2 seconds automatically
- Historical trends show last 10 rounds

#### Aggregation Not Triggering

**Symptoms**: Clients upload but aggregation doesn't start

**Solutions**:
1. **Check threshold**: Verify `auto_aggregation_threshold` in `central/config.json`
2. **Verify uploads**: Ensure enough clients have uploaded (check server logs)
3. **Check if enabled**: Ensure `auto_aggregation_enabled: true` in config
4. **Review server logs**: Look for "Auto-aggregation threshold reached"

**Expected Server Logs**:
```
INFO - Upload tracked for client_abc123: 2/2 clients uploaded
INFO - Auto-aggregation threshold reached: 2/2 clients
INFO - Auto-aggregation triggered in background thread
INFO - KNN aggregation completed: 2 clients, accuracy=0.8528
```

#### Port Already in Use

**Error**:
```
✗ ERROR: Port 8000 is already in use
```

**Solutions**:

1. **Find and stop the conflicting process**:
   ```bash
   # Windows
   netstat -ano | findstr :8000
   taskkill /PID <PID> /F
   
   # Linux/Mac
   lsof -i :8000
   kill -9 <PID>
   ```

2. **Use a different port**: Edit `central/config.json` and change the `port` value

3. **Check for other instances**: Make sure you don't have another server running

#### Dataset/Partition Not Found

**Error**:
```
✗ Partition file not found: data/partitions/client_0.pkl
```

**Solution**: Run the partitioning script first:
```bash
python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 3
```

#### Client Cannot Connect to Server

**Error**:
```
 Error: Cannot connect to server at http://localhost:8000
```

**Solutions**:

1. **Start central server first**:
   ```bash
   python central/main.py
   ```

2. **Check server URL**: Verify `server_url` in client config matches actual server

3. **Test server health**:
   ```bash
   curl http://localhost:8000/health
   ```

4. **Check firewall**: Ensure port 8000 is not blocked

#### Feature Extraction Stuck

**Symptoms**: Progress bar frozen or taking too long

**Solutions**:

1. **Be patient**: 1-3 minutes is normal for 8,000 samples
2. **Check system memory**: Ensure sufficient RAM available
3. **Reduce dataset size**: Use fewer partitions if needed
4. **Restart client**: If truly stuck, restart and try again

#### Training Fails

**Error**:
```
 Training failed: ...
```

**Solutions**:

1. **Extract features first**: Must complete feature extraction before training
2. **Verify model selection**: Ensure model type is selected
3. **Check training parameters**: Verify epochs, batch size are reasonable
4. **Review client logs**: Check terminal output for specific errors

#### Upload Fails

**Error**:
```
 Failed to upload weights: 500 - Internal Server Error
```

**Solutions**:

1. **Check server connection**: Ensure server is running and accessible
2. **Verify training completed**: Must train model before uploading
3. **Check server logs**: Review central server terminal for error messages
4. **Try manual upload**: Disable auto-upload and use manual button

#### Dashboard Shows No Data

**Symptoms**: Dashboard shows zeros or empty plots

**Solutions**:

1. **Ensure aggregation has run**: Metrics update after aggregation completes
2. **Check client training**: Clients must complete training and upload weights
3. **Trigger aggregation manually**:
   ```bash
   curl -X POST "http://localhost:8000/aggregate?model_type=knn"
   ```
4. **Hard refresh browser**: Windows: `Ctrl + F5`, Mac: `Cmd + Shift + R`
5. **Restart server**: Sometimes a fresh start resolves state issues

#### Aggregation Not Working

**Error**:
```
✗ No knn models available for aggregation
```

**Solutions**:

1. **Check model type**: Ensure clients trained with correct model type (KNN or DT)
2. **Verify uploads**: Check that clients successfully uploaded weights
3. **Check server status**:
   ```bash
   curl http://localhost:8000/status
   ```
4. **Restart server**: Restart to load latest code changes

### Configuration Options

#### Auto-Aggregation Settings

Edit `central/config.json`:

```json
{
  "auto_aggregation_enabled": true,
  "auto_aggregation_threshold": 2
}
```

**Parameters**:
- `auto_aggregation_enabled`: Set to `false` to disable auto-aggregation (requires manual trigger)
- `auto_aggregation_threshold`: Number of clients required to trigger aggregation (default: 2)

**Examples**:
```json
// Require 3 clients before aggregating
{"auto_aggregation_threshold": 3}

// Disable auto-aggregation (manual only)
{"auto_aggregation_enabled": false}
```

#### Client Auto-Upload Settings

**In Client UI**:
- "Auto-upload after training" checkbox (enabled by default)
- Can be toggled before each training session

**To disable**: Uncheck the box before clicking "Train Model"

### Best Practices

1. **Start server before clients**: Always start the central server first
2. **Partition before training**: Create partitions before starting experiments
3. **Use unique client IDs**: Avoid ID conflicts with `--auto-id` flag
4. **Enable auto-upload**: Keep checkbox enabled for seamless workflow
5. **Monitor dashboard**: Keep dashboard open at http://localhost:7860
6. **Check logs**: Review terminal output for upload/aggregation status
7. **Wait for aggregation**: Let auto-aggregation complete before starting new round
8. **Clean shutdown**: Use Ctrl+C to stop, don't force kill processes


## API Reference

### Base URL

```
http://localhost:8000
```

### Endpoints

#### Health Check

```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-11-10T14:32:15.123456"
}
```

#### Get Server Status

```http
GET /status
```

**Response**:
```json
{
  "connected_clients": 3,
  "total_samples": 24000,
  "current_round": 5,
  "last_aggregation": "2024-11-10T14:32:15",
  "auto_aggregation_enabled": true,
  "auto_aggregation_threshold": 2,
  "pending_uploads": 0,
  "clients": [
    {
      "client_id": "client_0",
      "status": "uploaded",
      "last_upload": "2024-11-10T14:30:00",
      "n_samples": 8000,
      "model_type": "knn"
    }
  ]
}
```

#### Upload Weights

```http
POST /upload_weights
Content-Type: multipart/form-data

Parameters:
- client_id: string (required)
- weights_file: file (required)
- n_samples: integer (required)
- model_type: string (required) - "knn", "dt", or "neural"
```

**Response**:
```json
{
  "status": "success",
  "message": "Weights uploaded successfully",
  "client_id": "client_0",
  "aggregation_triggered": true
}
```

#### Trigger Aggregation

```http
POST /aggregate?model_type=knn
```

**Query Parameters**:
- `model_type`: "knn", "dt", or "neural" (required)

**Response**:
```json
{
  "status": "success",
  "model_type": "knn",
  "num_clients": 3,
  "total_samples": 24000,
  "accuracy": 0.8528,
  "training_time": 0.005,
  "inference_time_ms": 0.002,
  "per_snr_accuracy": {
    "-20": 0.65,
    "-18": 0.70,
    "0": 0.85,
    "18": 0.96
  },
  "confusion_matrix": [[45, 5], [3, 47]],
  "timestamp": "2024-11-10T14:32:15",
  "message": "KNN aggregation completed successfully"
}
```

#### Download Global Model

```http
GET /download_model?model_type=knn
```

**Query Parameters**:
- `model_type`: "knn", "dt", or "neural" (required)

**Response**: Binary file (model weights)

#### Get Metrics History

```http
GET /metrics_history?last_n=10
```

**Query Parameters**:
- `last_n`: Number of recent rounds to retrieve (default: 10)

**Response**:
```json
{
  "rounds": [
    {
      "round": 1,
      "timestamp": "2024-11-10T14:30:00",
      "num_clients": 3,
      "before": {
        "knn_accuracy": 0.72,
        "dt_accuracy": 0.68
      },
      "after": {
        "knn_accuracy": 0.85,
        "dt_accuracy": 0.82
      },
      "improvement": {
        "knn": 0.13,
        "dt": 0.14
      }
    }
  ]
}
```

### Configuration Options

#### Auto-Aggregation Settings

Edit `central/config.json`:

```json
{
  "auto_aggregation_enabled": true,
  "auto_aggregation_threshold": 2
}
```

**Parameters**:
- `auto_aggregation_enabled`: Enable/disable automatic aggregation (boolean)
- `auto_aggregation_threshold`: Minimum number of clients required to trigger aggregation (integer, default: 2)

**Behavior**:
- When `enabled=true` and `pending_uploads >= threshold`, aggregation triggers automatically
- When `enabled=false`, aggregation must be triggered manually via API
- Setting `threshold=0` effectively disables auto-aggregation

#### Server Settings

```json
{
  "host": "127.0.0.1",
  "port": 8000,
  "log_level": "INFO",
  "model_save_path": "./central/model_store/global_model.pth"
}
```

**Parameters**:
- `host`: Server host address (use "0.0.0.0" for remote access)
- `port`: FastAPI server port
- `log_level`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `model_save_path`: Directory for storing global models

### Interactive API Documentation

When the server is running, access interactive API documentation at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

These provide:
- Complete endpoint documentation
- Request/response schemas
- Interactive testing interface
- Example requests and responses


## Additional Information

### Testing

Run the test suite to verify functionality:

```bash
# Run all tests
uv run pytest

# Run specific test
uv run python tests/test_dashboard_metrics.py
uv run python tests/test_multi_client_simulation.py
uv run python tests/test_e2e_workflow.py
```

### Performance Tips

1. **Feature extraction**: Most time-consuming step, be patient
2. **Batch size**: Larger batches = faster training (if memory allows)
3. **Model choice**: Decision Tree is faster than KNN for inference
4. **Local network**: Use local server for best performance
5. **Resource monitoring**: Watch CPU and memory during training

### Advanced Usage

#### Running on Remote Server

To make the server accessible from other machines:

1. Edit `central/config.json`:
   ```json
   {
     "host": "0.0.0.0",
     "port": 8000
   }
   ```

2. Update firewall rules to allow ports 8000 and 7860

3. Clients connect using server IP:
   ```bash
   python client/main.py --server-url http://192.168.1.100:8000
   ```

**Security Note**: Only expose the server on trusted networks. Consider adding authentication for production use.

#### Multiple Experiments

To run multiple experiments simultaneously:

1. Use different ports for each experiment
2. Use separate model storage directories
3. Clients specify which server to connect to

### Success Indicators

You know it's working when:
-Central server shows "🟢 Server is running"
-Clients can load partitions and see dataset info
-Feature extraction completes with progress updates
-Training shows accuracy metrics
-Dashboard shows all clients in monitoring section
-Aggregation completes with metrics
-Confusion matrices display heatmaps
-Accuracy vs SNR plot shows curves
-Global model can be downloaded

### Contributing

This project is part of a federated learning research initiative. For questions or contributions, please refer to the project documentation in `.kiro/specs/`.

### License

[Add your license information here]

### Acknowledgments

- RadioML dataset: https://www.deepsig.ai/datasets
- Built with FastAPI, Gradio, scikit-learn, and PyTorch

---

## Quick Reference Card

### Essential Commands

```bash
# Start server
uv run python central/main.py

# Start client with auto-generated ID
uv run python client/main.py --partition-id 0 --port 7861 --auto-id

# Partition dataset
uv run python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 3
```

### Key URLs

- **Dashboard**: http://localhost:7860
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### Default Settings

- **Auto-aggregation threshold**: 2 clients
- **Auto-upload**: Enabled by default
- **Dashboard refresh**: Every 2 seconds
- **Model types**: KNN, Decision Tree
- **SNR range**: -20 to 18 dB (2 dB steps)

### Training Workflow

1. Load partition → 2. Extract features → 3. Train model → 4. Auto-upload  → 5. Auto-aggregate 

### What to Watch

**Client Status**:
```
 Model uploaded successfully!
 Upload progress: 2/2 clients
 Auto-aggregation will trigger automatically!
```

**Server Logs**:
```
INFO - Auto-aggregation threshold reached: 2/2 clients
INFO - Auto-aggregation triggered in background thread
INFO - KNN aggregation completed: 2 clients, accuracy=0.8528
```

**Dashboard**:
- Training Round: Increments after each aggregation
- Historical Trends: Shows last 10 rounds
- Confusion Matrices: Updates for KNN and DT
- Accuracy vs SNR: Performance curves

---

**Happy federated learning!** 

Based on `amc-rml2016a-updated.ipynb` | RadioML 2016.10a Dataset | Analog Modulation Classification (AM/FM)
