# RadioML Dataset Instructions

## Required Dataset

This federated learning client requires the **RML2016.10a** dataset for radio signal modulation classification.

## How to Obtain the Dataset

1. Visit the RadioML dataset page on Kaggle or DeepSig:
   - Kaggle: https://www.kaggle.com/datasets/
   - DeepSig: https://www.deepsig.ai/datasets

2. Download the **RML2016.10a_dict.pkl** file

3. Place the downloaded file in this `data/` directory:
   ```
   data/RML2016.10a_dict.pkl
   ```

## Dataset Details

- **Format**: Python pickle file (.pkl)
- **Contents**: I/Q samples for 11 modulation types at various SNR levels
- **Modulation Types**: BPSK, QPSK, 8PSK, QAM16, QAM64, CPFSK, GFSK, PAM4, WBFM, AM-SSB, AM-DSB
- **SNR Range**: -20 dB to +18 dB (2 dB steps)
- **Samples per modulation/SNR**: 1000

## Usage

Once the dataset is placed in this directory, you can:
1. Launch the client application
2. Select the dataset file path in the Gradio UI
3. Extract features and begin training

## Note

The dataset file is not included in this repository due to its size and licensing. You must download it separately.
