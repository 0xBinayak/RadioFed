"""
Feature Extraction Module

This module converts I/Q samples from the RadioML dataset into
feature vectors for machine learning. Supports both traditional
16-dimensional features and AMC-specific 8-dimensional features
based on instantaneous amplitude and frequency.
"""

import numpy as np
from scipy import stats
from scipy.fft import fft
from typing import Tuple, Dict


def compute_instantaneous_amplitude(signal: np.ndarray) -> np.ndarray:
    """
    Compute instantaneous amplitude from complex signal.
    
    Args:
        signal: Complex-valued signal array
        
    Returns:
        Instantaneous amplitude (magnitude) of the signal
    """
    return np.abs(signal)


def compute_instantaneous_phase(signal: np.ndarray) -> np.ndarray:
    """
    Compute unwrapped instantaneous phase from complex signal.
    
    Args:
        signal: Complex-valued signal array
        
    Returns:
        Unwrapped instantaneous phase in radians
    """
    phase = np.angle(signal)
    unwrapped_phase = np.unwrap(phase)
    return unwrapped_phase


def compute_instantaneous_frequency(signal: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    Compute instantaneous frequency from complex signal.
    
    Args:
        signal: Complex-valued signal array
        fs: Sampling frequency (default: 128 for RML2016.10a dataset)
        
    Returns:
        Instantaneous frequency with same length as input (padded)
    """
    
    phase = compute_instantaneous_phase(signal)
    
    #freq = (1 / 2*pi) * d(phase)/dt = (1 / 2*pi) * d(phase)/dn * fs"""
    freq_diff = np.diff(phase) / (2 * np.pi) * fs
    
    
    instantaneous_freq = np.pad(freq_diff, (0, 1), mode='edge')
    
    return instantaneous_freq


def compute_statistical_features(data: np.ndarray, epsilon: float = 1e-9) -> Dict[str, float]:
    """
    Compute statistical features (mean, variance, skewness, kurtosis) from data.
    
    Uses epsilon-based stability for skewness and kurtosis calculations to handle
    edge cases with zero or near-zero standard deviation.
    
    Args:
        data: 1D array of values
        epsilon: Small value for numerical stability (default: 1e-9)
        
    Returns:
        Dictionary with keys: mean, variance, skewness, kurtosis (excess kurtosis)
    """
    mean = np.mean(data)
    variance = np.var(data)
    std = np.sqrt(variance)
    
    
    if std < epsilon:
        return {
            'mean': float(mean),
            'variance': float(variance),
            'skewness': 0.0,
            'kurtosis': 0.0  
        }
    
    
    centered = data - mean
    
    
    skewness = np.mean(centered ** 3) / (std ** 3 + epsilon)
    
    
    raw_kurtosis = np.mean(centered ** 4) / (std ** 4 + epsilon)
    excess_kurtosis = raw_kurtosis - 3.0
    
    return {
        'mean': float(mean),
        'variance': float(variance),
        'skewness': float(skewness),
        'kurtosis': float(excess_kurtosis)
    }


def extract_analog_features(signal: np.ndarray, fs: int = 128) -> np.ndarray:
    """
    Extract 8-dimensional feature vector for analog modulation classification.
    
    This function implements the feature extraction approach from the notebook
    for AMC using instantaneous amplitude and frequency statistics.
    
    Args:
        signal: Complex-valued signal array (I + jQ)
        fs: Sampling frequency (default: 128 for RML2016.10a dataset)
        
    Returns:
        8-dimensional feature vector as numpy array with the following features:
        [amp_mean, amp_variance, amp_skewness, amp_kurtosis,
         freq_mean, freq_variance, freq_skewness, freq_kurtosis]
    """
    
    amplitude = compute_instantaneous_amplitude(signal)
    
    frequency = compute_instantaneous_frequency(signal, fs=fs)
    
    
    amp_features = compute_statistical_features(amplitude)
    
    
    freq_features = compute_statistical_features(frequency)
    
    
    feature_vector = np.array([
        amp_features['mean'],
        amp_features['variance'],
        amp_features['skewness'],
        amp_features['kurtosis'],
        freq_features['mean'],
        freq_features['variance'],
        freq_features['skewness'],
        freq_features['kurtosis']
    ], dtype=np.float32)
    
   
    if np.any(np.isnan(feature_vector)) or np.any(np.isinf(feature_vector)):
        
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_vector


def extract_features_from_iq(iq_sample: np.ndarray, use_analog_features: bool = False) -> np.ndarray:
    """
    Extract feature vector from I/Q samples.
    
    Supports two modes:
    1. Traditional 16-dimensional features (default)
    2. Analog AMC 8-dimensional features (when use_analog_features=True)
    
    Traditional features include:
    - Statistical features (10): mean, std, variance, skewness, kurtosis for I and Q
    - Frequency domain features (4): FFT peak magnitude, FFT peak frequency, spectral centroid, spectral bandwidth
    - Time domain features (2): zero-crossing rate, energy
    
    Analog AMC features include:
    - Amplitude statistics (4): mean, variance, skewness, kurtosis
    - Frequency statistics (4): mean, variance, skewness, kurtosis
    
    Args:
        iq_sample: I/Q sample with shape (2, 128) where first row is I, second is Q
        use_analog_features: If True, extract 8D analog features; if False, extract 16D traditional features
        
    Returns:
        Feature vector of shape (8,) or (16,) depending on mode
    """
    
    if iq_sample.shape != (2, 128):
        raise ValueError(f"Expected shape (2, 128), got {iq_sample.shape}")
    
    
    signal = iq_sample[0, :] + 1j * iq_sample[1, :]
    
    
    if use_analog_features:
        return extract_analog_features(signal, fs=128)
    
    
    i_channel = iq_sample[0, :]
    q_channel = iq_sample[1, :]
    
    features = []
    
    
    features.append(np.mean(i_channel))
    features.append(np.std(i_channel))
    features.append(np.var(i_channel))
    i_skew = stats.skew(i_channel)
    features.append(0.0 if np.isnan(i_skew) else i_skew)
    i_kurt = stats.kurtosis(i_channel)
    features.append(0.0 if np.isnan(i_kurt) else i_kurt)
    
    
    features.append(np.mean(q_channel))
    features.append(np.std(q_channel))
    features.append(np.var(q_channel))
    q_skew = stats.skew(q_channel)
    features.append(0.0 if np.isnan(q_skew) else q_skew)
    q_kurt = stats.kurtosis(q_channel)
    features.append(0.0 if np.isnan(q_kurt) else q_kurt)
    
    
    complex_signal = i_channel + 1j * q_channel
    
    
    fft_result = fft(complex_signal)
    fft_magnitude = np.abs(fft_result)
    fft_freqs = np.fft.fftfreq(len(complex_signal))
    
    
    peak_magnitude = np.max(fft_magnitude)
    features.append(peak_magnitude)
    
    
    peak_idx = np.argmax(fft_magnitude)
    peak_frequency = np.abs(fft_freqs[peak_idx])
    features.append(peak_frequency)
    
    
    spectral_centroid = np.sum(fft_freqs[:len(fft_freqs)//2] * fft_magnitude[:len(fft_magnitude)//2]) / (np.sum(fft_magnitude[:len(fft_magnitude)//2]) + 1e-10)
    features.append(spectral_centroid)
    
   
    spectral_bandwidth = np.sqrt(np.sum(((fft_freqs[:len(fft_freqs)//2] - spectral_centroid) ** 2) * fft_magnitude[:len(fft_magnitude)//2]) / (np.sum(fft_magnitude[:len(fft_magnitude)//2]) + 1e-10))
    features.append(spectral_bandwidth)
    
    
    i_zero_crossings = np.sum(np.diff(np.sign(i_channel)) != 0) / len(i_channel)
    q_zero_crossings = np.sum(np.diff(np.sign(q_channel)) != 0) / len(q_channel)
    zero_crossing_rate = (i_zero_crossings + q_zero_crossings) / 2
    features.append(zero_crossing_rate)
    
    
    energy = np.sum(np.abs(complex_signal) ** 2) / len(complex_signal)
    features.append(energy)
    
    return np.array(features, dtype=np.float32)


def process_dataset(samples: np.ndarray, labels: np.ndarray, verbose: bool = True, use_analog_features: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features for an entire dataset of I/Q samples.
    
    Args:
        samples: Array of I/Q samples with shape (n_samples, 2, 128)
        labels: Array of labels with shape (n_samples,)
        verbose: Whether to print progress information
        use_analog_features: If True, extract 8D analog features; if False, extract 16D traditional features
        
    Returns:
        Tuple of (features, labels) where:
            - features: numpy array of shape (n_samples, 8) or (n_samples, 16)
            - labels: numpy array of shape (n_samples,) (unchanged)
    """
    n_samples = samples.shape[0]
    features_list = []
    feature_dim = 8 if use_analog_features else 16
    
    if verbose:
        feature_type = "analog AMC" if use_analog_features else "traditional"
        print(f"Extracting {feature_type} features from {n_samples} samples...")
    
    for i in range(n_samples):
        try:
            features = extract_features_from_iq(samples[i], use_analog_features=use_analog_features)
            features_list.append(features)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to extract features for sample {i}: {str(e)}")
            
            features_list.append(np.zeros(feature_dim, dtype=np.float32))
        
        
        if verbose and (i + 1) % 10000 == 0:
            print(f"Processed {i + 1}/{n_samples} samples...")
    
    features_array = np.array(features_list, dtype=np.float32)
    
    if verbose:
        print(f"Feature extraction complete. Shape: {features_array.shape}")
    
    return features_array, labels


def normalize_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features using z-score normalization.
    
    Args:
        features: Feature array of shape (n_samples, 16)
        
    Returns:
        Tuple of (normalized_features, mean, std) where:
            - normalized_features: Normalized feature array
            - mean: Mean values for each feature dimension
            - std: Standard deviation for each feature dimension
    """
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    
    
    std = np.where(std == 0, 1, std)
    
    normalized_features = (features - mean) / std
    
    return normalized_features, mean, std
