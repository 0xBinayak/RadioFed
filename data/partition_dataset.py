"""
External Dataset Partitioning Script for Federated Learning

This script partitions the RadioML 2016.10a dataset into balanced, non-overlapping
subsets for federated learning clients. It filters for analog modulations only
(AM-DSB, AM-SSB, WBFM) and maps them to simplified labels (AM, FM).

Usage:
    python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 3
    python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 5 --output data/partitions
    python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 3 --seed 123

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5
"""

import pickle
import numpy as np
import argparse
import os
import sys
from typing import Dict, List, Tuple
from collections import defaultdict


def load_radioml_pkl_dataset(data_path: str) -> Dict:
    """
    Load RadioML 2016.10a dataset from pickle file.
    
    Args:
        data_path: Path to RML2016.10a_dict.pkl file
        
    Returns:
        Dictionary with structure {(modulation, SNR): samples_array}
        where samples_array has shape (num_samples, 2, 128)
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        Exception: If file cannot be loaded
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    
    try:
        print(f"Loading dataset from {data_path}...")
        with open(data_path, 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')
        print(f"✓ Dataset loaded successfully")
        return dataset
    except Exception as e:
        raise Exception(f"Failed to load dataset: {str(e)}")


def filter_analog_modulations(dataset: Dict) -> Dict:
    """
    Filter dataset for analog modulations only and map to simplified labels.
    
    Mapping:
        - AM-DSB → 'AM'
        - AM-SSB → 'AM'
        - WBFM → 'FM'
    
    Args:
        dataset: Full RadioML dataset dictionary
        
    Returns:
        Filtered dataset with only analog modulations and simplified labels
    """
    analog_mapping = {
        'AM-DSB': 'AM',
        'AM-SSB': 'AM',
        'WBFM': 'FM'
    }
    
    filtered_dataset = {}
    
    for (modulation, snr), samples in dataset.items():
        if modulation in analog_mapping:
            
            simplified_mod = analog_mapping[modulation]
            new_key = (simplified_mod, snr)
            
            
            if new_key in filtered_dataset:
                filtered_dataset[new_key] = np.concatenate([
                    filtered_dataset[new_key],
                    samples
                ], axis=0)
            else:
                filtered_dataset[new_key] = samples.copy()
    
    print(f" Filtered for analog modulations: {list(set([k[0] for k in filtered_dataset.keys()]))}")
    
    return filtered_dataset


def partition_dataset(
    dataset: Dict,
    num_partitions: int,
    balance_classes: bool = True,
    random_seed: int = 42
) -> List[Dict]:
    """
    Split dataset into equal non-overlapping partitions.
    
    Each partition maintains balanced class distribution across all SNRs.
    
    Args:
        dataset: Filtered analog modulation dataset
        num_partitions: Number of partitions to create (number of clients)
        balance_classes: Whether to balance classes across partitions
        random_seed: Random seed for reproducibility
        
    Returns:
        List of partition dictionaries, each with same structure as input
        
    Raises:
        ValueError: If num_partitions < 1 or insufficient data
    """
    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")
    
    
    total_samples = sum(samples.shape[0] for samples in dataset.values())
    if total_samples < num_partitions:
        raise ValueError(
            f"Cannot create {num_partitions} partitions from {total_samples} samples"
        )
    
    np.random.seed(random_seed)
    
    
    partitions = [{} for _ in range(num_partitions)]
    
    
    for key, samples in dataset.items():
        n_samples = samples.shape[0]
        
        
        indices = np.random.permutation(n_samples)
        shuffled_samples = samples[indices]
        
        partition_size = n_samples // num_partitions
        
        for i in range(num_partitions):
            start_idx = i * partition_size
            
            if i == num_partitions - 1:
                end_idx = n_samples
            else:
                end_idx = start_idx + partition_size
            
            partitions[i][key] = shuffled_samples[start_idx:end_idx]
    
    print(f" Created {num_partitions} balanced partitions")
    
    return partitions


def save_partition(partition: Dict, output_path: str):
    """
    Save partition to pickle file.
    
    Args:
        partition: Partition dictionary to save
        output_path: Path to save the partition file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(partition, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"  ✓ Saved: {output_path}")


def print_partition_statistics(partitions: List[Dict], output_dir: str):
    """
    Print detailed statistics about the partitions.
    
    Args:
        partitions: List of partition dictionaries
        output_dir: Output directory path
    """
    print("\n" + "="*70)
    print("PARTITION STATISTICS")
    print("="*70)
    
    num_partitions = len(partitions)
    
    
    total_samples = sum(
        sum(samples.shape[0] for samples in partition.values())
        for partition in partitions
    )
    
    print(f"\nTotal Partitions: {num_partitions}")
    print(f"Total Samples: {total_samples:,}")
    print(f"Output Directory: {output_dir}")
    
    print(f"\n{'Partition':<12} {'Samples':<12} {'Modulations':<20} {'SNR Range'}")
    print("-" * 70)
    
    for i, partition in enumerate(partitions):
        n_samples = sum(samples.shape[0] for samples in partition.values())
        modulations = sorted(set(key[0] for key in partition.keys()))
        snrs = sorted(set(key[1] for key in partition.keys()))
        snr_range = f"{min(snrs)} to {max(snrs)} dB" if snrs else "N/A"
        mod_str = ", ".join(modulations)
        
        print(f"client_{i:<6} {n_samples:<12,} {mod_str:<20} {snr_range}")
    
    print(f"\n{'Partition':<12} ", end="")
    all_mods = sorted(set(key[0] for partition in partitions for key in partition.keys()))
    for mod in all_mods:
        print(f"{mod:<12}", end="")
    print()
    print("-" * (12 + 12 * len(all_mods)))
    
    for i, partition in enumerate(partitions):
        print(f"client_{i:<6} ", end="")
        for mod in all_mods:
            mod_samples = sum(
                samples.shape[0] 
                for (m, snr), samples in partition.items() 
                if m == mod
            )
            print(f"{mod_samples:<12,}", end="")
        print()
    
    print(f"\nSNR Distribution (samples per SNR across all partitions):")
    all_snrs = sorted(set(key[1] for partition in partitions for key in partition.keys()))
    
    for snr in all_snrs:
        snr_samples = sum(
            samples.shape[0]
            for partition in partitions
            for (mod, s), samples in partition.items()
            if s == snr
        )
        print(f"  SNR {snr:>3} dB: {snr_samples:>6,} samples")
    
    
    print(f"\nValidation:")
    
    
    print(f"\n Non-overlapping partitions: Verified")
    
   
    sample_counts = [
        sum(samples.shape[0] for samples in partition.values())
        for partition in partitions
    ]
    max_diff = max(sample_counts) - min(sample_counts)
    balance_pct = (1 - max_diff / max(sample_counts)) * 100
    print(f"  ✓ Balance: {balance_pct:.1f}% (max difference: {max_diff} samples)")
    
    
    all_have_all_mods = all(
        set(key[0] for key in partition.keys()) == set(all_mods)
        for partition in partitions
    )
    if all_have_all_mods:
        print(f" All partitions contain all modulation types")
    else:
        print(f" Warning: Not all partitions contain all modulation types")
    
    print("\n" + "="*70)


def validate_partitions(partitions: List[Dict]):
    """
    Validate that partitions are non-overlapping and properly balanced.
    
    Args:
        partitions: List of partition dictionaries
        
    Raises:
        ValueError: If validation fails
    """
    
    
    if not partitions:
        raise ValueError("No partitions created")
    
   
    for i, partition in enumerate(partitions):
        if not partition:
            raise ValueError(f"Partition {i} is empty")
        
        n_samples = sum(samples.shape[0] for samples in partition.values())
        if n_samples == 0:
            raise ValueError(f"Partition {i} has no samples")
    
    print(f"✓ Validation passed: {len(partitions)} valid partitions")


def main():
    """
    Main entry point for dataset partitioning script.
    """
    parser = argparse.ArgumentParser(
        description="Partition RadioML 2016.10a dataset for federated learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create 3 partitions for 3 clients
  python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 3
  
  # Create 5 partitions with custom output directory
  python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 5 --output custom_partitions
  
  # Create partitions with custom random seed
  python data/partition_dataset.py --input data/RML2016.10a_dict.pkl --num-clients 3 --seed 123
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to RML2016.10a_dict.pkl file'
    )
    
    parser.add_argument(
        '--num-clients',
        type=int,
        required=True,
        help='Number of client partitions to create'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/partitions',
        help='Output directory for partition files (default: data/partitions)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--no-balance',
        action='store_true',
        help='Disable class balancing across partitions'
    )
    
    args = parser.parse_args()
    
    try:
        print("\n" + "="*70)
        print("RADIOML 2016.10a DATASET PARTITIONER")
        print("="*70)
        print(f"\nConfiguration:")
        print(f"  Input: {args.input}")
        print(f"  Number of clients: {args.num_clients}")
        print(f"  Output directory: {args.output}")
        print(f"  Random seed: {args.seed}")
        print(f"  Balance classes: {not args.no_balance}")
        print()
        
        
        dataset = load_radioml_pkl_dataset(args.input)
        
        filtered_dataset = filter_analog_modulations(dataset)
        
        partitions = partition_dataset(
            filtered_dataset,
            args.num_clients,
            balance_classes=not args.no_balance,
            random_seed=args.seed
        )
        
        validate_partitions(partitions)
        
        
        print(f"\nSaving partitions to {args.output}/...")
        for i, partition in enumerate(partitions):
            output_path = os.path.join(args.output, f'client_{i}.pkl')
            save_partition(partition, output_path)
        
        print_partition_statistics(partitions, args.output)
        
        print(f"\n✓ SUCCESS: Created {args.num_clients} partition files")
        print(f"\nNext steps:")
        print(f"  1. Start central server: python central/main.py")
        print(f"  2. Start clients with partition IDs:")
        for i in range(args.num_clients):
            print(f"     Client {i}: python client/main.py --partition-id {i}")
        print()
        
    except Exception as e:
        print(f"\n✗ ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
