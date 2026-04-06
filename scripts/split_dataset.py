"""
Split CSV dataset into training and validation sets
"""

import numpy as np
import pandas as pd
from pathlib import Path

def split_csv_dataset(
    input_csv_path,
    train_output_path,
    valid_output_path,
    train_ratio=0.8,
    random_seed=42
):
    """
    Split a CSV file into training and validation sets.
    
    Parameters:
    -----------
    input_csv_path : str
        Path to input CSV file
    train_output_path : str
        Path to save training set
    valid_output_path : str
        Path to save validation set
    train_ratio : float
        Ratio of data to use for training (default: 0.8)
    random_seed : int
        Random seed for reproducibility
    """
    print(f"Loading data from {input_csv_path}...")
    
    # Load the CSV file
    df = pd.read_csv(input_csv_path, header=None)
    
    print(f"Total samples: {len(df)}")
    print(f"Total features (including targets): {df.shape[1]}")
    print(f"Input features: {df.shape[1] - 4}")
    print(f"Output targets: 4 (last 4 columns)")
    
    # Shuffle the data
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(df))
    df_shuffled = df.iloc[shuffled_indices].reset_index(drop=True)
    
    # Split into train and validation
    split_idx = int(len(df_shuffled) * train_ratio)
    df_train = df_shuffled.iloc[:split_idx]
    df_valid = df_shuffled.iloc[split_idx:]
    
    print(f"\nTraining samples: {len(df_train)} ({train_ratio*100:.1f}%)")
    print(f"Validation samples: {len(df_valid)} ({(1-train_ratio)*100:.1f}%)")
    
    # Create output directories if they don't exist
    Path(train_output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(valid_output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df_train.to_csv(train_output_path, header=False, index=False)
    df_valid.to_csv(valid_output_path, header=False, index=False)
    
    print(f"\nSaved training set to: {train_output_path}")
    print(f"Saved validation set to: {valid_output_path}")
    
    return df_train, df_valid


if __name__ == "__main__":
    # Configuration
    input_csv = "/home/muqtasid/Desktop/hiwi_actual_work/NNTrain/Data/MomentFit_1M_NoVertices/Training_1M_NoVertices.csv"  # UPDATE THIS PATH
    train_output = "Data/Training_1M_NoVertices.csv"
    valid_output = "Data/Valid_1M_NoVertices.csv"
    train_ratio = 0.8
    
    # Split the dataset
    df_train, df_valid = split_csv_dataset(
        input_csv_path=input_csv,
        train_output_path=train_output,
        valid_output_path=valid_output,
        train_ratio=train_ratio,
        random_seed=42
    )
    
    print("\nDataset split complete!")
