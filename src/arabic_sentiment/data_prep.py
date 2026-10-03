from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data(input_path: str, output_dir: str) -> None:
    """Loads raw data, cleans it, and applies a 70/15/15 stratified split."""
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    
    initial_len = len(df)
    
    # 1. Drop missing review descriptions
    df = df.dropna(subset=['review_description'])
    
    # 2. Drop exact duplicate rows
    df = df.drop_duplicates()
    
    print(f"Dropped {initial_len - len(df)} rows (missing/duplicates). Remaining: {len(df)}")
    
    # 3. Stratified Split 70/15/15
    # First split: 70% train, 30% temp (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size=0.30, 
        stratify=df['rating'], 
        random_state=42
    )
    
    # Second split: split the 30% temp in half -> 15% val, 15% test
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        stratify=temp_df['rating'], 
        random_state=42
    )
    
    print(f"Split sizes -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # 4. Save splits
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(out_path / "train.csv", index=False)
    val_df.to_csv(out_path / "val.csv", index=False)
    test_df.to_csv(out_path / "test.csv", index=False)
    print(f"Data successfully saved to {output_dir}")

if __name__ == "__main__":
    prepare_data(
        input_path="data/raw/Final_Data.csv", 
        output_dir="data/processed"
    )