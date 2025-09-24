import pandas as pd
import os
from pathlib import Path

# Get the directory where this script is located
script_dir = Path(__file__).parent.absolute()
# Go up one level to M&M directory and then to data directory
project_root = script_dir.parent

# Set data paths relative to project root
data_path = project_root / "data" / "M&M_dataset.csv"
output_dir = project_root / "data" / "splits"

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Read dataset
df = pd.read_csv(data_path)

print(f"Total dataset rows: {len(df)}")
print(f"Column names: {df.columns.tolist()}")
print(f"\nType categories in dataset:")
print(df['type'].value_counts())

# Split dataset by type column and sample
test_data = pd.DataFrame()

for type_name in df['type'].unique():
    # Filter and save data for current type
    type_df = df[df['type'] == type_name]
    file_path = output_dir / f"{type_name}_split.csv"
    type_df.to_csv(file_path, index=False)

    print(f"Saved {type_name} type data to: {file_path}")
    print(f"  - Number of rows: {len(type_df)}")
    print(f"  - File size: {os.path.getsize(file_path) / 1024:.1f} KB\n")

    # Sample directly from memory without reloading file
    sampled_df = type_df.sample(n=2, random_state=42)
    test_data = pd.concat([test_data, sampled_df], ignore_index=True)
    print(f"Sampled {len(sampled_df)} rows from {type_name}")

# Save test data in the same directory as the script
test_data_path = script_dir / "test_data.csv"
test_data.to_csv(test_data_path, index=False)

print(f"\nTest dataset saved to: {test_data_path}")
print(f"Total test dataset rows: {len(test_data)}")
print(f"\nType distribution:")
print(test_data['type'].value_counts())