import os
import pandas as pd

def create_samples():
    raw_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
    sample_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw_samples")
    
    os.makedirs(sample_dir, exist_ok=True)
    
    for file in os.listdir(raw_dir):
        if file.endswith(".csv"):
            file_path = os.path.join(raw_dir, file)
            df = pd.read_csv(file_path)
            
            # Get 500 rows, or all rows if less than 500
            sample_df = df.head(500)
            
            sample_path = os.path.join(sample_dir, file)
            sample_df.to_csv(sample_path, index=False)
            print(f"Sampled {len(sample_df)} rows from {file} -> saved to raw_samples")

if __name__ == "__main__":
    create_samples()
