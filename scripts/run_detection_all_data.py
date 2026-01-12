# Dependencies
import sys
import subprocess
from pathlib import Path


DATASETS = ["synthetic_phq9_data_exponential.csv",
            "synthetic_phq9_data_gamma.csv",
            "synthetic_phq9_data_lognormal.csv"
           ]

def main():
    print("=" * 70)
    print("Running Change-Point Detection on All Datasets")
    print("=" * 70)
    
    data_dir = Path("data/raw")
    
    for i, dataset in enumerate(DATASETS, 1):
        dist_name  = dataset.replace("synthetic_phq9_data_", "").replace(".csv", "")
        data_path  = data_dir / dataset
        output_dir = Path(f"results/detection/{dist_name}")
        
        print(f"\n[{i}/{len(DATASETS)}] Processing: {dataset}")
        print("-" * 70)
        
        if not data_path.exists():
            print(f"✗ File not found: {data_path}")
            continue
        
        cmd = ["python", "scripts/run_detection.py",
               "--execution-mode", "compare",
               "--data", str(data_path),
               "--output-dir", str(output_dir),
               "--log-level", "INFO"
              ]
        
        try:
            subprocess.run(cmd, check = True)
            print(f"✓ Completed: {dist_name}")

        except subprocess.CalledProcessError as e:
            print(f"✗ Failed: {dist_name}")
            return 1
    
    print("\n" + "=" * 70)
    print("All detections completed!")
    print("=" * 70)
    return 0


# Execution
if __name__ == "__main__":
    sys.exit(main())