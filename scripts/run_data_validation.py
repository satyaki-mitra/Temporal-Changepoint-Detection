# Dependencies
import sys
import json
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.comparator import DatasetComparator
from src.utils.logging_util import setup_logger, log_section_header
from src.validation.literature_validator import LiteratureValidator


DATASETS = ["gamma",
            "lognormal",
            "exponential",
           ]


def main():
    logger             = setup_logger('validation', log_level = 'INFO')
    
    log_section_header(logger, "LITERATURE-BASED DATA VALIDATION")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    validator          = LiteratureValidator()
    validation_reports = list()
    
    # Validate each dataset
    for dataset in DATASETS:
        logger.info(f"\nValidating dataset: {dataset}")
        logger.info("-" * 70)
        
        data_path    = Path(f"data/raw/synthetic_phq9_data_{dataset}.csv")
        eda_path     = Path(f"results/eda/{dataset}/analysis_summary.json")
        gen_val_path = Path(f"results/generation/validation_reports/validation_report_{dataset}.json")
        
        if not all([data_path.exists(), eda_path.exists(), gen_val_path.exists()]):
            logger.error(f"Missing files for {dataset}")
            continue
        
        # Run validation
        validation   = validator.validate_dataset(data_path, eda_path, gen_val_path)
        validation_reports.append(validation)
        
        # Log results
        logger.info(f"  Compliance score: {validation['literature_compliance_score']:.3f}")
        logger.info(f"  Overall valid: {validation['overall_valid']}")
        logger.info(f"  Warnings: {len(validation['warnings'])}")
        logger.info(f"  Errors: {len(validation['errors'])}")
        
        # Save individual validation
        val_dir      = Path("results/validation")
        val_dir.mkdir(parents = True, exist_ok = True)
        
        with open(val_dir / f"{dataset}_validation.json", 'w') as f:
            json.dump(obj     = validation, 
                      fp      = f, 
                      indent  = 2, 
                      default = str,
                     )
    
    # Compare and select best
    logger.info("\n" + "=" * 70)
    logger.info("DATASET COMPARISON & SELECTION")
    logger.info("=" * 70)
    
    comparator        = DatasetComparator(validation_reports)
    
    # Generate comparison report
    comparison_report = comparator.generate_comparison_report(Path("results/validation/comparison_report.json"))
    
    # Rank datasets
    ranking           = comparator.rank_datasets()

    logger.info("\nDataset Rankings:")
    logger.info(ranking.to_string(index = False))
    
    # Generate plots
    comparator.plot_comparison(Path("results/validation/dataset_comparison.png"))
    
    # Select best dataset
    try:
        best_dataset = comparator.select_best_dataset()
        logger.info(f"\n✓ BEST DATASET SELECTED: {best_dataset}")
        
        # Copy to finalized_data/
        source       = Path(f"data/raw/synthetic_phq9_data_{best_dataset}.csv")
        dest_dir     = Path("data/finalized_data")
        dest_dir.mkdir(parents = True, exist_ok = True)

        dest         = dest_dir / "phq9_data_finalized.csv"
        
        shutil.copy(source, dest)
        logger.info(f"  Copied to: {dest}")
        
        # Save selection metadata
        metadata     = {'selected_dataset'    : best_dataset,
                        'selection_timestamp' : datetime.now().isoformat(),
                        'compliance_score'    : float(ranking.iloc[0]['compliance_score']),
                        'all_rankings'        : ranking.to_dict(orient = 'records'),
                       }
        
        with open(dest_dir / "selection_metadata.json", 'w') as f:
            json.dump(obj    = metadata, 
                      fp     = f, 
                      indent = 4,
                     )
        
        logger.info("\n✓ Validation and selection complete!")
        return 0
    
    except ValueError as e:
        logger.error(f"\n✗ No valid dataset found: {e}")
        return 1
    
    finally:
        logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Execution
if __name__ == "__main__":
    sys.exit(main())