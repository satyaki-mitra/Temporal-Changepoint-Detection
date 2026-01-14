# Dependencies
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.eda_config import EDAConfig
from src.eda.analyzer import PHQ9DataAnalyzer
from src.utils.logging_util import setup_logger
from src.utils.logging_util import log_section_header
from src.eda.visualizations import VisualizationGenerator
from src.eda.distribution_comparator import DistributionComparator


def parse_arguments():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description     = 'Compare EDA results across multiple PHQ-9 datasets',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                    )
    
    parser.add_argument('--data-dir', type = Path, default = Path('data/raw'), help = 'Directory containing data files')
    parser.add_argument('--output-dir', type = Path, default = Path('results/comparison'), help = 'Output directory for comparison results')
    parser.add_argument('--patterns', nargs = '+', default = ['exponential', 'gamma', 'lognormal'], help = 'Distribution patterns to compare')
    parser.add_argument('--log-level', choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR'], default = 'INFO', help = 'Logging level')
    
    return parser.parse_args()


def run_eda_for_dataset(data_path: Path, output_dir: Path, logger) -> dict:
    """
    Run EDA for a single dataset
    
    Arguments:
    ----------
        data_path  { Path }   : Path to data file
        
        output_dir { Path }   : Output directory
        
        logger     { Logger } : Logger instance
    
    Returns:
    --------
             { dict }         : EDA results dictionary
    """
    # Create config
    config                        = EDAConfig()
    config.data_file_path         = data_path
    config.results_base_directory = output_dir
    
    # Initialize analyzer
    analyzer                      = PHQ9DataAnalyzer(config)
    
    # Load data
    analyzer.load_data()
    
    # Run full analysis
    results                       = analyzer.run_full_analysis()
    
    return {'dataset_name' : data_path.stem,
            'data'         : analyzer.data,
            'labels'       : results['labels'],
            'metadata'     : analyzer.metadata,
            'results'      : results,
           }


def main():
    """
    Main execution
    """
    args   = parse_arguments()
    
    # Setup logging
    logger = setup_logger('comparison', log_level = args.log_level)
    
    log_section_header(logger, "PHQ-9 DISTRIBUTION COMPARISON")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Find data files
        data_files = list()
        
        for pattern in args.patterns:
            pattern_file = args.data_dir / f"synthetic_phq9_data_{pattern}.csv"
            
            if pattern_file.exists():
                data_files.append(pattern_file)
                logger.info(f"Found: {pattern_file}")
            
            else:
                logger.warning(f"Missing: {pattern_file}")
                
            if not data_files:
                logger.error("No data files found!")
                return 1
    
        logger.info(f"\nAnalyzing {len(data_files)} datasets...")
        
        # Run EDA for each dataset
        eda_results_list = list()
        
        for data_file in data_files:
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing: {data_file.stem}")
            logger.info(f"{'='*80}")
            
            output_dir = args.output_dir / data_file.stem
            
            eda_result = run_eda_for_dataset(data_file, output_dir, logger)
            eda_results_list.append(eda_result)
            
            logger.info(f"Completed: {data_file.stem}")
        
        # Compare datasets
        logger.info(f"\n{'='*80}")
        logger.info("COMPARING DATASETS")
        logger.info(f"{'='*80}")
        
        comparator    = DistributionComparator(logger = logger)
        comparison_df = comparator.compare_datasets(eda_results_list)
        
        # Log results
        logger.info("\nComparison Results:")
        logger.info(f"\n{comparison_df.to_string()}")
        
        # Save comparison
        comparison_output_dir = args.output_dir / "comparison_summary"
        comparison_output_dir.mkdir(parents = True, exist_ok = True)
        
        comparison_df.to_csv(path_or_buf = comparison_output_dir / "dataset_comparison.csv", 
                             index       = False,
                            )
        
        # Generate comparison report
        comparator.generate_comparison_report(comparison_df, 
                                              comparison_output_dir / "comparison_report.json",
                                             )
        
        # Visualizations
        logger.info("\nGenerating comparison visualizations...")
        
        visualizer = VisualizationGenerator()
        
        visualizer.plot_distribution_comparison(comparison_df,
                                                save_path = comparison_output_dir / "distribution_comparison.png",
                                               )
        
        visualizer.plot_composite_scores(comparison_df,
                                         save_path = comparison_output_dir / "composite_scores.png",
                                        )
        
        # Recommendation
        best_dataset = comparison_df.iloc[0]['dataset_name']
        
        logger.info("\n" + "="*80)
        logger.info("\nDATASET RECOMMENDATION")
        logger.info("="*80)
        logger.info(f"Best dataset: {best_dataset}")
        logger.info(f"Composite score: {comparison_df.iloc[0]['composite_score']:.2f}/100")
        logger.info(f"\nUse this dataset for change-point detection:")
        logger.info(f"  {args.data_dir / f'{best_dataset}.csv'}\n")
        
        logger.info("\nComparison completed successfully!")
        return 0
        
    except Exception as e:
        logger.exception(f"Error during comparison: {e}")
        return 1

    finally:
        logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        
if __name__ == "__main__":
    sys.exit(main())

