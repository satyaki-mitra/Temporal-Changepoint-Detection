# Dependencies
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.eda_config import EDAConfig
from phq9_analysis.eda.analyzer import PHQ9DataAnalyzer
from phq9_analysis.utils.logging_util import setup_logger
from phq9_analysis.utils.logging_util import log_section_header


def parse_arguments():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description     = 'Perform exploratory data analysis on PHQ-9 data',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                    )
    
    parser.add_argument('--data', type = Path, default = None, help = 'Path to PHQ-9 data CSV file')
    parser.add_argument('--max-clusters', type = int, default = None, help = 'Maximum number of clusters to test')
    parser.add_argument('--n-clusters', type = int, default = None, help = 'Number of clusters (skip optimization if specified)')
    parser.add_argument('--temporal', action = 'store_true', help = 'Use temporal-aware clustering')
    parser.add_argument( '--output-dir', type = Path, default = None, help = 'Output directory for results')
    parser.add_argument('--log-level', choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR'], default = 'INFO', help = 'Logging level')
    
    return parser.parse_args()


def main():
    """
    Main execution
    """
    args   = parse_arguments()
    
    # Create config
    config = EDAConfig()
    
    # Override with CLI args
    if args.data:
        config.data_file_path = args.data
    
    if args.max_clusters:
        config.max_clusters_to_test = args.max_clusters
    
    if args.temporal:
        config.use_temporal_clustering = True
    
    if args.output_dir:
        config.results_base_directory = args.output_dir
    
    # Setup logging
    logger = setup_logger('eda', log_level = args.log_level)
    
    log_section_header(logger, "PHQ-9 EXPLORATORY DATA ANALYSIS")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Initialize analyzer
        logger.info("\nInitializing analyzer...")
        analyzer = PHQ9DataAnalyzer(config)
        
        # Load data
        logger.info("\nLoading data...")
        analyzer.load_data()
        
        # Run analysis
        if args.n_clusters:
            # Use specified number of clusters
            logger.info(f"\nUsing {args.n_clusters} clusters (skipping optimization)...")
            labels           = analyzer.fit_clustering(args.n_clusters)
            cluster_analysis = analyzer.analyze_clusters(labels)
            analyzer.generate_visualizations(labels, args.n_clusters)
        
        else:
            # Full analysis with optimization
            logger.info("\nRunning full analysis...")
            results = analyzer.run_full_analysis()
        
        logger.info("\n" + "="*80)
        logger.info("EDA COMPLETE")
        logger.info("="*80)
        logger.info(f"\nResults saved to: {config.results_base_directory}")
        
        logger.info("\nEDA completed successfully!")
        return 0
        
    except Exception as e:
        logger.exception(f"Error during EDA: {e}")
        return 1
    
    finally:
        logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    sys.exit(main())