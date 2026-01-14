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
    parser.add_argument('--output-dir', type = Path, default = None, help = 'Output directory for results')
    parser.add_argument('--no-metadata', action = 'store_true', help = 'Skip metadata loading')
    parser.add_argument('--no-response-patterns', action = 'store_true', help = 'Skip response pattern analysis')
    parser.add_argument('--no-relapses', action = 'store_true', help = 'Skip relapse detection')
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
    
    if args.no_metadata:
        config.load_metadata = False
    
    if args.no_response_patterns:
        config.analyze_response_patterns = False
    
    if args.no_relapses:
        config.detect_relapses = False
    
    # Setup logging
    logger = setup_logger('eda', log_level = args.log_level)
    
    log_section_header(logger, "PHQ-9 EXPLORATORY DATA ANALYSIS v2.0")
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
            
            # Response patterns and relapses
            response_analysis = analyzer.analyze_response_patterns()
            relapse_results   = analyzer.detect_relapses()
            
            # Visualizations
            analyzer.generate_visualizations(labels           = labels, 
                                            n_clusters       = args.n_clusters,
                                            response_analysis = response_analysis,
                                            relapse_results  = relapse_results,
                                           )
        
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