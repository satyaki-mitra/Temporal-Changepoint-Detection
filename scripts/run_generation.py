# Dependencies
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.generation_config import DataGenerationConfig
from phq9_analysis.utils.logging_util import setup_logger
from phq9_analysis.utils.logging_util import log_section_header
from config.generation_config import validate_against_literature
from phq9_analysis.generation.generator import PHQ9DataGenerator
from phq9_analysis.generation.validators import print_validation_report


def parse_arguments():
    """
    Parse command-line arguments
    """
    parser = argparse.ArgumentParser(description     = 'Generate synthetic PHQ-9 data with temporal dependencies',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                    )
    
    # Sample size
    parser.add_argument('--patients', type = int, default = None, help = 'Number of patients')
    parser.add_argument('--days', type = int, default = None, help = 'Study duration in days')
    parser.add_argument('--observation-days', type = int, default = None, help = 'Number of observation days')
    
    # Model parameters
    parser.add_argument('--ar-coef', type = float, default = None, help = 'AR(1) autocorrelation coefficient')
    parser.add_argument('--baseline', type = float, default = None, help = 'Mean baseline PHQ-9 score')
    parser.add_argument('--recovery-rate', type = float, default = None, help = 'Mean daily recovery rate')
    
    # Control
    parser.add_argument('--seed', type = int, default = None, help = 'Random seed')
    parser.add_argument('--output', type = Path, default = None, help = 'Output CSV path')
    parser.add_argument('--skip-validation', action = 'store_true', help = 'Skip literature validation')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default = 'INFO', help = 'Logging level')
    
    return parser.parse_args()


def main():
    """
    Main execution
    """
    args   = parse_arguments()
    
    # Create config
    config = DataGenerationConfig()
    
    # Override with CLI args
    if args.patients:
        config.total_patients = args.patients

    if args.days:
        config.total_days = args.days
    
    if args.observation_days:
        config.required_sample_count = args.observation_days
    
    if args.ar_coef:
        config.ar_coefficient = args.ar_coef
    
    if args.baseline:
        config.baseline_mean_score = args.baseline
    
    if args.recovery_rate:
        config.recovery_rate_mean = args.recovery_rate
    
    if args.seed:
        config.random_seed = args.seed
    
    if args.output:
        config.output_data_path = args.output
    
    # Setup logging
    logger = setup_logger('generation', log_level = args.log_level)
    
    log_section_header(logger, "PHQ-9 SYNTHETIC DATA GENERATION")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Validate config
        if not args.skip_validation:
            logger.info("\nValidating configuration against literature...")
            lit_validation = validate_against_literature(config)
            
            if lit_validation['warnings']:
                logger.warning("Configuration has warnings:")
                for warning in lit_validation['warnings']:
                    logger.warning(f"  {warning}")
        
        # Generate data
        logger.info("\nInitializing generator...")
        generator        = PHQ9DataGenerator(config)
        
        logger.info("\nGenerating data...")
        data, validation = generator.generate_and_validate()
        
        # Print results
        logger.info("\n" + "="*80)
        logger.info("GENERATION COMPLETE")
        logger.info("="*80)
        logger.info(f"\nDataset shape: {data.shape}")
        logger.info(f"Output: {config.output_data_path}")
        logger.info(f"Validation: {config.validation_report_path}")
        
        # Print validation report
        print_validation_report(validation)
        
        if not validation['overall_valid']:
            logger.warning("\nData generated with warnings. Review validation report.")
            return 2
        
        logger.info("\nData generation completed successfully!")
        return 0
        
    except Exception as e:
        logger.exception(f"Error during generation: {e}")
        return 1
    
    finally:
        logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    sys.exit(main())