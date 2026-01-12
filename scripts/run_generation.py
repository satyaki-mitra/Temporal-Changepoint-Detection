# Dependencies
import sys
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_util import setup_logger
from src.utils.logging_util import log_section_header
from src.generation.generator import PHQ9DataGenerator
from config.generation_config import DataGenerationConfig
from src.generation.validators import print_validation_report
from config.generation_config import validate_against_literature



def parse_arguments():
    """
    Parse command-line arguments for PHQ-9 synthetic data generation
    """
    parser = argparse.ArgumentParser(description     = 'Generate synthetic PHQ-9 data with temporal dependencies and clinical realism',
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                    )

    # Sample size controls
    parser.add_argument("--patients", type = int, help = "Number of patients")
    parser.add_argument("--days", type = int, help = "Study duration in days")

    # Model parameters
    parser.add_argument("--ar-coef", type = float, help = "AR(1) autocorrelation coefficient")
    parser.add_argument("--baseline", type = float, help = "Mean baseline PHQ-9 score")
    parser.add_argument("--recovery-rate", type = float, help = "Mean daily recovery rate")
    parser.add_argument("--relapse-dist", choices = ["exponential", "gamma", "lognormal"], help = "Relapse magnitude distribution")

    # NEW: Response pattern controls
    parser.add_argument("--enable-response-patterns", action = "store_true", help = "Enable heterogeneous response patterns")
    parser.add_argument("--disable-response-patterns", action = "store_true", help = "Disable response patterns (uniform trajectories)")
    parser.add_argument("--enable-plateau", action = "store_true", help = "Enable plateau logic")
    parser.add_argument("--disable-plateau", action = "store_true", help = "Disable plateau logic")

    # Execution control
    parser.add_argument("--seed", type = int, help = "Random seed")
    parser.add_argument("--output", type = Path, help = "Output CSV path")
    parser.add_argument("--skip-validation", action = "store_true", help = "Skip literature validation")
    parser.add_argument("--log-level", choices = ["DEBUG", "INFO", "WARNING", "ERROR"], default = "INFO", help = "Logging level")

    return parser.parse_args()


def _apply_cli_overrides(config: DataGenerationConfig, args: argparse.Namespace) -> DataGenerationConfig:
    """
    Apply CLI overrides to the DataGenerationConfig instance and returns a new validated config instance to ensure all validators run
    """
    # Convert config to dict
    try:
        config_dict = config.model_dump()

    except AttributeError:
        config_dict = config.dict()
    
    # Apply CLI overrides to dict
    if (args.patients is not None):
        config_dict['total_patients'] = args.patients

    if (args.days is not None):
        config_dict['total_days'] = args.days

    if (args.ar_coef is not None):
        config_dict['ar_coefficient'] = args.ar_coef

    if (args.baseline is not None):
        config_dict['baseline_mean_score'] = args.baseline

    if (args.recovery_rate is not None):
        config_dict['recovery_rate_mean'] = args.recovery_rate

    if (args.relapse_dist is not None):
        config_dict['relapse_distribution'] = args.relapse_dist

    if (args.seed is not None):
        config_dict['random_seed'] = args.seed

    if (args.output is not None):
        config_dict['output_data_path'] = args.output

    # NEW: Response pattern overrides
    if args.enable_response_patterns:
        config_dict['enable_response_patterns'] = True

    elif args.disable_response_patterns:
        config_dict['enable_response_patterns'] = False

    if args.enable_plateau:
        config_dict['enable_plateau_logic'] = True

    elif args.disable_plateau:
        config_dict['enable_plateau_logic'] = False
    
    # Reconstruct config to trigger full validation
    return DataGenerationConfig(**config_dict)


def main() -> int:
    """
    Main execution entry point
    """
    args   = parse_arguments()

    # Initialize logging early
    logger = setup_logger("generation", 
                          log_level = args.log_level,
                         )

    log_section_header(logger, "PHQ-9 SYNTHETIC DATA GENERATION v2.0")

    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Create configuration
        config = DataGenerationConfig()

        # Apply CLI overrides (returns NEW validated config)
        config = _apply_cli_overrides(config, args) 

        # Print configuration summary
        logger.info("Configuration Summary:")
        for section, params in config.get_summary().items():
            logger.info(f"  {section}:")
            
            for key, value in params.items():
                logger.info(f"    {key}: {value}")

        # Validate configuration against literature
        if not args.skip_validation:
            logger.info("Validating configuration against literature...")
            literature_validation = validate_against_literature(config)

            if literature_validation["warnings"]:
                logger.warning("Configuration warnings detected:")
                
                for warning in literature_validation["warnings"]:
                    logger.warning(f"  {warning}")

        # Initialize generator
        logger.info("Initializing PHQ-9 data generator...")
        generator        = PHQ9DataGenerator(config)

        # Run generation pipeline
        logger.info("Starting data generation pipeline...")
        data, validation = generator.generate_and_validate()

        # Summary logging
        logger.info("=" * 80)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Dataset shape : {data.shape}")
        logger.info(f"Data output   : {config.output_data_path}")
        logger.info(f"Metadata      : {config.output_data_path.with_suffix('.metadata.json')}")
        logger.info(f"Validation    : {config.validation_report_path}")

        # Print validation report to stdout
        print_validation_report(validation)

        if not validation["overall_valid"]:
            logger.warning("Data generated with warnings. Review validation report")
            return 2

        logger.info("Data generation completed successfully")
        return 0

    except Exception as exc:
        logger.exception(f"Fatal error during generation: {exc}")
        return 1

    finally:
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



# EXECUTION 
if __name__ == "__main__":
    sys.exit(main())