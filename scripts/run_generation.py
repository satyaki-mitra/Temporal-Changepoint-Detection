# Dependencies
import sys
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from phq9_analysis.utils.logging_util import setup_logger
from config.generation_config import DataGenerationConfig
from phq9_analysis.utils.logging_util import log_section_header
from config.generation_config import validate_against_literature
from phq9_analysis.generation.generator import PHQ9DataGenerator
from phq9_analysis.generation.validators import print_validation_report


def parse_arguments():
    """
    Parse command-line arguments for PHQ-9 synthetic data generation.
    """
    parser = argparse.ArgumentParser(description     = "Generate synthetic PHQ-9 data with temporal dependencies",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                    )

    # Sample size controls
    parser.add_argument("--patients", type = int, help = "Number of patients")
    parser.add_argument("--days", type = int, help = "Study duration in days")

    # Model parameters
    parser.add_argument("--ar-coef", type = float, help = "AR(1) autocorrelation coefficient")
    parser.add_argument("--baseline", type = float, help = "Mean baseline PHQ-9 score")
    parser.add_argument("--recovery-rate", type = float, help = "Mean daily recovery rate")

    # Execution control
    parser.add_argument("--seed", type = int, help = "Random seed")
    parser.add_argument("--output", type = Path, help = "Output CSV path")
    parser.add_argument("--skip-validation", action = "store_true", help = "Skip literature validation")
    parser.add_argument("--log-level", choices = ["DEBUG", "INFO", "WARNING", "ERROR"], default = "INFO", help = "Logging level")

    return parser.parse_args()


def _apply_cli_overrides(config: DataGenerationConfig, args: argparse.Namespace):
    """
    Apply CLI overrides to the DataGenerationConfig instance.
    """
    if args.patients is not None:
        config.total_patients = args.patients

    if args.days is not None:
        config.total_days = args.days

    if args.ar_coef is not None:
        config.ar_coefficient = args.ar_coef

    if args.baseline is not None:
        config.baseline_mean_score = args.baseline

    if args.recovery_rate is not None:
        config.recovery_rate_mean = args.recovery_rate

    if args.seed is not None:
        config.random_seed = args.seed

    if args.output is not None:
        config.output_data_path = args.output


def main() -> int:
    """
    Main execution entry point.
    """
    args   = parse_arguments()

    # Initialize logging early
    logger = setup_logger("generation", log_level = args.log_level)

    log_section_header(logger, "PHQ-9 SYNTHETIC DATA GENERATION")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        # Create configuration
        config = DataGenerationConfig()

        # Apply CLI overrides
        _apply_cli_overrides(config, args)

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
        logger.info(f"Validation    : {config.validation_report_path}")

        # Print validation report to stdout
        print_validation_report(validation)

        if not validation["overall_valid"]:
            logger.warning("Data generated with warnings. Review validation report.")
            return 2

        logger.info("Data generation completed successfully.")
        return 0

    except Exception as exc:
        logger.exception(f"Fatal error during generation: {exc}")
        return 1

    finally:
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    sys.exit(main())