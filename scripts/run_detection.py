# Dependencies
import sys
import argparse
from typing import List
from pathlib import Path
from datetime import datetime

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phq9_analysis.utils.logging_util import setup_logger
from config.model_selection_config import ModelSelectorConfig
from config.detection_config import ChangePointDetectionConfig
from phq9_analysis.utils.logging_util import log_section_header
from phq9_analysis.detection.model_selector import ModelSelector
from phq9_analysis.detection.detector import ChangePointDetectionOrchestrator


# Argument Parsing
def parse_arguments():
    """
    Parse command-line arguments: CLI arguments override config defaults selectively
    """
    parser = argparse.ArgumentParser(description     = "PHQ-9 Change Point Detection (PELT / BOCPD)",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                    )


    # Execution control
    parser.add_argument('--execution-mode', choices = ['single', 'compare'], default = None, help = "Execution mode: single detector or side-by-side comparison")

    parser.add_argument('--detectors', nargs = '+', choices = ['pelt', 'bocpd'], default = None, help = "Detectors to run (order preserved)")

    # Data
    parser.add_argument('--data', type = Path, default = None, help = "Path to PHQ-9 data CSV file")

    # PELT parameters
    parser.add_argument('--penalty', type = float, default = None)
    parser.add_argument('--auto-tune', action = 'store_true')
    parser.add_argument('--cost-model', choices = ['l1', 'l2', 'rbf', 'ar'], default = None)
    parser.add_argument('--min-size', type = int, default = None)

    # BOCPD parameters
    parser.add_argument('--hazard-lambda', type = float, default = None)
    parser.add_argument('--posterior-threshold', type = float, default = None)

    # Statistical testing
    parser.add_argument('--alpha', type = float, default = None)
    parser.add_argument('--correction', choices = ['bonferroni', 'fdr_bh', 'none'], default = None)

    # Output & logging
    parser.add_argument('--output-dir', type = Path, default = None)
    parser.add_argument('--log-level', choices = ['DEBUG', 'INFO', 'WARNING', 'ERROR'], default = 'INFO')

    return parser.parse_args()



# Main Entry Point
def main():
    args = parse_arguments()

    # Build configuration
    config = ChangePointDetectionConfig()

    # Execution control
    if args.execution_mode:
        config.execution_mode = args.execution_mode

    if args.detectors:
        config.detectors = args.detectors

    # Data
    if args.data:
        config.data_path = args.data

    # PELT overrides
    if args.penalty is not None:
        config.penalty = args.penalty

    if args.auto_tune:
        config.auto_tune_penalty = True

    if args.cost_model:
        config.cost_model = args.cost_model

    if args.min_size:
        config.minimum_segment_size = args.min_size

    # BOCPD overrides
    if args.hazard_lambda is not None:
        config.hazard_lambda = args.hazard_lambda

    if args.posterior_threshold is not None:
        config.cp_posterior_threshold = args.posterior_threshold

    # Statistical testing
    if args.alpha:
        config.alpha = args.alpha

    if args.correction:
        config.multiple_testing_correction = args.correction

    # Output
    if args.output_dir:
        config.results_base_directory = args.output_dir

    # Logging
    logger = setup_logger(module_name = 'detection',
                          log_level   = args.log_level,
                          log_dir     = Path('logs'),
                         )

    log_section_header(logger, "PHQ-9 CHANGE POINT DETECTION")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    logger.info("Configuration Summary:")

    for section, values in config.get_summary().items():
        logger.info(f"  {section}: {values}")

    try:
        # Run orchestration
        orchestrator = ChangePointDetectionOrchestrator(config = config)

        logger.info("\nRunning detection pipeline...")
        results      = orchestrator.run()

        # Summaries
        logger.info("\n" + "=" * 80)
        logger.info("DETECTION SUMMARY")
        logger.info("=" * 80)

        for method, res in results.items():
            logger.info(f"\nDetector: {method.upper()}")

            if (method == 'pelt'):
                logger.info(f"  Change points: {len(res['change_points']) - 1}")
                logger.info(f"  Penalty used : {res['penalty_used']:.3f}")
                logger.info(f"  Significant : {res['validation']['n_significant']}/ {res['validation']['n_changepoints']}")

            if (method == 'bocpd'):
                logger.info(f"  Change points: {res['validation']['n_changepoints']}")
                logger.info(f"  Posterior threshold: {res['validation']['posterior_threshold']}")

        # Model selection (compare mode only)
        if (config.execution_mode == 'compare'):
            logger.info("\nRunning model selection...")

            selector_config  = ModelSelectorConfig()
            model_selector   = ModelSelector(config = selector_config)

            selection_result = model_selector.select(raw_results = results)

            logger.info("\n" + "=" * 80)
            logger.info("MODEL SELECTION RESULT")
            logger.info("=" * 80)

            logger.info(f"Best model: {selection_result['best_model']}")
            logger.info(f"Ranking  : {selection_result['ranking']}")

            if selector_config.generate_explanations:
                logger.info("\nExplanation:")
                logger.info(selection_result['explanation'])

        logger.info("\nOutput directory:")
        logger.info(f"  {config.results_base_directory}")

        logger.info("\nDetection completed successfully.")
        return 0

    except Exception as exc:
        logger.exception(f"\nDetection failed: {exc}")
        return 1

    finally:
        logger.info(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



# Execution
if __name__ == "__main__":
    sys.exit(main())