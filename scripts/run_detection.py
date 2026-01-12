# Dependencies
import sys
import argparse
from pathlib import Path
from datetime import datetime

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.logging_util import setup_logger
from src.utils.logging_util import log_section_header
from src.detection.model_selector import ModelSelector
from config.model_selection_config import ModelSelectorConfig
from config.detection_config import ChangePointDetectionConfig
from src.detection.detector import ChangePointDetectionOrchestrator


# CLI Argument Parsing
def parse_arguments():
    """
    Parse command-line arguments: 
    
    - CLI arguments override configuration defaults selectively
    """
    parser = argparse.ArgumentParser(description     = "PHQ-9 Change Point Detection (PELT / BOCPD)",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                    )

    # Execution control
    parser.add_argument('--execution-mode',
                        choices = ['single', 'compare', 'ensemble'],
                        default = None,
                        help    = "Execution mode"
                       )

    parser.add_argument('--detectors',
                        nargs   = '+',
                        choices = ['pelt', 'bocpd'],
                        default = None,
                        help    = "Detectors to run (subset allowed)"
                       )

    # Data
    parser.add_argument('--data',
                        type    = Path,
                        default = None,
                        help    = "Path to PHQ-9 data CSV"
                       )

    # PELT parameters
    parser.add_argument('--penalty', type = float, default = None)
    parser.add_argument('--auto-tune-penalty', action = 'store_true')
    parser.add_argument('--cost-model', choices = ['l1', 'l2', 'rbf', 'ar'], default = None)
    parser.add_argument('--min-size', type = int, default = None)

    # BOCPD parameters (tuning only)
    parser.add_argument('--hazard-tuning-method', choices = ['heuristic', 'predictive_ll'], default = None, help = "BOCPD hazard tuning method")

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
    args        = parse_arguments()

    # Build detection configuration
    try:
        config_dict = ChangePointDetectionConfig().model_dump()
    
    except AttributeError:
        config_dict = ChangePointDetectionConfig().dict()

    # Execution control
    if args.execution_mode:
        config_dict['execution_mode'] = args.execution_mode

    if args.detectors:
        config_dict['detectors'] = args.detectors

    # Data
    if args.data:
        config_dict['data_path'] = args.data

    # PELT overrides
    if args.penalty is not None:
        config_dict['penalty'] = args.penalty

    if args.auto_tune_penalty:
        config_dict['auto_tune_penalty'] = True

    if args.cost_model:
        config_dict['pelt_cost_models'] = [args.cost_model]

    if args.min_size:
        config_dict['minimum_segment_size'] = args.min_size

    # BOCPD overrides
    if args.hazard_tuning_method:
        config_dict['hazard_tuning_method'] = args.hazard_tuning_method

    if args.posterior_threshold is not None:
        config_dict['cp_posterior_threshold'] = args.posterior_threshold

    # Statistical testing
    if args.alpha is not None:
        config_dict['alpha'] = args.alpha

    if args.correction:
        config_dict['multiple_testing_correction'] = args.correction

    # Output
    if args.output_dir:
        config_dict['results_base_directory'] = args.output_dir

    # Reconstruct with validation
    config = ChangePointDetectionConfig(**config_dict)

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


    # Run Pipeline
    try:
        orchestrator = ChangePointDetectionOrchestrator(config = config,
                                                        logger = logger,
                                                       )

        logger.info("Running detection pipeline...")
        results      = orchestrator.run()

        # Detection Summary
        logger.info("\n" + "=" * 80)
        logger.info("DETECTION SUMMARY")
        logger.info("=" * 80)

        for method, res in results.items():
            logger.info(f"\nDetector: {method.upper()}")

            if (method == 'pelt'):
                logger.info(f"  Change points detected : {res['n_changepoints']}")
                logger.info(f"  Penalty used           : {res['penalty_used']:.3f}")
                logger.info(f"  Significant CPs        : "
                            f"{res['validation']['n_significant']} / {res['validation']['n_changepoints']}"
                           )

            elif( method == 'bocpd'):
                logger.info(f"  Change points detected : {res['validation']['n_changepoints']}")
                logger.info(f"  Posterior threshold   : {res['validation']['posterior_threshold']}")

                if res.get('hazard_tuning'):
                    logger.info(f"  Hazard tuning method  : {res['hazard_tuning']['method']}")


        # Optional Model Selection
        if config.selection_enabled:
            logger.info("\n" + "=" * 80)
            logger.info("MODEL SELECTION")
            logger.info("=" * 80)

            selector_config = ModelSelectorConfig.parse_file(config.model_selection_config_path)

            selector        = ModelSelector(config = selector_config)
            selection       = selector.select(raw_results = results)

            logger.info(f"Best model : {selection['best_model']}")
            logger.info(f"Ranking    : {selection['ranking']}")

            if selector_config.generate_explanations:
                logger.info("\nExplanation:")
                logger.info(selection['explanation'])
        
        else:
            logger.info("\nModel selection skipped (selection_enabled = False)")

        logger.info("\nResults directory:")
        logger.info(f"  {config.results_base_directory}")

        logger.info("\nDetection completed successfully.")
        
        return 0

    except Exception as exc:
        logger.exception(f"Detection failed: {exc}")
        
        return 1

    finally:
        logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Script Execution
if __name__ == "__main__":
    sys.exit(main())