# Dependencies
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime


# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_util import setup_logger
from src.utils.logging_util import log_section_header
from src.detection.model_selector import ModelSelector
from src.detection.results_saver import DetectionResultsSaver
from config.model_selection_config import ModelSelectorConfig
from config.detection_config import ChangePointDetectionConfig
from src.detection.detector import ChangePointDetectionOrchestrator


# CLI Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description     = "PHQ-9 Change Point Detection (PELT / BOCPD)",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                    )

    # Execution
    parser.add_argument("--execution-mode", choices = ["single", "compare", "ensemble"], default = None, help = "Detection mode: single detector, side-by-side comparison, or ensemble with selection")
    
    parser.add_argument("--detectors", nargs="+", choices = ["pelt", "bocpd"], default = None, help = "Detectors to run")

    # Data
    parser.add_argument("--data", type = Path, default = None, help = "Path to PHQ-9 data CSV")
    
    parser.add_argument("--dataset", type = str, choices = ["exponential", "gamma", "lognormal"], help = "Use predefined dataset (auto-sets data path and output dir)")
    
    parser.add_argument("--all-datasets", action = "store_true", help = "Run on all three datasets (exponential, gamma, lognormal)")

    # PELT
    parser.add_argument("--penalty", type = float, default = None, help = "PELT penalty parameter")
    
    parser.add_argument("--auto-tune-penalty", action = "store_true", help = "Enable BIC-based penalty tuning")
    
    parser.add_argument("--cost-model", choices = ["l1", "l2", "rbf", "ar"], default = None, help = "PELT cost function (if specified, only this model runs)")
    
    parser.add_argument("--min-size", type = int, default = None, help = "Minimum segment size for PELT")

    # BOCPD
    parser.add_argument("--hazard-tuning-method", choices = ["heuristic", "predictive_ll"], default = None, help = "BOCPD hazard tuning method")
    
    parser.add_argument("--hazard-lambda", type = float, default = None, help = "BOCPD hazard lambda (expected run length)")
    
    parser.add_argument("--posterior-threshold", type = float, default = None, help = "BOCPD posterior probability threshold")
    
    parser.add_argument("--bocpd-persistence", type = int, default = None, help = "BOCPD persistence requirement (consecutive timesteps)")

    # Statistical testing
    parser.add_argument("--alpha", type = float, default = None, help = "Significance level for statistical tests")
    
    parser.add_argument("--correction", choices = ["bonferroni", "fdr_bh", "none"], default = None, help = "Multiple testing correction method")

    # Model selection
    parser.add_argument("--select-model", action = "store_true", help = "Enable model selection to identify best model")

    # Output
    parser.add_argument("--output-dir", type = Path, default = None, help = "Output directory for results")
    
    parser.add_argument("--log-level", choices = ["DEBUG", "INFO", "WARNING", "ERROR"], default = "INFO", help = "Logging level")

    return parser.parse_args()


# Helpers
def dump_json(obj, path: Path):
    path.parent.mkdir(parents = True, exist_ok = True)

    with open(path, "w") as f:
        json.dump(obj     = obj,
                  fp      = f,
                  indent  = 4,
                  default = str,
                 )


def run_single_dataset(dataset_name: str, args, logger):
    """
    Run detection pipeline for a single dataset

    Arguments:
    ----------
        dataset_name { str }    : Dataset name (e.g., 'exponential')
        
        args         { Namespace } : Command-line arguments
        
        logger       { Logger }    : Logger instance

    Returns:
    --------
              { dict }          : Model results
    """
    log_section_header(logger, f"PROCESSING DATASET: {dataset_name.upper()}")

    # Base config
    base_cfg = ChangePointDetectionConfig().model_dump()

    # Dataset-specific paths
    if dataset_name:
        base_cfg["data_path"]                   = Path(f"data/raw/synthetic_phq9_data_{dataset_name}.csv")
        base_cfg["results_base_directory"]      = Path(f"results/detection/{dataset_name}")
    
    # Execution
    if args.execution_mode:
        base_cfg["execution_mode"]              = args.execution_mode
    
    elif args.select_model:
        # If --select-model specified but no mode, use compare
        base_cfg["execution_mode"]              = "compare"

    if args.detectors:
        base_cfg["detectors"]                   = args.detectors

    # Data path override
    if args.data:
        base_cfg["data_path"]                   = args.data

    # Output override
    if args.output_dir:
        base_cfg["results_base_directory"]      = args.output_dir

    # PELT
    if args.penalty is not None:
        base_cfg["penalty"]                     = args.penalty

    if args.auto_tune_penalty:
        base_cfg["auto_tune_penalty"]           = True

    if args.cost_model:
        base_cfg["pelt_cost_models"]            = [args.cost_model]

    if args.min_size:
        base_cfg["minimum_segment_size"]        = args.min_size

    # BOCPD
    if args.hazard_tuning_method:
        base_cfg["hazard_tuning_method"]        = args.hazard_tuning_method

    if args.hazard_lambda is not None:
        base_cfg["hazard_lambda"]               = args.hazard_lambda
        base_cfg["auto_tune_hazard"]            = False  # Disable tuning if manual value provided

    if args.posterior_threshold is not None:
        base_cfg["cp_posterior_threshold"]      = args.posterior_threshold

    if args.bocpd_persistence is not None:
        base_cfg["bocpd_persistence"]           = args.bocpd_persistence

    # Statistical testing
    if args.alpha is not None:
        base_cfg["alpha"]                       = args.alpha

    if args.correction:
        base_cfg["multiple_testing_correction"] = args.correction

    # Model selection
    if args.select_model:
        base_cfg["selection_enabled"]           = True

    # Ensemble implies selection
    if (base_cfg["execution_mode"] == "ensemble"):
        base_cfg["selection_enabled"]           = True

    config = ChangePointDetectionConfig(**base_cfg)

    logger.info("Configuration Summary:")
    for k, v in config.get_summary().items():
        logger.info(f"  {k}: {v}")

    # Run detection
    orchestrator = ChangePointDetectionOrchestrator(config = config,
                                                    logger = logger,
                                                   )

    logger.info("Running detection pipeline...")
    results      = orchestrator.run()

    # Model selection
    if config.selection_enabled:
        log_section_header(logger, "MODEL SELECTION")

        selector_cfg = ModelSelectorConfig()
        selector     = ModelSelector(selector_cfg)
        selection    = selector.select(results)

        # Save selection results
        selection_path = config.results_base_directory / "model_selection.json"
        dump_json(obj  = selection, 
                  path = selection_path,
                 )

        best_model   = selection.get("best_model")

        if best_model:
            logger.info(f"✓ Best model selected: {best_model}")
            logger.info(f"  Score: {selection['scores'].get(best_model, 0):.3f}")
            logger.info(f"  Ranking: {selection['ranking']}")
            
            # Save best model using results saver
            results_saver = DetectionResultsSaver(config.results_base_directory)
            results_saver.save_best_model(best_model_id       = best_model,
                                          model_results       = results,
                                          selection_metadata  = selection,
                                         )
        else:
            logger.warning("Model selection returned None (no valid models)")
    
    else:
        logger.info("Model selection skipped (selection_enabled=False)")

    # Summary
    logger.info(f"\nResults saved at: {config.results_base_directory}")
    
    log_section_header(logger, f"SUMMARY: {dataset_name.upper()}")
    
    for model_id, result in results.items():
        n_cps    = result.get('n_changepoints', 0)
        is_valid = result.get('validation', {}).get('overall_significant', False)
        status   = "✓" if is_valid else "✗"
        
        logger.info(f"  {status} {model_id}: {n_cps} CPs")

    return results


# Main
def main():
    args     = parse_arguments()

    # Logging
    logger = setup_logger(module_name = "detection",
                          log_level   = args.log_level,
                          log_dir     = Path("logs"),
                         )

    log_section_header(logger, "PHQ-9 CHANGE POINT DETECTION")
    logger.info(f"Start time: {datetime.now().isoformat()}")

    # Handle multiple datasets
    if args.all_datasets:
        datasets    = ["exponential", "gamma", "lognormal"]
        all_results = {}

        for dataset in datasets:
            try:
                results              = run_single_dataset(dataset_name = dataset,
                                                          args         = args,
                                                          logger       = logger,
                                                         )
                
                all_results[dataset] = results

            except Exception as e:
                logger.error(f"Failed to process {dataset}: {e}", exc_info=True)
                continue

        # Cross-dataset comparison
        log_section_header(logger, "CROSS-DATASET COMPARISON")

        for dataset, results in all_results.items():
            logger.info(f"\n{dataset.upper()}:")
            logger.info(f"  Models run: {len(results)}")
            
            for model_id, result in results.items():
                n_cps = result.get('n_changepoints', 0)
                logger.info(f"    - {model_id}: {n_cps} CPs")

    elif args.dataset:
        # Single predefined dataset
        run_single_dataset(dataset_name = args.dataset,
                           args         = args,
                           logger       = logger,
                          )

    else:
        # Custom data path
        if not args.data:
            logger.error("Must specify either --dataset, --all-datasets, or --data")
            return 1

        run_single_dataset(dataset_name = None,
                           args         = args,
                           logger       = logger,
                          )

    logger.info("Detection completed successfully.")
    logger.info(f"End time: {datetime.now().isoformat()}")

    return 0


# Execution
if __name__ == "__main__":
    sys.exit(main())