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
from config.model_selection_config import ModelSelectorConfig
from config.detection_config import ChangePointDetectionConfig
from src.detection.detector import ChangePointDetectionOrchestrator


# CLI Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description     = "PHQ-9 Change Point Detection (PELT / BOCPD)",
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter,
                                    )

    # Execution
    parser.add_argument("--execution-mode", choices = ["single", "compare", "ensemble"], default = None)
    parser.add_argument("--detectors", nargs="+", choices = ["pelt", "bocpd"], default = None)

    # Data
    parser.add_argument("--data", type = Path, default = None)

    # PELT
    parser.add_argument("--penalty", type = float, default = None)
    parser.add_argument("--auto-tune-penalty", action = "store_true")
    parser.add_argument("--cost-model", choices = ["l1", "l2", "rbf", "ar"], default = None)
    parser.add_argument("--min-size", type = int, default = None)

    # BOCPD
    parser.add_argument("--hazard-tuning-method", choices = ["heuristic", "predictive_ll"], default = None)
    parser.add_argument("--posterior-threshold", type = float, default = None)

    # Statistical testing
    parser.add_argument("--alpha", type = float, default = None)
    parser.add_argument("--correction", choices = ["bonferroni", "fdr_bh", "none"], default = None)

    # Output
    parser.add_argument("--output-dir", type = Path, default = None)
    parser.add_argument("--log-level", choices = ["DEBUG", "INFO", "WARNING", "ERROR"], default = "INFO")

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


def promote_best_model(best_model_id: str, all_results: dict, base_dir: Path, logger):
    """
    Promote best model artifacts into best_model/ directory
    """
    best_dir = base_dir / "best_model"
    best_dir.mkdir(parents  = True, 
                   exist_ok = True,
                  )

    result   = all_results[best_model_id]

    logger.info(f"Promoting best model → {best_model_id}")

    # Canonical result
    dump_json(result, best_dir / "model_result.json")

    # Copy plots related to best model
    plots_dir = base_dir / "plots"

    if plots_dir.exists():
        for p in plots_dir.glob(f"*{best_model_id}*"):
            shutil.copy(p, best_dir / p.name)

    # Metadata
    metadata = {"best_model_id"  : best_model_id,
                "method"         : result.get("method"),
                "n_changepoints" : result.get("n_changepoints"),
                "timestamp"      : datetime.now().isoformat(),
               }

    dump_json(metadata, best_dir / "metadata.json")


# Main
def main():
    args     = parse_arguments()

    # Base config
    base_cfg = ChangePointDetectionConfig().model_dump()

    # Execution
    if args.execution_mode:
        base_cfg["execution_mode"] = args.execution_mode

    if args.detectors:
        base_cfg["detectors"] = args.detectors

    # Data
    if args.data:
        base_cfg["data_path"] = args.data

    # PELT
    if args.penalty is not None:
        base_cfg["penalty"] = args.penalty

    if args.auto_tune_penalty:
        base_cfg["auto_tune_penalty"] = True

    if args.cost_model:
        base_cfg["pelt_cost_models"] = [args.cost_model]

    if args.min_size:
        base_cfg["minimum_segment_size"] = args.min_size

    # BOCPD
    if args.hazard_tuning_method:
        base_cfg["hazard_tuning_method"] = args.hazard_tuning_method

    if args.posterior_threshold is not None:
        base_cfg["cp_posterior_threshold"] = args.posterior_threshold

    # Statistical testing
    if args.alpha is not None:
        base_cfg["alpha"] = args.alpha

    if args.correction:
        base_cfg["multiple_testing_correction"] = args.correction

    # Output
    if args.output_dir:
        base_cfg["results_base_directory"] = args.output_dir

    # Ensemble implies selection
    if (base_cfg["execution_mode"] == "ensemble"):
        base_cfg["selection_enabled"] = True

    config = ChangePointDetectionConfig(**base_cfg)

    # Logging
    logger = setup_logger(module_name = "detection",
                          log_level   = args.log_level,
                          log_dir     = Path("logs"),
                         )

    log_section_header(logger, "PHQ-9 CHANGE POINT DETECTION")
    logger.info(f"Start time: {datetime.now().isoformat()}")

    logger.info("Configuration Summary:")
    for k, v in config.get_summary().items():
        logger.info(f"  {k}: {v}")

    # Run detection
    orchestrator = ChangePointDetectionOrchestrator(config = config,
                                                    logger = logger,
                                                   )

    logger.info("Running detection pipeline...")
    results      = orchestrator.run()

    # Persist all model outputs (includes:
    # - tuning results
    # - statistical tests
    # - validation summaries
    # - segments
    dump_json(obj  = results, 
              path = config.results_base_directory / "all_model_results.json",
             )

    # Model selection
    if config.selection_enabled:
        log_section_header(logger, "MODEL SELECTION")

        selector_cfg = ModelSelectorConfig.parse_file(config.model_selection_config_path)

        selector     = ModelSelector(selector_cfg)
        selection    = selector.select(results)

        dump_json(obj  = selection, 
                  path = config.results_base_directory / "model_selection.json",
                 )

        best_model   = selection.get("best_model")
        
        logger.info(f"Best model selected: {best_model}")

        promote_best_model(best_model_id = best_model,
                           all_results   = results,
                           base_dir      = config.results_base_directory,
                           logger        = logger,
                          )
    else:
        logger.info("Model selection skipped (selection_enabled=False)")

    logger.info(f"Results saved at: {config.results_base_directory}")
    logger.info("Detection completed successfully.")
    logger.info(f"End time: {datetime.now().isoformat()}")

    return 0


# Execution
if __name__ == "__main__":
    sys.exit(main())