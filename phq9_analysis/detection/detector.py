# Dependencies
import numpy as np
import pandas as pd
from typing import Dict
from phq9_analysis.detection.pelt_detector import PELTDetector
from phq9_analysis.detection.bocpd_detector import BOCPDDetector
from phq9_analysis.detection.hazard_tuning import BOCPDHazardTuner
from phq9_analysis.detection.visualizations import DetectionVisualizer


class ChangePointDetectionOrchestrator:
    """
    Orchestrates change point detection across a configurable model grid

    Design principles:
    ------------------
    - Model-driven (not detector-driven)
    - Supports full corpus, subsets, or single models
    - Deterministic, reproducible execution
    - Visualization-ready outputs
    """
    def __init__(self, config, logger=None):
        """
        Initialize detection orchestrator

        Arguments:
        ----------
            config { ChangePointDetectionConfig } : Unified detection configuration
           
            logger    { logging.Logger | None }   : Optional logger
        """
        self.config     = config
        self.logger     = logger

        self.visualizer = DetectionVisualizer(figsize = config.figure_size,
                                              dpi     = config.dpi,
                                             )


    # DATA HANDLING
    def load_data(self) -> pd.DataFrame:
        """
        Load PHQ-9 data from CSV

        Returns:
        --------
            { pd.DataFrame } : Raw PHQ-9 data 
        """
        dataframe = pd.read_csv(filepath_or_buffer = self.config.data_path, 
                                index_col          = 0,
                               )

        return dataframe


    def aggregate_cv(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute daily coefficient of variation (CV)

        CV = std / mean
        """
        mean     = data.mean(axis   = 1, 
                             skipna = True,
                            )

        std      = data.std(axis   = 1, 
                            skipna = True,
                          )

        cv       = std / mean.replace(0, np.nan)

        clean_cv = cv.dropna()

        return clean_cv


    # MAIN PIPELINE
    def run(self) -> Dict:
        """
        Run change point detection for all configured models

        Returns:
        --------
            results { dict } : model_id -> detection + validation + tuning outputs
        """
        self.config.create_output_directories()

        data          = self.load_data()
        aggregated    = self.aggregate_cv(data)

        signal        = aggregated.values
        index         = aggregated.index

        model_results = dict()

        # PELT MODEL GRID
        if ("pelt" in self.config.detectors):
            for cost_model in self.config.pelt_cost_models:
                model_id = f"pelt_{cost_model}"

                if self.logger:
                    self.logger.info(f"Running {model_id}")

                cfg                     = self.config.copy(update = {"cost_model": cost_model})
                detector                = PELTDetector(config = cfg)

                result                  = detector.detect(signal = signal)

                segments                = detector.extract_segments(aggregated_index = index,
                                                                    change_points    = result["change_points"],
                                                                   )

                model_results[model_id] = {**result,
                                           "segments"  : segments,
                                           "color"     : "red",
                                           "linestyle" : "--",
                                          }

        # BOCPD MODEL GRID
        if ("bocpd" in self.config.detectors):
            for method in ["heuristic", "predictive_ll"]:
                model_id = f"bocpd_gaussian_{method}"

                if self.logger:
                    self.logger.info(f"Running {model_id}")

                cfg                  = self.config.copy()

                hazard_tuning_result = None

                if cfg.auto_tune_hazard:
                    hazard_tuner         = BOCPDHazardTuner(config = cfg, 
                                                            method = method,
                                                           )

                    hazard_tuning_result = hazard_tuner.tune(signal = signal)
                    cfg.hazard_lambda    = hazard_tuning_result["optimal_hazard_lambda"]

                # Detect CPs
                detector                = BOCPDDetector(config = cfg, 
                                                        logger = self.logger,
                                                       )

                result                  = detector.detect(signal = signal)

                validation              = detector.validate(result["cp_posterior"])

                model_results[model_id] = {**result,
                                           "validation"    : validation,
                                           "hazard_tuning" : hazard_tuning_result,
                                           "change_points" : validation["indices"],
                                           "color"         : "blue" if (method == "heuristic") else "green",
                                           "linestyle"     : "-" if (method == "heuristic") else ":",
                                          }

                # BOCPD posterior diagnostics
                self.visualizer.plot_bocpd_posterior(run_length_posterior = result["run_length_posterior"],
                                                     cp_posterior         = result["cp_posterior"],
                                                     posterior_threshold  = cfg.cp_posterior_threshold,
                                                     save_path            = self.config.results_base_directory / "plots" / f"{model_id}_posterior.png",
                                                    )

        # VISUALIZATION
        self.visualizer.plot_aggregated_cv_with_all_models(aggregated_data = aggregated,
                                                           model_results   = model_results,
                                                           save_path       = self.config.results_base_directory  / "plots" / "aggregated_cv_all_models.png",
                                                          )

        self.visualizer.plot_model_comparison_grid(aggregated_data = aggregated,
                                                   model_results   = model_results,
                                                   save_path       = self.config.results_base_directory / "plots" / "model_comparison_grid.png",
                                                  )

        return model_results