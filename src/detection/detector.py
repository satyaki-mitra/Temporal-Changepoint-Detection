# Dependencies
import numpy as np
import pandas as pd
from typing import Dict
import matplotlib.pyplot as plt
from src.detection.pelt_detector import PELTDetector
from src.detection.bocpd_detector import BOCPDDetector
from src.detection.hazard_tuning import BOCPDHazardTuner
from src.detection.visualizations import DetectionVisualizer
from config.detection_config import ChangePointDetectionConfig


# Helper
def get_model_style(model_id: str, idx: int) -> dict:
    """
    Generate consistent colors/linestyles for models
    """
    colors = plt.cm.tab10.colors
    
    if ('pelt' in model_id):
        return {'color'     : colors[idx % len(colors)],
                'linestyle' : '--',
               }

    elif ('bocpd' in model_id):
        return {'color'     : colors[(idx + 5) % len(colors)],
                'linestyle' : '-',
               }
    
    else:
        pass



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
    def __init__(self, config, logger = None):
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
            for idx, cost_model in enumerate(self.config.pelt_cost_models):
                model_id = f"pelt_{cost_model}"

                if self.logger:
                    self.logger.info(f"Running {model_id}")
                
                # Initializing PELT Detector
                detector                = PELTDetector(config     = self.config, 
                                                       cost_model = cost_model,
                                                      )

                result                  = detector.detect(aggregated_signal = signal)

                segments                = detector.extract_segments(aggregated_index = index,
                                                                    change_points    = result["change_points"],
                                                                   )
                style                   = get_model_style(model_id = model_id, 
                                                          idx      = idx,
                                                         )

                model_results[model_id] = {**result,
                                           "segments" : segments,
                                           **style,
                                          }

        # BOCPD MODEL GRID
        if ("bocpd" in self.config.detectors):
            # Respect user's choice or run both for comparison
            if (self.config.execution_mode == 'single'):
                methods = [self.config.hazard_tuning_method]
            
            else:
                methods = ["heuristic", "predictive_ll"]
            
            for idx, method in enumerate(methods):
                model_id = f"bocpd_gaussian_{method}"

                if self.logger:
                    self.logger.info(f"Running {model_id}")

                # Hazard tuning
                hazard_tuning_result = None
                hazard_lambda        = self.config.hazard_lambda

                if self.config.auto_tune_hazard:
                    hazard_tuner         = BOCPDHazardTuner(config = self.config, 
                                                            method = method,
                                                           )

                    hazard_tuning_result = hazard_tuner.tune(signal = signal)

                    hazard_lambda        = hazard_tuning_result["optimal_hazard_lambda"]

                # DON'T reconstruct config - pass hazard_lambda override to detector
                detector                = BOCPDDetector(config = self.config,
                                                        logger = self.logger,
                                                       )

                result                  = detector.detect(signal = signal)

                style                   = get_model_style(model_id = model_id, 
                                                          idx      = idx + len(self.config.pelt_cost_models),
                                                         )
                model_results[model_id] = {**result,
                                           "hazard_tuning" : hazard_tuning_result,
                                           **style,
                                          }

                # BOCPD posterior diagnostics
                self.visualizer.plot_bocpd_posterior(run_length_posterior = result["run_length_posterior"],
                                                     cp_posterior         = result["cp_posterior"],
                                                     posterior_threshold  = self.config.cp_posterior_threshold,
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