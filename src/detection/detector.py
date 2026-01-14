# Dependencies
import numpy as np
import pandas as pd
from typing import Dict
from typing import Optional
import matplotlib.pyplot as plt
from src.detection.pelt_detector import PELTDetector
from src.detection.bocpd_detector import BOCPDDetector
from src.detection.hazard_tuning import BOCPDHazardTuner
from src.detection.visualizations import DetectionVisualizer
from config.detection_config import ChangePointDetectionConfig


# Model styles configurable via config
def get_model_style(model_id: str, idx: int, style_config: Optional[Dict] = None) -> dict:
    """
    Generate consistent colors/linestyles for models

    Arguments:
    ----------
        model_id      { str }  : Model identifier (e.g., 'pelt_l1')

        idx           { int }  : Model index for color selection
        
        style_config  { dict } : Optional style configuration override

    Returns:
    --------
              { dict }         : Style dictionary with 'color' and 'linestyle'
    """
    # Default style configuration
    default_styles = {'pelt'  : {'linestyle': '--', 'color_offset': 0},
                      'bocpd' : {'linestyle': '-', 'color_offset': 5},
                     }

    style_config   = style_config or default_styles
    colors         = plt.cm.tab10.colors

    # Determine detector family
    if ('pelt' in model_id):
        family = 'pelt'

    elif ('bocpd' in model_id):
        family = 'bocpd'

    else:
        # Fallback for unknown families
        return {'color'     : colors[idx % len(colors)],
                'linestyle' : '-',
               }

    family_style = style_config.get(family, {})
    color_offset = family_style.get('color_offset', 0)
    linestyle    = family_style.get('linestyle', '-')

    return {'color'     : colors[(idx + color_offset) % len(colors)],
            'linestyle' : linestyle,
           }



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

        # Model style configuration (can be overridden)
        self.style_config = None


    # DATA HANDLING
    def load_data(self) -> pd.DataFrame:
        """
        Load PHQ-9 data from CSV

        Returns:
        --------
            { pd.DataFrame } : Raw PHQ-9 data (Days x Patients)
        """
        try:
            dataframe = pd.read_csv(filepath_or_buffer = self.config.data_path, 
                                    index_col          = 0,
                                   )

            if self.logger:
                self.logger.info(f"Loaded data: shape={dataframe.shape}")

            return dataframe

        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")

        except Exception as e:
            raise RuntimeError(f"Failed to load data from {self.config.data_path}: {e}")


    def aggregate_cv(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute daily coefficient of variation (CV) with validation of aggregated signal

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

        # ADDED: Validation
        if (len(clean_cv) < 20):
            raise ValueError(f"Aggregated CV signal too short (length={len(clean_cv)}, minimum=20)")

        if (clean_cv.std() < 1e-6):
            raise ValueError("Aggregated CV has near-zero variance - no change points can be detected")

        if self.logger:
            self.logger.info(f"Aggregated CV: length={len(clean_cv)}, mean={clean_cv.mean():.3f}, std={clean_cv.std():.3f}")

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

        # Load and aggregate
        data          = self.load_data()
        aggregated    = self.aggregate_cv(data)

        signal        = aggregated.values
        index         = aggregated.index

        model_results = dict()

        # PELT MODEL GRID
        if ("pelt" in self.config.detectors):
            if self.logger:
                self.logger.info(f"Running PELT with {len(self.config.pelt_cost_models)} cost models...")

            for idx, cost_model in enumerate(self.config.pelt_cost_models):
                model_id = f"pelt_{cost_model}"

                if self.logger:
                    self.logger.info(f"  [{idx+1}/{len(self.config.pelt_cost_models)}] {model_id}")

                try:
                    # Initializing PELT Detector
                    detector                = PELTDetector(config     = self.config, 
                                                           cost_model = cost_model,
                                                          )

                    result                  = detector.detect(aggregated_signal = signal)

                    segments                = detector.extract_segments(aggregated_index = index,
                                                                        change_points    = result["change_points"],
                                                                       )
                    style                   = get_model_style(model_id      = model_id, 
                                                              idx           = idx,
                                                              style_config  = self.style_config,
                                                             )

                    model_results[model_id] = {**result,
                                               "segments" : segments,
                                               **style,
                                              }

                    if self.logger:
                        self.logger.info(f"    → Detected {result['n_changepoints']} change points")

                except Exception as e:
                    if self.logger:
                        self.logger.error(f"    → PELT {cost_model} failed: {e}")
                        
                    continue

        # BOCPD MODEL GRID
        if ("bocpd" in self.config.detectors):
            # Respect user's choice or run both for comparison
            if (self.config.execution_mode == 'single'):
                methods = [self.config.hazard_tuning_method]
            
            else:
                methods = ["heuristic", "predictive_ll"]

            if self.logger:
                self.logger.info(f"Running BOCPD with {len(methods)} hazard tuning methods...")
            
            for idx, method in enumerate(methods):
                model_id = f"bocpd_gaussian_{method}"

                if self.logger:
                    self.logger.info(f"  [{idx+1}/{len(methods)}] {model_id}")

                try:
                    # Hazard tuning
                    hazard_tuning_result = None
                    hazard_lambda        = self.config.hazard_lambda

                    if self.config.auto_tune_hazard:
                        if self.logger:
                            self.logger.info(f"    Tuning hazard (method={method})...")

                        hazard_tuner         = BOCPDHazardTuner(config = self.config, 
                                                                method = method,
                                                               )

                        hazard_tuning_result = hazard_tuner.tune(signal = signal)

                        hazard_lambda        = hazard_tuning_result["optimal_hazard_lambda"]

                        if self.logger:
                            self.logger.info(f"    → Optimal λ = {hazard_lambda:.1f}")

                    # Detection
                    detector                = BOCPDDetector(config = self.config,
                                                            logger = self.logger,
                                                           )

                    result                  = detector.detect(signal        = signal,
                                                              hazard_lambda = hazard_lambda,
                                                             )

                    style                   = get_model_style(model_id     = model_id, 
                                                              idx          = idx + len(self.config.pelt_cost_models),
                                                              style_config = self.style_config,
                                                             )
                    model_results[model_id] = {**result,
                                               "hazard_tuning"      : hazard_tuning_result,
                                               "hazard_lambda_used" : hazard_lambda,
                                               **style,
                                              }

                    if self.logger:
                        self.logger.info(f"    → Detected {result['n_changepoints']} change points")

                    # BOCPD posterior diagnostics
                    if (result.get('run_length_posterior') is not None):
                        self.visualizer.plot_bocpd_posterior(run_length_posterior = result["run_length_posterior"],
                                                             cp_posterior         = result["cp_posterior"],
                                                             posterior_threshold  = self.config.cp_posterior_threshold,
                                                             save_path            = self.config.results_base_directory / "plots" / f"{model_id}_posterior.png",
                                                            )

                except Exception as e:
                    if self.logger:
                        self.logger.error(f"    → BOCPD {method} failed: {e}")
                    continue

        # Check if any models succeeded
        if not model_results:
            raise RuntimeError("All detection models failed - check logs for details")

        # VISUALIZATION
        if self.logger:
            self.logger.info("Generating visualizations...")

        try:
            self.visualizer.plot_aggregated_cv_with_all_models(aggregated_data = aggregated,
                                                               model_results   = model_results,
                                                               save_path       = self.config.results_base_directory  / "plots" / "aggregated_cv_all_models.png",
                                                              )

            self.visualizer.plot_model_comparison_grid(aggregated_data = aggregated,
                                                       model_results   = model_results,
                                                       save_path       = self.config.results_base_directory / "plots" / "model_comparison_grid.png",
                                                      )

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Visualization failed: {e}")

        return model_results