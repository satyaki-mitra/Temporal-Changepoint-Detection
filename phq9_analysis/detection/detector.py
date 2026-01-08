# Dependencies
import numpy as np
import pandas as pd
from typing import Dict
from pathlib import Path
from phq9_analysis.detection.pelt_detector import PELTDetector
from phq9_analysis.detection.bocpd_detector import BOCPDDetector
from phq9_analysis.detection.visualizations import DetectionVisualizer


class ChangePointDetectionOrchestrator:
    """
    Orchestrates change point detection across multiple detectors

    Responsibilities:
    -----------------
    - Load data
    - Compute aggregated statistic
    - Dispatch detectors
    - Collect and align results
    - Drive visualization
    """
    def __init__(self, config):
        """
        Initialize orchestrator

        Arguments:
        ----------
            config { ChangePointDetectionConfig } : Unified detection configuration
        """
        self.config     = config
        self.visualizer = DetectionVisualizer(figsize = config.figure_size,
                                              dpi     = config.dpi,
                                             )


    # Data Handling
    def load_data(self) -> pd.DataFrame:
        """
        Load PHQ-9 data from CSV
        """
        loaded_data = pd.read_csv(filepath_or_buffer = self.config.data_path,
                                  index_col          = 0,
                                 )

        return loaded_data


    def aggregate_cv(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute daily coefficient of variation (CV)

        CV = std / mean
        """
        mean          = data.mean(axis   = 1, 
                                  skipna = True,
                                 )

        std           = data.std(axis   = 1, 
                                 skipna = True,
                                )

        cv            = std / mean.replace(0, np.nan)
        aggregated_cv = cv.dropna()
        
        return aggregated_cv 


    # Detection Pipeline
    def run(self) -> Dict:
        """
        Run detection pipeline according to configuration

        Returns:
        --------
            results { dict } : Raw detection + validation outputs for all detectors
        """
        data       = self.load_data()
        aggregated = self.aggregate_cv(data = data)

        results    = dict()

        # PELT (Offline)
        if ('pelt' in self.config.detectors):
            pelt_detector     = PELTDetector(config = self.config)

            detection_result  = pelt_detector.detect(aggregated_signal = aggregated.values)

            validation_result = pelt_detector.validate(aggregated_signal = aggregated.values,
                                                       change_points     = detection_result['change_points'],
                                                      )

            segments          = pelt_detector.extract_segments(aggregated_index = aggregated.index,
                                                               change_points    = detection_result['change_points'],
                                                              )

            results['pelt']   = {**detection_result,
                                 'validation' : validation_result,
                                 'segments'   : segments,
                                }

        # BOCPD (Online Bayesian)
        if ('bocpd' in self.config.detectors):
            bocpd_detector    = BOCPDDetector(config = self.config)

            detection_result  = bocpd_detector.detect(signal = aggregated.values)

            validation_result = bocpd_detector.validate(cp_posterior = detection_result['cp_posterior'])

            results['bocpd']  = {**detection_result,
                                 'validation' : validation_result,
                                }

        # Visualization (Executor-Level Only)
        self.config.create_output_directories()

        self.visualizer.plot_aggregated_cv(aggregated_data = aggregated,
                                           pelt_cps        = results.get('pelt', {}).get('change_points'),
                                           bocpd_cps       = results.get('bocpd', {}).get('validation', {}).get('indices'),
                                           save_path       = self.config.results_base_directory / 'plots' / 'aggregated_cv.png',
                                          )

        if ('bocpd' in results):
            self.visualizer.plot_bocpd_posterior(run_length_posterior = results['bocpd']['run_length_posterior'],
                                                 cp_posterior         = results['bocpd']['cp_posterior'],
                                                 save_path            = self.config.results_base_directory / 'plots' / 'bocpd_posterior.png',
                                                )

        return results
