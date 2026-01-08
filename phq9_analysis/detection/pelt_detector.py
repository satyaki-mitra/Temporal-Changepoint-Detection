# Dependencies
import numpy as np
import pandas as pd
import ruptures as rpt
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from phq9_analysis.detection.penalty_tuning import tune_penalty_bic
from phq9_analysis.detection.statistical_tests import PELTStatisticalValidator


class PELTDetector:
    """
    Offline change point detector using the PELT algorithm

    Responsibilities:
    -----------------
    - Run PELT on aggregated statistics
    - Perform penalty tuning (BIC)
    - Validate change points via frequentist tests
    - Extract segment / cluster boundaries

    This class is algorithm-specific and does NOT:
    - Load data
    - Perform aggregation
    - Generate plots
    - Save files
    """
    def __init__(self, config):
        """
        Initialize PELT detector

        Arguments:
        ----------
            config { ChangePointDetectionConfig } : Unified detection configuration
        """
        self.config = config


    def detect(self, aggregated_signal: np.ndarray) -> Dict:
        """
        Detect change points using PELT

        Arguments:
        ----------
            aggregated_signal { np.ndarray } : Aggregated 1D statistic (e.g., daily CV)

        Returns:
        --------
                     { dict }                : Detection results including change points and tuning metadata
        """
        tuning_results = None

        if self.config.auto_tune_penalty:
            optimal_penalty, tuning_results = tune_penalty_bic(signal        = aggregated_signal,
                                                               cost_model    = self.config.cost_model,
                                                               min_size      = self.config.minimum_segment_size,
                                                               jump          = self.config.jump,
                                                               penalty_range = self.config.penalty_range,
                                                              )

            penalty_used                    = optimal_penalty
        
        else:
            penalty_used = self.config.penalty

        # Initialize PELT algorithm
        pelt_algorithm = rpt.Pelt(model    = self.config.cost_model,
                                  min_size = self.config.minimum_segment_size,
                                  jump     = self.config.jump,
                                 )

        # Fit PELT on the aggregated signal
        pelt_algorithm.fit(aggregated_signal)

        # Detect change points (ruptures includes terminal index)
        change_points  = pelt_algorithm.predict(pen = penalty_used)

        # Remove terminal endpoint
        change_points  = [cp for cp in change_points if (cp < len(aggregated_signal))]

        return {'method'         : 'pelt',
                'variant'        : f"pelt_{self.config.cost_model}",
                'change_points'  : change_points,
                'penalty_used'   : float(penalty_used),
                'tuning_results' : tuning_results,
               }


    def validate(self, aggregated_signal: np.ndarray, change_points: List[int]) -> Dict:
        """
        Validate detected change points using frequentist statistics

        Arguments:
        ----------
            aggregated_signal { np.ndarray } : Aggregated statistic
            
            change_points     { list }       : Detected change points

        Returns:
        --------
                      { dict }               : Validation results
        """
        pelt_validator          = PELTStatisticalValidator(alpha                 = self.config.alpha,
                                                           correction_method     = self.config.multiple_testing_correction,
                                                           effect_size_threshold = self.config.effect_size_threshold,
                                                          )

        validated_change_points = pelt_validator.validate_all_changepoints(signal        = aggregated_signal,
                                                                           change_points = change_points,
                                                                          )

        return validated_change_points


    def extract_segments(self, aggregated_index: pd.Index, change_points: List[int]) -> Dict:
        """
        Extract segment boundaries from change points

        Arguments:
        ----------
            aggregated_index { pd.Index } : Index of aggregated series (time)
            
            change_points    { list }     : Change point indices
 
        Returns:
        --------
                       { dict }           : Segment metadata
        """
        segments = dict()
        prev     = 0

        for i, cp in enumerate(change_points):
            if (cp == 0):
                continue

            end                        = min(cp - 1, len(aggregated_index) - 1)

            segments[f'Segment_{i+1}'] = {'start_index' : int(prev),
                                          'end_index'   : int(end),
                                          'start_time'  : aggregated_index[prev],
                                          'end_time'    : aggregated_index[end],
                                          'length'      : int(cp - prev),
                                         }

            prev                       = cp

        return segments