# Dependencies
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
from config.eda_constants import eda_constants_instance


class ResponsePatternAnalyzer:
    """
    Analyze and classify patient response patterns from PHQ-9 trajectories
    
    Classifies patients into:
    - Early responders (fast improvement in first 6 weeks)
    - Gradual responders (steady improvement over 12 weeks)
    - Late responders (slow/delayed improvement)
    - Non-responders (minimal improvement)
    """
    def __init__(self, logger = None):
        """
        Initialize analyzer
        
        Arguments:
        ----------
            logger { Logger } : Optional logger instance
        """
        self.logger = logger
    

    def analyze_all_patients(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze response patterns for all patients
        
        Arguments:
        ----------
            data { pd.DataFrame } : PHQ-9 data (Days x Patients)
        
        Returns:
        --------
            { pd.DataFrame }      : Patient-level response pattern classification
        """
        results = list()
        
        for patient_col in data.columns:
            patient_scores = data[patient_col].dropna()
            
            if (len(patient_scores) < eda_constants_instance.MIN_OBSERVATIONS_FOR_TRAJECTORY):
                continue
            
            # Extract patient ID from column name
            patient_id             = patient_col.replace('Patient_', '')
            
            # Analyze trajectory
            analysis               = self._analyze_patient_trajectory(patient_scores)
            analysis['patient_id'] = patient_id
            
            results.append(analysis)
        
        results_dataframe = pd.DataFrame(data = results)
        
        return results_dataframe
    

    def _analyze_patient_trajectory(self, scores: pd.Series) -> Dict:
        """
        Analyze single patient trajectory
        
        Arguments:
        ----------
            scores { pd.Series } : PHQ-9 scores over time
        
        Returns:
        --------
            { dict } : Trajectory analysis results
        """
        # Extract day indices
        days            = np.array([int(idx.split('_')[1]) for idx in scores.index])
        score_values    = scores.values
        
        # Basic statistics
        baseline        = float(score_values[0])
        final_score     = float(score_values[-1])
        improvement     = baseline - final_score
        
        # Calculate improvement percentage
        improvement_pct = (improvement / baseline) if baseline > 0 else 0.0
        
        # Fit linear trend
        if (len(scores) >= eda_constants_instance.MIN_OBSERVATIONS_FOR_TRAJECTORY):
            slope, intercept, r_value, _, _ = stats.linregress(days, score_values)
        
        else:
            slope   = 0.0
            r_value = 0.0
        
        # Classify response pattern
        response_pattern                = self._classify_response_pattern(improvement_pct = improvement_pct, 
                                                                          slope           = slope, 
                                                                          days            = days, 
                                                                          scores          = score_values,
                                                                         )
        
        # Detect plateau
        plateau_detected, plateau_start = self._detect_plateau(days   = days, 
                                                               scores = score_values,
                                                              )
        
        return {'baseline'          : baseline,
                'final_score'       : final_score,
                'improvement'       : improvement,
                'improvement_pct'   : improvement_pct,
                'slope'             : slope,
                'r_squared'         : r_value ** 2,
                'response_pattern'  : response_pattern,
                'plateau_detected'  : plateau_detected,
                'plateau_start_day' : plateau_start,
                'n_observations'    : len(scores),
                'duration_days'     : int(days[-1] - days[0]),
               }
    

    def _classify_response_pattern(self, improvement_pct: float, slope: float, days: np.ndarray, scores: np.ndarray) -> str:
        """
        Classify response pattern based on improvement and trajectory
        
        Arguments:
        ----------
            improvement_pct { float }      : Total improvement percentage
            
            slope           { float }      : Trajectory slope
            
            days            { np.ndarray } : Day indices
            
            scores          { np.ndarray } : Score values
        
        Returns:
        --------
                    { str }                : Response pattern classification
        """
        # Non-responder: < 20% improvement OR minimal negative slope
        if ((improvement_pct < eda_constants_instance.MINIMAL_IMPROVEMENT_THRESHOLD) or 
            (slope > eda_constants_instance.TRAJECTORY_SLOPE_MINIMAL_THRESHOLD)):
            return 'non_responder'
        
        # Responder categories based on slope and timing
        if (slope <= eda_constants_instance.TRAJECTORY_SLOPE_FAST_THRESHOLD):
            # Fast improvement = early responder
            return 'early_responder'
        
        elif (slope <= eda_constants_instance.TRAJECTORY_SLOPE_SLOW_THRESHOLD):
            # Check if improvement occurred early or late
            if (len(days) >= 84):  # 12 weeks
                early_improvement = self._calculate_early_improvement(days, scores)
                
                if (early_improvement > 0.30): 
                    # >30% improvement in first 12 weeks
                    return 'gradual_responder'
                
                else:
                    return 'late_responder'
            
            else:
                return 'gradual_responder'
        
        else:
            # Slow slope but responder
            return 'late_responder'
    

    def _calculate_early_improvement(self, days: np.ndarray, scores: np.ndarray) -> float:
        """
        Calculate improvement in first 12 weeks
        
        Arguments:
        ----------
            days   { np.ndarray } : Day indices
            
            scores { np.ndarray } : Score values
        
        Returns:
        --------
                { float }         : Improvement percentage in first 84 days
        """
        # Find observations in first 12 weeks
        early_mask = (days <= (eda_constants_instance.GRADUAL_RESPONSE_WEEKS * 7))
        
        if (np.sum(early_mask) < 2):
            return 0.0
        
        early_scores  = scores[early_mask]
        baseline      = early_scores[0]
        final_early   = early_scores[-1]
        
        improvement   = (baseline - final_early) / baseline if baseline > 0 else 0.0
        
        return improvement
    

    def _detect_plateau(self, days: np.ndarray, scores: np.ndarray) -> Tuple[bool, Optional[int]]:
        """
        Detect plateau phase in trajectory, where plateau criteria:
        - Low variance in rolling window
        - Near-zero slope
        
        Arguments:
        ----------
            days   { np.ndarray } : Day indices
            
            scores { np.ndarray } : Score values
        
        Returns:
        --------
               { tuple }          : (plateau_detected, plateau_start_day)
        """
        if (len(scores) < eda_constants_instance.MIN_OBSERVATIONS_FOR_TRAJECTORY):
            return False, None
        
        # Use rolling window if enough observations
        window_size = min(4, len(scores) // 2)
        
        if (window_size < 3):
            return False, None
        
        # Calculate rolling variance and slope
        for i in range(len(scores) - window_size + 1):
            window_scores = scores[i:i + window_size]
            window_days   = days[i:i + window_size]
            
            # Check variance
            variance      = np.var(window_scores)
            
            if (variance > eda_constants_instance.PLATEAU_VARIANCE_THRESHOLD):
                continue
            
            # Check slope
            if (len(window_scores) >= 3):
                slope, _, _, _, _ = stats.linregress(window_days, window_scores)
                
                if (abs(slope) > eda_constants_instance.PLATEAU_SLOPE_THRESHOLD):
                    continue
            
            # Plateau detected
            return True, int(window_days[0])
        
        return False, None
    

    def compare_to_metadata(self, observed_patterns: pd.DataFrame, metadata: Dict) -> Dict:
        """
        Compare observed response patterns to expected from metadata
        
        Arguments:
        ----------
            observed_patterns { pd.DataFrame } : Results from analyze_all_patients()
            
            metadata          { dict }         : Generation metadata
        
        Returns:
        --------
                        { dict }               : Comparison results
        """
        # Count observed patterns
        observed_counts = observed_patterns['response_pattern'].value_counts().to_dict()
        
        # Get expected from metadata
        expected_counts = metadata.get('response_pattern_distribution', {})
        
        if not expected_counts:
            return {'comparison_available' : False,
                    'reason'               : 'No expected patterns in metadata',
                   }
        
        # Calculate proportions
        total_observed = len(observed_patterns)
        total_expected = sum(expected_counts.values())
        
        comparison     = {'patterns' : {},
                          'overall_match_score' : 0.0,
                         }
        
        for pattern in ['early_responder', 'gradual_responder', 'late_responder', 'non_responder']:
            obs_count                       = observed_counts.get(pattern, 0)
            exp_count                       = expected_counts.get(pattern, 0)
            obs_pct                         = (obs_count / total_observed * 100) if total_observed > 0 else 0.0
            exp_pct                         = (exp_count / total_expected * 100) if total_expected > 0 else 0.0
            difference                      = abs(obs_pct - exp_pct)
            
            comparison['patterns'][pattern] = {'observed_count' : obs_count,
                                               'expected_count' : exp_count,
                                               'observed_pct'   : obs_pct,
                                               'expected_pct'   : exp_pct,
                                               'difference_pct' : difference,
                                              }
        
        # Calculate overall match score (100 - average difference)
        avg_diff                          = np.mean([p['difference_pct'] for p in comparison['patterns'].values()])
        comparison['overall_match_score'] = max(0.0, 100.0 - avg_diff)
        
        return comparison


class RelapseDetector:
    """
    Detect relapse events (sudden score increases) in patient trajectories
    """
    def __init__(self, threshold: float = None):
        """
        Initialize detector
        
        Arguments:
        ----------
            threshold { float } : Score increase threshold for relapse detection
        """
        self.threshold = threshold or eda_constants_instance.RELAPSE_SCORE_INCREASE_THRESHOLD
    

    def detect_relapses(self, data: pd.DataFrame) -> Dict:
        """
        Detect relapses across all patients
        
        Arguments:
        ----------
            data { pd.DataFrame } : PHQ-9 data (Days x Patients)
        
        Returns:
        --------
            { dict } : Relapse detection results
        """
        total_relapses         = 0
        patients_with_relapses = 0
        relapse_details        = list()
        
        for patient_col in data.columns:
            patient_scores = data[patient_col].dropna()
            
            if (len(patient_scores) < 2):
                continue
            
            # Extract day indices
            days             = np.array([int(idx.split('_')[1]) for idx in patient_scores.index])
            score_values     = patient_scores.values
            
            # Detect relapses for this patient
            patient_relapses = self._detect_patient_relapses(days, score_values)
            
            if (len(patient_relapses) > 0):
                patients_with_relapses += 1
                total_relapses         += len(patient_relapses)
                patient_id              = patient_col.replace('Patient_', '')
                
                for relapse in patient_relapses:
                    relapse_details.append({'patient_id' : patient_id,
                                            'day'        : relapse['day'],
                                            'magnitude'  : relapse['magnitude'],
                                           })
        
        n_patients = data.shape[1]
        
        return {'total_relapses'            : total_relapses,
                'patients_with_relapses'    : patients_with_relapses,
                'relapse_rate'              : patients_with_relapses / n_patients if n_patients > 0 else 0.0,
                'mean_relapses_per_patient' : total_relapses / n_patients if n_patients > 0 else 0.0,
                'relapse_details'           : relapse_details,
               }
    

    def _detect_patient_relapses(self, days: np.ndarray, scores: np.ndarray) -> List[Dict]:
        """
        Detect relapses for single patient
        
        Arguments:
        ----------
            days   { np.ndarray } : Day indices
            
            scores { np.ndarray } : Score values
        
        Returns:
        --------
               { list }           : List of relapse events
        """
        relapses = list()
        
        for i in range(len(scores) - 1):
            gap       = days[i + 1] - days[i]
            
            # Only consider reasonable gaps
            if ((gap < eda_constants_instance.RELAPSE_DETECTION_MIN_GAP_DAYS) or (gap > eda_constants_instance.RELAPSE_DETECTION_MAX_GAP_DAYS)):
                continue
            
            # Check for score increase
            increase = scores[i + 1] - scores[i]
            
            if (increase >= self.threshold):
                relapses.append({'day'       : int(days[i + 1]),
                                 'magnitude' : float(increase),
                                })
        
        return relapses