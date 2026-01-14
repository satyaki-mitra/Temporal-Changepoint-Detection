# Dependencies
import numpy as np
import pandas as pd
from typing import Dict
from scipy import stats
from typing import Optional
from config.eda_constants import eda_constants_instance


class EDADataValidator:
    """
    Validate PHQ-9 data quality for EDA analysis: checks data integrity, missingness, and basic statistical properties
    """
    def __init__(self):
        """
        Initialize validator
        """
        pass
    

    def validate_all(self, data: pd.DataFrame, metadata: Optional[Dict] = None) -> Dict:
        """
        Run all validation checks
        
        Arguments:
        ----------
            data     { pd.DataFrame } : PHQ-9 data (Days x Patients)
            
            metadata { dict }         : Optional metadata for enhanced validation
        
        Returns:
        --------
            { dict } : Comprehensive validation report
        """
        validation = {'overall_valid' : True,
                      'checks'        : {},
                      'warnings'      : [],
                      'errors'        : [],
                     }
        
        self._validate_shape(data, validation)
        self._validate_score_range(data, validation)
        self._validate_missingness(data, validation)
        self._validate_sufficient_data(data, validation)
        self._validate_outliers(data, validation)
        
        # Enhanced validation with metadata
        if metadata is not None:
            self._validate_against_metadata(data, metadata, validation)
        
        validation['overall_valid'] = ((len(validation['errors']) == 0) and (len(validation['warnings']) <= 3))
        
        return validation
    

    def _validate_shape(self, data: pd.DataFrame, validation: Dict):
        """
        Validate data shape
        """
        n_days, n_patients = data.shape
        
        validation['checks']['shape'] = {'n_days'     : n_days,
                                         'n_patients' : n_patients,
                                         'valid'      : True,
                                        }
        
        if (n_days < eda_constants_instance.MIN_DAYS_FOR_ANALYSIS):
            validation['errors'].append(f"Insufficient days: {n_days} < {eda_constants_instance.MIN_DAYS_FOR_ANALYSIS}")
            validation['overall_valid'] = False
    

    def _validate_score_range(self, data: pd.DataFrame, validation: Dict):
        """
        Ensure all scores are in valid PHQ-9 range [0, 27]
        """
        scores                              = data.values.flatten()
        scores                              = scores[~np.isnan(scores)]
        min_score                           = float(np.min(scores))
        max_score                           = float(np.max(scores))
        valid                               = ((0 <= min_score <= 27) and (0 <= max_score <= 27))
        
        validation['checks']['score_range'] = {'min'   : min_score,
                                               'max'   : max_score,
                                               'valid' : valid,
                                              }
        
        if not valid:
            validation['errors'].append(f"Scores outside [0, 27]: min={min_score:.2f}, max={max_score:.2f}")
    

    def _validate_missingness(self, data: pd.DataFrame, validation: Dict):
        """
        Check missingness patterns
        """
        total_missingness                   = float(data.isna().sum().sum() / data.size)
        
        validation['checks']['missingness'] = {'total_rate' : total_missingness,
                                               'valid'      : (total_missingness <= eda_constants_instance.MAX_MISSINGNESS_RATE),
                                              }
        
        if (total_missingness > eda_constants_instance.MAX_MISSINGNESS_RATE):
            validation['warnings'].append(f"High missingness: {total_missingness:.1%} > {eda_constants_instance.MAX_MISSINGNESS_RATE:.1%}")
    

    def _validate_sufficient_data(self, data: pd.DataFrame, validation: Dict):
        """
        Check if sufficient observations for analysis
        """
        total_obs                            = int(data.notna().sum().sum())
        
        validation['checks']['observations'] = {'total'  : total_obs,
                                                'valid'  : (total_obs >= eda_constants_instance.MIN_OBSERVATIONS_TOTAL),
                                               }
        
        if (total_obs < eda_constants_instance.MIN_OBSERVATIONS_TOTAL):
            validation['errors'].append(f"Insufficient observations: {total_obs} < {eda_constants_instance.MIN_OBSERVATIONS_TOTAL}")
    

    def _validate_outliers(self, data: pd.DataFrame, validation: Dict):
        """
        Detect outliers using Z-score method
        """
        scores                           = data.values.flatten()
        scores                           = scores[~np.isnan(scores)]
        z_scores                         = np.abs(stats.zscore(scores, nan_policy = 'omit'))
        z_scores                         = z_scores[~np.isnan(z_scores)]
        outliers                         = np.sum(z_scores > eda_constants_instance.OUTLIER_Z_SCORE_THRESHOLD)
        outlier_rate                     = float(outliers / len(z_scores)) if (len(z_scores) > 0) else 0.0
        
        validation['checks']['outliers'] = {'count' : int(outliers),
                                            'rate'  : outlier_rate,
                                            'valid' : outlier_rate <= eda_constants_instance.OUTLIER_RATE_ACCEPTABLE,
                                           }
        
        if (outlier_rate > eda_constants_instance.OUTLIER_RATE_ACCEPTABLE):
            validation['warnings'].append(f"High outlier rate: {outlier_rate:.2%} (threshold: {eda_constants_instance.OUTLIER_RATE_ACCEPTABLE:.0%})")
    

    def _validate_against_metadata(self, data: pd.DataFrame, metadata: Dict, validation: Dict):
        """
        Enhanced validation using generation metadata
        """
        study_design                     = metadata.get('study_design', {})
        gen_stats                        = metadata.get('generation_statistics', {})
        
        # Check shape consistency
        expected_days                    = study_design.get('total_days')
        expected_patients                = study_design.get('n_patients')
        
        observed_days, observed_patients = data.shape
        
        if (expected_days is not None) and (observed_days != expected_days):
            validation['warnings'].append(f"Days mismatch: expected {expected_days}, observed {observed_days}")
        
        if (expected_patients is not None) and (observed_patients != expected_patients):
            validation['warnings'].append(f"Patients mismatch: expected {expected_patients}, observed {observed_patients}")
        
        # Check missingness consistency
        expected_missingness = gen_stats.get('missingness_rate')
        
        if expected_missingness is not None:
            observed_missingness = float(data.isna().sum().sum() / data.size)
            difference           = abs(observed_missingness - expected_missingness)
            
            if (difference > 0.01):
                validation['warnings'].append(f"Missingness difference: expected {expected_missingness:.3f}, observed {observed_missingness:.3f}")


def validate_clustering_results(labels: np.ndarray, data: pd.DataFrame, n_clusters: int) -> Dict:
    """
    Validate clustering quality
    
    Arguments:
    ----------
        labels     { np.ndarray }  : Cluster labels
        
        data       { pd.DataFrame }: PHQ-9 data
        
        n_clusters { int }         : Number of clusters
    
    Returns:
    --------
               { dict }            : Validation results
    """
    validation = {'valid'    : True,
                  'warnings' : [],
                  'checks'   : {},
                 }
    
    # Check cluster sizes
    unique, counts                        = np.unique(labels, return_counts = True)
    
    min_size                              = int(np.min(counts))
    max_size                              = int(np.max(counts))
    
    validation['checks']['cluster_sizes'] = {'min'    : min_size,
                                             'max'    : max_size,
                                             'counts' : dict(zip(unique.tolist(), counts.tolist())),
                                            }
    
    # Flag very small clusters
    if (min_size < eda_constants_instance.MIN_DAYS_PER_CLUSTER):
        validation['warnings'].append(f"Small cluster detected: {min_size} days < {eda_constants_instance.MIN_DAYS_PER_CLUSTER}")
    
    # Check imbalance
    if (max_size > 0):
        imbalance = max_size / min_size
        
        validation['checks']['imbalance'] = imbalance
        
        if ((min_size / max_size) < (1.0 - eda_constants_instance.MAX_CLUSTER_SIZE_IMBALANCE)):
            validation['warnings'].append(f"Cluster size imbalance: {imbalance:.1f}x")
    
    return validation