# Dependencies
import numpy as np
import pandas as pd
from typing import Dict
from typing import List
from scipy import stats
from typing import Tuple


class DataValidator:
    """
    Comprehensive validator for synthetic PHQ-9 data
    
    - Checks data quality against clinical literature benchmarks and provides detailed diagnostic information
    """
    def __init__(self, expected_autocorr_range: Tuple[float, float] = (0.6, 0.8), expected_baseline_range: Tuple[float, float] = (14.0, 20.0),
                 expected_response_rate_range: Tuple[float, float] = (0.40, 0.70), min_improvement: float = 3.0):
        """
        Initialize validator with expected ranges
        
        Arguments:
        ----------
            expected_autocorr_range      { tuple } : Expected (min, max) for lag-1 autocorrelation

            expected_baseline_range      { tuple } : Expected (min, max) for mean baseline scores
            
            expected_response_rate_range { tuple } : Expected (min, max) for response rate
            
            min_improvement              { float } : Minimum expected improvement (baseline - endpoint)
        """
        self.expected_autocorr_range      = expected_autocorr_range
        self.expected_baseline_range      = expected_baseline_range
        self.expected_response_rate_range = expected_response_rate_range
        self.min_improvement              = min_improvement
    

    def validate_all(self, data: pd.DataFrame) -> Dict:
        """
        Run all validation checks
        
        Arguments:
        ----------
            data { pd.DataFrame } : Generated PHQ-9 DataFrame (Days x Patients)
        
        Returns:
        --------
                 { dict }         : Dictionary with validation results
        """
        validation = {'overall_valid' : True,
                      'checks'        : {},
                      'warnings'      : [],
                      'errors'        : [],
                     }
        
        # Score range
        self._validate_score_range(data, validation)
        
        # Temporal autocorrelation
        self._validate_autocorrelation(data, validation)
        
        # Baseline severity
        self._validate_baseline(data, validation)
        
        # Improvement trend
        self._validate_improvement(data, validation)
        
        # Response rate
        self._validate_response_rate(data, validation)
        
        # Missingness pattern
        self._validate_missingness(data, validation)
        
        # Distribution properties
        self._validate_distributions(data, validation)
        
        # Set overall validity
        validation['overall_valid'] = ((len(validation['errors']) == 0) and (len(validation['warnings']) <= 2))
        
        return validation
    

    def _validate_score_range(self, data: pd.DataFrame, validation: Dict):
        """
        Check all scores are in valid PHQ-9 range [0, 27]
        """
        all_scores                          = data.values.flatten()
        all_scores                          = all_scores[~np.isnan(all_scores)]
        
        min_score                           = np.min(all_scores)
        max_score                           = np.max(all_scores)
        
        validation['checks']['score_range'] = {'min'   : float(min_score),
                                               'max'   : float(max_score),
                                               'valid' : (min_score >= 0) and (max_score <= 27),
                                              }
        
        if not validation['checks']['score_range']['valid']:
            validation['errors'].append(f"Scores outside valid range [0, 27]: min={min_score:.2f}, max={max_score:.2f}")

    
    def _validate_autocorrelation(self, data: pd.DataFrame, validation: Dict):
        """
        Check temporal autocorrelation matches expected range
        """
        autocorrs = list()
        
        for col in data.columns:
            scores = data[col].dropna()
            if (len(scores) > 1):
                # Calculate lag-1 autocorrelation
                autocorr = scores.autocorr(lag = 1)

                if not np.isnan(autocorr):
                    autocorrs.append(autocorr)
        
        if autocorrs:
            mean_autocorr                           = np.mean(autocorrs)
            std_autocorr                            = np.std(autocorrs)
            
            validation['checks']['autocorrelation'] = {'mean'              : float(mean_autocorr),
                                                       'std'               : float(std_autocorr),
                                                       'min'               : float(np.min(autocorrs)),
                                                       'max'               : float(np.max(autocorrs)),
                                                       'n_patients'        : len(autocorrs),
                                                       'in_expected_range' : (self.expected_autocorr_range[0] <= mean_autocorr <= self.expected_autocorr_range[1],)
                                                      }
            
            if not validation['checks']['autocorrelation']['in_expected_range']:
                validation['warnings'].append(f"Mean autocorrelation {mean_autocorr:.3f} outside expected range {self.expected_autocorr_range}. Literature: Kroenke et al. (2001) r=0.84")
        
        else:
            validation['errors'].append("Could not calculate autocorrelation (insufficient data)")
    

    def _validate_baseline(self, data: pd.DataFrame, validation: Dict):
        """
        Check baseline severity is in expected range
        """
        # Get first observation for each patient
        baseline_scores                  = data.iloc[0].dropna()
        
        baseline_mean                    = baseline_scores.mean()
        baseline_std                     = baseline_scores.std()
        
        validation['checks']['baseline'] = {'mean'              : float(baseline_mean),
                                            'std'               : float(baseline_std),
                                            'min'               : float(baseline_scores.min()),
                                            'max'               : float(baseline_scores.max()),
                                            'n_patients'        : len(baseline_scores),
                                            'in_expected_range' : (self.expected_baseline_range[0] <= baseline_mean <= self.expected_baseline_range[1]),
                                           }
        
        if not validation['checks']['baseline']['in_expected_range']:
            validation['warnings'].append(f"Baseline mean {baseline_mean:.2f} outside expected range {self.expected_baseline_range}. Typical moderate-severe depression trials: 15-17")
    

    def _validate_improvement(self, data: pd.DataFrame, validation: Dict):
        """
        Check overall improvement trend
        """
        # Compare early vs late period
        n_days                              = len(data)
        early_days                          = min(10, n_days // 4)
        late_days                           = min(10, n_days // 4)
        
        early_mean                          = data.iloc[:early_days].mean(axis=1).mean()
        late_mean                           = data.iloc[-late_days:].mean(axis=1).mean()
        
        improvement                         = early_mean - late_mean
        improvement_pct                     = (improvement / early_mean * 100) if (early_mean > 0) else 0
        
        validation['checks']['improvement'] = {'early_mean'           : float(early_mean),
                                               'late_mean'            : float(late_mean),
                                               'absolute_improvement' : float(improvement),
                                               'percent_improvement'  : float(improvement_pct),
                                               'shows_improvement'    : (improvement > self.min_improvement),
                                              }
        
        if not validation['checks']['improvement']['shows_improvement']:
            validation['warnings'].append(f"Insufficient improvement: {improvement:.2f} points (expected ≥{self.min_improvement:.1f})")

    
    def _validate_response_rate(self, data: pd.DataFrame, validation: Dict):
        """
        Check proportion of patients with ≥50% symptom reduction
        """
        responders = list()
        
        for col in data.columns:
            scores = data[col].dropna()
            
            if (len(scores) >= 2):
                baseline = scores.iloc[0]
                endpoint = scores.iloc[-1]
                
                if (baseline > 0):
                    reduction_pct = (baseline - endpoint) / baseline
                    responders.append(reduction_pct >= 0.50)
        
        if responders:
            response_rate                         = np.mean(responders)
            
            validation['checks']['response_rate'] = {'rate'                 : float(response_rate),
                                                     'n_evaluable_patients' : len(responders),
                                                     'n_responders'         : int(np.sum(responders)),
                                                     'in_expected_range'    : (self.expected_response_rate_range[0] <= response_rate <= self.expected_response_rate_range[1]),
                                                    }
            
            if not validation['checks']['response_rate']['in_expected_range']:
                validation['warnings'].append(f"Response rate {response_rate:.1%} outside expected range {self.expected_response_rate_range}. Literature: STAR*D Level 1 ~47%")

        else:
            validation['errors'].append("Could not calculate response rate (insufficient data)")

    
    def _validate_missingness(self, data: pd.DataFrame, validation: Dict):
        """
        Check missingness pattern is realistic
        """
        total_cells                         = data.size
        missing_cells                       = data.isna().sum().sum()
        missingness_rate                    = missing_cells / total_cells
        
        # Calculate per-patient missingness
        patient_missingness                 = data.isna().sum(axis=0) / len(data)
        
        validation['checks']['missingness'] = {'overall_rate'     : float(missingness_rate),
                                               'mean_per_patient' : float(patient_missingness.mean()),
                                               'std_per_patient'  : float(patient_missingness.std()),
                                               'min_per_patient'  : float(patient_missingness.min()),
                                               'max_per_patient'  : float(patient_missingness.max()),
                                              }
        
        # Check if missingness is reasonable (15-30% typical)
        if ((missingness_rate < 0.10) or (missingness_rate > 0.40)):
            validation['warnings'].append(f"Missingness rate {missingness_rate:.1%} unusual (typical range: 15-30% for depression trials)")

    
    def _validate_distributions(self, data: pd.DataFrame, validation: Dict):
        """
        Check distributional properties
        """
        all_scores                            = data.values.flatten()
        all_scores                            = all_scores[~np.isnan(all_scores)]
        
        # Normality test (should NOT be normal due to bounded range)
        _, p_value_normality                  = stats.shapiro(all_scores[:5000])  # Sample for efficiency
        
        # Calculate skewness and kurtosis
        skewness                              = stats.skew(all_scores)
        kurtosis                              = stats.kurtosis(all_scores)
        
        validation['checks']['distributions'] = {'mean'              : float(np.mean(all_scores)),
                                                 'std'               : float(np.std(all_scores)),
                                                 'skewness'          : float(skewness),
                                                 'kurtosis'          : float(kurtosis),
                                                 'normality_p_value' : float(p_value_normality),
                                                }




def validate_against_literature(data: pd.DataFrame) -> Dict:
    """
    Convenience function to validate data against literature
    
    Arguments:
    ----------
        data { pd.DataFrame } : Generated PHQ-9 DataFrame
    
    Returns:
    --------
             { dict }         : Validation results dictionary
    """
    validator = DataValidator()
    return validator.validate_all(data)


def print_validation_report(validation: Dict):
    """
    Print formatted validation report
    
    Arguments:
    ----------
        validation { dict } : Validation results from validate_all()
    """
    print("=" * 80)
    print("DATA VALIDATION REPORT")
    print("=" * 80)
    
    # Overall status
    status = "PASS" if validation['overall_valid'] else "FAIL"
    print(f"\nOverall Status: {status}")
    
    # Errors
    if validation['errors']:
        print(f"\n{'─' * 80}")
        print("ERRORS")
        print('─' * 80)
        for error in validation['errors']:
            print(f"  {error}")
    
    # Warnings
    if validation['warnings']:
        print(f"\n{'─' * 80}")
        print("WARNINGS")
        print('─' * 80)
        for warning in validation['warnings']:
            print(f"  {warning}")
    
    # Detailed checks
    print(f"\n{'─' * 80}")
    print("DETAILED CHECKS")
    print('─' * 80)
    
    for check_name, check_results in validation['checks'].items():
        print(f"\n{check_name.upper().replace('_', ' ')}:")
        for key, value in check_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    print("\n" + "=" * 80)
