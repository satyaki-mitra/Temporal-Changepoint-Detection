# Dependencies
import numpy as np
import pandas as pd
from typing import List
from typing import Dict
from scipy import stats
from typing import Tuple


class DataValidator:
    """
    Comprehensive validator for synthetic PHQ-9 data

    This validator is designed specifically for:
    - Sparse and irregular patient observations
    - Patient-level longitudinal trajectories
    - Population-level aggregation (CV, change-point detection readiness)

    It validates generated data against clinical literature benchmarks while remaining robust to missingness and non-aligned observation days
    """
    def __init__(self, expected_autocorr_range: Tuple[float, float] = (0.6, 0.85), expected_baseline_range: Tuple[float, float] = (14.0, 18.0),
                 expected_response_rate_range: Tuple[float, float] = (0.40, 0.70), min_improvement: float = 3.0):
        """
        Initialize validator with expected clinical ranges

        Arguments:
        ----------
            expected_autocorr_range      { tuple } : Expected lag-1 autocorrelation range

            expected_baseline_range      { tuple } : Expected mean baseline PHQ-9 range
            
            expected_response_rate_range { tuple } : Expected responder proportion (≥50% reduction)
            
            min_improvement              { float } : Minimum clinically meaningful improvement
        """
        self.expected_autocorr_range      = expected_autocorr_range
        self.expected_baseline_range      = expected_baseline_range
        self.expected_response_rate_range = expected_response_rate_range
        self.min_improvement              = min_improvement

    
    def validate_all(self, data: pd.DataFrame) -> Dict:
        """
        Run all validation checks on generated PHQ-9 data

        Arguments:
        ----------
            data { pd.DataFrame } : PHQ-9 data (rows = observation days, columns = patients)

        Returns:
        --------
            { dict }              : Structured validation report containing:
                                    - overall_valid
                                    - checks
                                    - warnings
                                    - errors
        """
        validation = {'overall_valid' : True,
                      'checks'        : {},
                      'warnings'      : [],
                      'errors'        : [],
                     }


        self._validate_score_range(data, validation)
        self._validate_autocorrelation(data, validation)
        self._validate_baseline(data, validation)
        self._validate_improvement(data, validation)
        self._validate_response_rate(data, validation)
        self._validate_missingness(data, validation)
        self._validate_distributions(data, validation)

        validation['overall_valid'] = ((len(validation['errors']) == 0) and (len(validation['warnings']) <= 3))

        return validation


    def _validate_score_range(self, data: pd.DataFrame, validation: Dict):
        """
        Ensure all PHQ-9 scores lie within the valid range [0, 27]
        """
        scores                              = data.to_numpy().ravel()
        scores                              = scores[~np.isnan(scores)]

        min_score                           = float(np.min(scores))
        max_score                           = float(np.max(scores))

        valid                               = ((0.0 <= min_score <= 27.0) and (0.0 <= max_score <= 27.0))

        validation['checks']['score_range'] = {'min'   : min_score,
                                               'max'   : max_score,
                                               'valid' : valid,
                                              }

        if not valid:
            validation['errors'].append(f"Scores outside PHQ-9 range [0,27]: min={min_score:.2f}, max={max_score:.2f}")


    def _validate_autocorrelation(self, data: pd.DataFrame, validation: Dict):
        """
        Validate temporal autocorrelation using patient-level lag-1 correlations
        
        Notes:
        ------
        - Operates only on observed sequences (ignores NaNs)
        - Requires at least 3 observations per patient
        - Reports diagnostic statistics for irregular sampling
        """
        autocorrs           = list()
        naive_autocorrs     = list()
        temporal_gaps       = list()
        valid_lag_pairs     = 0
        total_pairs         = 0
        
        for col in data.columns:
            patient_data = data[col].dropna()
            
            if (len(patient_data) < 3):
                continue
            
            scores       = patient_data.values
            days         = [int(idx.split('_')[1]) for idx in patient_data.index]
            
            # Naive autocorrelation (observation sequence, ignoring gaps)
            r_naive      = np.corrcoef(scores[:-1], scores[1:])[0, 1]
            
            if not np.isnan(r_naive):
                naive_autocorrs.append(r_naive)
            
            # Gap-aware autocorrelation (only pairs with reasonable temporal proximity)
            for i in range(len(scores) - 1):
                gap          = days[i + 1] - days[i]
                temporal_gaps.append(gap)

                total_pairs += 1
                
                # Only use pairs with lag â‰¤ 7 days for autocorrelation estimation
                if (gap <= 7):
                    valid_lag_pairs += 1
            
            # Compute weighted autocorrelation for this patient
            if (len(scores) >= 3):
                weighted_pairs = []
                
                for i in range(len(scores) - 1):
                    gap = days[i + 1] - days[i]
                    
                    # Exponential decay weight: closer observations get higher weight
                    if (gap <= 14):  # Only consider gaps up to 2 weeks
                        weight = np.exp(-gap / 7.0)  # Half-life of 7 days
                        weighted_pairs.append((scores[i], scores[i + 1], weight))
                
                if weighted_pairs:
                    s1, s2, weights = zip(*weighted_pairs)
                    s1              = np.array(s1)
                    s2              = np.array(s2)
                    weights         = np.array(weights)
                    
                    # Weighted correlation
                    mean_s1         = np.average(s1, weights = weights)
                    mean_s2         = np.average(s2, weights = weights)
                    
                    cov             = np.average((s1 - mean_s1) * (s2 - mean_s2), weights = weights)
                    var_s1          = np.average((s1 - mean_s1) ** 2, weights = weights)
                    var_s2          = np.average((s2 - mean_s2) ** 2, weights = weights)
                    
                    if ((var_s1 > 0) and (var_s2 > 0)):
                        r_weighted = cov / np.sqrt(var_s1 * var_s2)
                        
                        if not np.isnan(r_weighted):
                            autocorrs.append(r_weighted)
        
        # Error handling
        if not autocorrs:
            validation['errors'].append("Autocorrelation could not be computed (insufficient longitudinal data with adequate temporal proximity).")
            return
        
        # Compute summary statistics
        mean_r                                  = float(np.mean(autocorrs))
        mean_r_naive                            = float(np.mean(naive_autocorrs)) if naive_autocorrs else np.nan
        median_gap                              = float(np.median(temporal_gaps)) if temporal_gaps else np.nan
        
        in_range                                = (self.expected_autocorr_range[0] <= mean_r <= self.expected_autocorr_range[1])
        
        # Store comprehensive results
        validation['checks']['autocorrelation'] = {'mean_gap_aware'          : mean_r,
                                                   'mean_naive'              : mean_r_naive,
                                                   'std'                     : float(np.std(autocorrs)),
                                                   'min'                     : float(np.min(autocorrs)),
                                                   'max'                     : float(np.max(autocorrs)),
                                                   'n_patients'              : len(autocorrs),
                                                   'in_expected_range'       : in_range,
                                                   'median_temporal_gap'     : median_gap,
                                                   'valid_lag_pairs_pct'     : float(valid_lag_pairs / total_pairs * 100) if total_pairs > 0 else 0.0,
                                                   'total_observation_pairs' : total_pairs,
                                                  }
        
        # Warnings
        if not in_range:
            validation['warnings'].append(f"Gap-aware autocorrelation {mean_r:.3f} outside expected range {self.expected_autocorr_range}. Literature: Kroenke et al. (2001) test-retest r=0.84. Note: Sparse sampling may reduce observed correlation.")
        
        if (median_gap > 14):
            validation['warnings'].append(f"Median temporal gap between observations is {median_gap:.1f} days. Large gaps reduce autocorrelation estimation reliability.")
        
        if (mean_r_naive > mean_r + 0.15):
            validation['warnings'].append(f"Naive autocorrelation ({mean_r_naive:.3f}) substantially exceeds gap-aware estimate ({mean_r:.3f}). This suggests temporal gaps are not properly accounted for in naive calculation.")


    def _validate_baseline(self, data: pd.DataFrame, validation: Dict):
        """
        Validate baseline severity using the first observed score per patient

        Baseline is NOT assumed to occur on the same calendar day for all patients
        """
        baselines                        = [series.dropna().iloc[0] for _, series in data.items() if series.notna().sum() >= 1]

        baselines                        = np.array(baselines)

        mean_b                           = float(np.mean(baselines))

        in_range                         = (self.expected_baseline_range[0] <= mean_b <= self.expected_baseline_range[1])

        validation['checks']['baseline'] = {'mean'              : mean_b,
                                            'std'               : float(np.std(baselines)),
                                            'min'               : float(np.min(baselines)),
                                            'max'               : float(np.max(baselines)),
                                            'n_patients'        : len(baselines),
                                            'in_expected_range' : in_range,
                                           }

        if not in_range:
            validation['warnings'].append(f"Baseline mean {mean_b:.2f} outside expected range {self.expected_baseline_range}. Typical RCT enrollment: PHQ-9 15–17.")


    def _validate_improvement(self, data: pd.DataFrame, validation: Dict):
        """
        Validate overall improvement using patient-level first vs last observation
        """
        deltas = list()

        for _, series in data.items():
            scores = series.dropna()
            
            if (len(scores) >= 2):
                deltas.append(scores.iloc[0] - scores.iloc[-1])

        if not deltas:
            validation['errors'].append("Improvement could not be evaluated (insufficient data).")
            return

        mean_delta                          = float(np.mean(deltas))

        validation['checks']['improvement'] = {'mean_improvement'  : mean_delta,
                                               'std'               : float(np.std(deltas)),
                                               'min'               : float(np.min(deltas)),
                                               'max'               : float(np.max(deltas)),
                                               'n_patients'        : len(deltas),
                                               'shows_improvement' : mean_delta >= self.min_improvement,
                                              }

        if (mean_delta < self.min_improvement):
            validation['warnings'].append(f"Mean improvement {mean_delta:.2f} < expected minimum {self.min_improvement:.1f} points.")


    def _validate_response_rate(self, data: pd.DataFrame, validation: Dict):
        """
        Validate responder proportion (≥50% symptom reduction)
        """
        responders = list()

        for _, series in data.items():
            scores = series.dropna()
            if ((len(scores) >= 2) and (scores.iloc[0] > 0)):
                reduction = (scores.iloc[0] - scores.iloc[-1]) / scores.iloc[0]
                responders.append(reduction >= 0.50)

        if not responders:
            validation['errors'].append("Response rate could not be computed.")
            return

        rate                                  = float(np.mean(responders))

        in_range                              = (self.expected_response_rate_range[0] <= rate <= self.expected_response_rate_range[1])

        validation['checks']['response_rate'] = {'rate'                 : rate,
                                                 'n_evaluable_patients' : len(responders),
                                                 'n_responders'         : int(np.sum(responders)),
                                                 'in_expected_range'    : in_range,
                                                }

        if not in_range:
            validation['warnings'].append(f"Response rate {rate:.1%} outside expected range {self.expected_response_rate_range}. STAR*D Level-1 ≈ 47%.")


    def _validate_missingness(self, data: pd.DataFrame, validation: Dict):
        """
        Validate missingness patterns at population and patient levels
        """
        missing_rate                        = float(data.isna().sum().sum() / data.size)

        per_patient                         = data.isna().mean(axis=0)

        validation['checks']['missingness'] = {'overall_rate'     : missing_rate,
                                               'mean_per_patient' : float(per_patient.mean()),
                                               'std_per_patient'  : float(per_patient.std()),
                                               'min_per_patient'  : float(per_patient.min()),
                                               'max_per_patient'  : float(per_patient.max()),
                                              }

        if not (0.10 <= missing_rate <= 0.40):
            validation['warnings'].append(f"Missingness rate {missing_rate:.1%} outside typical depression trial range (10–40%).")


    def _validate_distributions(self, data: pd.DataFrame, validation: Dict):
        """
        Validate distributional characteristics of PHQ-9 scores

        - PHQ-9 is bounded and ordinal → normality is NOT expected
        - Shapiro-Wilk used only as a diagnostic indicator
        """
        scores                                = data.to_numpy().ravel()
        scores                                = scores[~np.isnan(scores)]

        skew                                  = float(stats.skew(scores))
        kurt                                  = float(stats.kurtosis(scores))

        # Shapiro only if sample is small enough and variable
        p_value                               = np.nan
        
        if ((len(scores) <= 5000) and (np.std(scores) > 0)):
            _, p_value = stats.shapiro(scores)

        validation['checks']['distributions'] = {'mean'              : float(np.mean(scores)),
                                                 'std'               : float(np.std(scores)),
                                                 'skewness'          : skew,
                                                 'kurtosis'          : kurt,
                                                 'normality_p_value' : float(p_value) if not np.isnan(p_value) else None,
                                                }



# Convenience utilities
def validate_against_literature(data: pd.DataFrame) -> Dict:
    """
    Convenience wrapper for validating PHQ-9 data against literature benchmarks
    """
    validator = DataValidator()
    return validator.validate_all(data)


def print_validation_report(validation: Dict):
    """
    Print a formatted, human-readable validation report
    """
    print("=" * 80)
    print("DATA VALIDATION REPORT")
    print("=" * 80)

    status = "PASS" if validation['overall_valid'] else "FAIL"
    print(f"\nOverall Status: {status}")

    if validation['errors']:
        print("\nERRORS")
        print("-" * 80)
        for e in validation['errors']:
            print(f"  {e}")

    if validation['warnings']:
        print("\nWARNINGS")
        print("-" * 80)
        for w in validation['warnings']:
            print(f"  {w}")

    print("\nDETAILED CHECKS")
    print("-" * 80)

    for name, results in validation['checks'].items():
        print(f"\n{name.upper().replace('_', ' ')}:")
        for k, v in results.items():
            print(f"  {k}: {v}")

    print("\n" + "=" * 80)