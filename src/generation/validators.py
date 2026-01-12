# Dependencies
import numpy as np
import pandas as pd
from typing import List
from typing import Dict
from scipy import stats
from typing import Tuple
from config.clinical_constants import clinical_constants_instance
from config.clinical_constants import get_expected_structural_sparsity


class DataValidator:
    """
    Comprehensive validator for synthetic PHQ-9 data

    This validator is designed specifically for:
    - Sparse and irregular patient observations
    - Patient-level longitudinal trajectories
    - Population-level aggregation (CV, change-point detection readiness)
    - Missingness decomposition (structural vs excess)
    - 12-week endpoint validation (STAR*D-aligned)
    - Enhanced autocorrelation diagnostics

    It validates generated data against clinical literature benchmarks while remaining robust to missingness and non-aligned observation days
    """
    def __init__(self, expected_autocorr_range: Tuple[float, float] = (clinical_constants_instance.EXPECTED_AUTOCORR_SPARSE_LOWER, clinical_constants_instance.EXPECTED_AUTOCORR_SPARSE_UPPER), 
                 expected_baseline_range: Tuple[float, float] = (clinical_constants_instance.BASELINE_VALIDATION_LOWER, clinical_constants_instance.BASELINE_VALIDATION_UPPER),
                 expected_response_rate_range: Tuple[float, float] = (clinical_constants_instance.EXPECTED_RESPONSE_RATE_LOWER, clinical_constants_instance.EXPECTED_RESPONSE_RATE_UPPER), 
                 min_improvement: float = clinical_constants_instance.MIN_CLINICALLY_MEANINGFUL_IMPROVEMENT, 
                 max_autocorr_gap_days: int = clinical_constants_instance.MAX_AUTOCORR_GAP_DAYS,
                 autocorr_weight_halflife: float = clinical_constants_instance.AUTOCORR_WEIGHT_HALFLIFE,  
                 max_autocorr_window_days: int = clinical_constants_instance.MAX_AUTOCORR_WINDOW_DAYS):
        """
        Initialize validator with expected clinical ranges

        Arguments:
        ----------
            expected_autocorr_range      { tuple } : Expected lag-1 autocorrelation range (adjusted for sparse data)

            expected_baseline_range      { tuple } : Expected mean baseline PHQ-9 range
            
            expected_response_rate_range { tuple } : Expected responder proportion (≥50% reduction) at 12 weeks
            
            min_improvement              { float } : Minimum clinically meaningful improvement

            max_autocorr_gap_days         { int }  : Max gap for autocorrelation pairs

            autocorr_weight_halflife     { float } : Half-life for exponential weighting
            
            max_autocorr_window_days      { int }  : Max window for weighted correlation
        """
        self.expected_autocorr_range      = expected_autocorr_range
        self.expected_baseline_range      = expected_baseline_range
        self.expected_response_rate_range = expected_response_rate_range
        self.min_improvement              = min_improvement
        self.max_autocorr_gap_days        = max_autocorr_gap_days
        self.autocorr_weight_halflife     = autocorr_weight_halflife
        self.max_autocorr_window_days     = max_autocorr_window_days

    
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
                                    - relapse statistics
                                    - dropout statistics
        """
        validation = {'overall_valid'      : True,
                      'checks'             : {},
                      'warnings'           : [],
                      'errors'             : [],
                      'relapse_statistics' : {},  
                      'dropout_statistics' : {}
                     }


        self._validate_score_range(data, validation)
        self._validate_autocorrelation(data, validation)
        self._validate_baseline(data, validation)
        self._validate_improvement_12week(data, validation)  
        self._validate_response_rate_12week(data, validation)  
        self._validate_missingness_decomposed(data, validation)  
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

        valid                               = ((clinical_constants_instance.PHQ9_MIN_SCORE <= min_score <= clinical_constants_instance.PHQ9_MAX_SCORE) and 
                                               (clinical_constants_instance.PHQ9_MIN_SCORE <= max_score <= clinical_constants_instance.PHQ9_MAX_SCORE)
                                              )

        validation['checks']['score_range'] = {'min'   : min_score,
                                               'max'   : max_score,
                                               'valid' : valid,
                                              }

        if not valid:
            validation['errors'].append(f"Scores outside PHQ-9 range [{clinical_constants_instance.PHQ9_MIN_SCORE},{clinical_constants_instance.PHQ9_MAX_SCORE}]: min={min_score:.2f}, max={max_score:.2f}")


    def _validate_autocorrelation(self, data: pd.DataFrame, validation: Dict):
        """
        Validate temporal autocorrelation using patient-level gap-aware lag-1 correlations
        
        NEW: Enhanced diagnostics for sparse data validation
        
        Notes:
        ------
        - Operates only on observed sequences (ignores NaNs)
        - Requires at least MIN_OBSERVATIONS_FOR_AUTOCORR observations per patient
        - Uses exponential decay weighting for temporal gaps
        - Reports diagnostic statistics for irregular sampling
        """
        autocorrs     = list()
        temporal_gaps = list()
        valid_pairs   = 0
        total_pairs   = 0
        
        for col in data.columns:
            patient_data = data[col].dropna()
            
            if (len(patient_data) < clinical_constants_instance.MIN_OBSERVATIONS_FOR_AUTOCORR):
                continue
            
            scores = patient_data.values
            days   = [int(idx.split('_')[1]) for idx in patient_data.index]
            
            # Track temporal gap statistics
            for i in range(len(scores) - 1):
                gap          = days[i + 1] - days[i]
                
                temporal_gaps.append(gap)

                total_pairs += 1
                
                # Only use pairs with reasonable temporal proximity
                if (gap <= self.max_autocorr_gap_days):
                    valid_pairs += 1
            
            # Compute gap-aware weighted autocorrelation
            if (len(scores) >= clinical_constants_instance.MIN_OBSERVATIONS_FOR_AUTOCORR):
                weighted_pairs = list()
                
                for i in range(len(scores) - 1):
                    gap = days[i + 1] - days[i]
                    
                    # Exponential decay weight: closer observations get higher weight
                    if (gap <= self.max_autocorr_window_days): 
                        weight = np.exp(-gap / self.autocorr_weight_halflife)
                        weighted_pairs.append((scores[i], scores[i + 1], weight))
                
                if weighted_pairs:
                    s1, s2, weights = zip(*weighted_pairs)
                    s1              = np.array(s1)
                    s2              = np.array(s2)
                    weights         = np.array(weights)
                    
                    # Numerical stability: Normalize weights
                    weights         = weights / weights.sum()
                    # Kish's effective sample size
                    effective_n     = 1.0 / np.sum(weights**2)  
                    
                    if (effective_n < clinical_constants_instance.MIN_OBSERVATIONS_FOR_AUTOCORR):
                        # Too few effective observations
                        continue  
                    
                    # Weighted correlation
                    mean_s1 = np.average(s1, weights = weights)
                    mean_s2 = np.average(s2, weights = weights)
                    
                    cov     = np.average((s1 - mean_s1) * (s2 - mean_s2), weights = weights)
                    var_s1  = np.average((s1 - mean_s1) ** 2, weights = weights)
                    var_s2  = np.average((s2 - mean_s2) ** 2, weights = weights)
                    
                    if ((var_s1 > 0) and (var_s2 > 0)):
                        r_weighted = cov / np.sqrt(var_s1 * var_s2)
                        
                        if (not np.isnan(r_weighted)):
                            autocorrs.append(r_weighted)
        
        # Error handling
        if not autocorrs:
            validation['errors'].append("Autocorrelation could not be computed (insufficient longitudinal data with adequate temporal proximity)")
            return
        
        # Compute summary statistics
        mean_r                                  = float(np.mean(autocorrs))
        median_gap                              = float(np.median(temporal_gaps)) if temporal_gaps else np.nan
        in_range                                = (self.expected_autocorr_range[0] <= mean_r <= self.expected_autocorr_range[1])
        
        # Store results
        validation['checks']['autocorrelation'] = {'mean_gap_aware'          : mean_r,
                                                   'std'                     : float(np.std(autocorrs)),
                                                   'min'                     : float(np.min(autocorrs)),
                                                   'max'                     : float(np.max(autocorrs)),
                                                   'n_patients'              : len(autocorrs),
                                                   'in_expected_range'       : in_range,
                                                   'median_temporal_gap'     : median_gap,
                                                   'valid_lag_pairs_pct'     : float(valid_pairs / total_pairs * 100) if total_pairs > 0 else 0.0,
                                                   'total_observation_pairs' : total_pairs,
                                                   'expected_range'          : f'[{self.expected_autocorr_range[0]:.2f}, {self.expected_autocorr_range[1]:.2f}]'
                                                  }
        
        # Warnings
        if not in_range:
            validation['warnings'].append(f"Gap-aware autocorrelation {mean_r:.3f} outside expected range {self.expected_autocorr_range}. "
                                          f"Literature: Kroenke et al. (2001) test-retest r={clinical_constants_instance.PHQ9_TEST_RETEST_RELIABILITY}. "
                                          f"Note: Sparse sampling reduces observed correlation (median gap = {median_gap:.1f} days)"
                                         )
        
        if (median_gap > 21):
            validation['warnings'].append(f"Median temporal gap between observations is {median_gap:.1f} days (>3 weeks). "
                                          f"Large gaps reduce autocorrelation estimation reliability"
                                         )
            

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
                                            'expected_range'    : f'[{self.expected_baseline_range[0]:.1f}, {self.expected_baseline_range[1]:.1f}]'
                                           }

        if not in_range:
            validation['warnings'].append(f"Baseline mean {mean_b:.2f} outside expected range {self.expected_baseline_range}. Typical RCT enrollment: PHQ-9 15–17.")


    def _validate_improvement_12week(self, data: pd.DataFrame, validation: Dict):
        """
        Validate overall improvement at 12-week endpoint (STAR*D-aligned): uses 12-week observations instead of last observation
        """
        deltas_12week = list()
        deltas_final  = list()
        
        for _, series in data.items():
            scores = series.dropna()
            
            if (len(scores) < 2):
                continue
            
            baseline           = scores.iloc[0]
            
            # Find 12-week observation (closest to day 84)
            days               = [int(idx.split('_')[1]) for idx in scores.index]
            closest_12week_idx = np.argmin([abs(d - clinical_constants_instance.STARD_PRIMARY_ENDPOINT_DAYS) for d in days])
            
            # Only use if within ±2 weeks of target
            if abs(days[closest_12week_idx] - clinical_constants_instance.STARD_PRIMARY_ENDPOINT_DAYS) <= 14:
                score_12week = scores.iloc[closest_12week_idx]
                deltas_12week.append(baseline - score_12week)
            
            # Also track final observation for comparison
            deltas_final.append(baseline - scores.iloc[-1])

        if not deltas_12week:
            validation['warnings'].append("12-week improvement could not be evaluated (insufficient observations near 12-week endpoint).")
            # Fallback to final observations
            deltas_12week = deltas_final  

        mean_delta_12week                   = float(np.mean(deltas_12week))
        mean_delta_final                    = float(np.mean(deltas_final))

        validation['checks']['improvement'] = {'mean_improvement_12week'  : mean_delta_12week,
                                               'mean_improvement_final'   : mean_delta_final,
                                               'std_12week'               : float(np.std(deltas_12week)),
                                               'n_patients_12week'        : len(deltas_12week),
                                               'n_patients_final'         : len(deltas_final),
                                               'shows_improvement'        : mean_delta_12week >= self.min_improvement,
                                               'endpoint'                 : f'{clinical_constants_instance.STARD_PRIMARY_ENDPOINT_WEEKS} weeks ({clinical_constants_instance.STARD_PRIMARY_ENDPOINT_DAYS} days)'
                                              }

        if (mean_delta_12week < self.min_improvement):
            validation['warnings'].append(f"Mean 12-week improvement {mean_delta_12week:.2f} < expected minimum {self.min_improvement:.1f} points. Final improvement: {mean_delta_final:.2f}")


    def _validate_response_rate_12week(self, data: pd.DataFrame, validation: Dict):
        """
        Validate responder proportion (≥50% symptom reduction) at 12-week endpoint: Uses 12-week observations instead of final observations
        """
        responders_12week = list()
        responders_final  = list()

        for _, series in data.items():
            scores = series.dropna()
            
            if ((len(scores) < 2) or (scores.iloc[0] <= 0)):
                continue
            
            baseline           = scores.iloc[0]
            
            # Find 12-week observation
            days               = [int(idx.split('_')[1]) for idx in scores.index]
            closest_12week_idx = np.argmin([abs(d - clinical_constants_instance.STARD_PRIMARY_ENDPOINT_DAYS) for d in days])
            
            if abs(days[closest_12week_idx] - clinical_constants_instance.STARD_PRIMARY_ENDPOINT_DAYS) <= 14:
                score_12week     = scores.iloc[closest_12week_idx]
                reduction_12week = (baseline - score_12week) / baseline
                responders_12week.append(reduction_12week >= 0.50)
            
            # Final observation
            reduction_final = (baseline - scores.iloc[-1]) / baseline
            responders_final.append(reduction_final >= 0.50)

        if not responders_12week:
            validation['warnings'].append("12-week response rate could not be computed (insufficient observations near 12-week endpoint).")
            responders_12week = responders_final

        rate_12week                           = float(np.mean(responders_12week))
        rate_final                            = float(np.mean(responders_final))

        in_range                              = (self.expected_response_rate_range[0] <= rate_12week <= self.expected_response_rate_range[1])

        validation['checks']['response_rate'] = {'rate_12week'             : rate_12week,
                                                 'rate_final'              : rate_final,
                                                 'n_evaluable_12week'      : len(responders_12week),
                                                 'n_evaluable_final'       : len(responders_final),
                                                 'n_responders_12week'     : int(np.sum(responders_12week)),
                                                 'n_responders_final'      : int(np.sum(responders_final)),
                                                 'in_expected_range'       : in_range,
                                                 'expected_range'          : f'[{self.expected_response_rate_range[0]:.0%}, {self.expected_response_rate_range[1]:.0%}]',
                                                 'endpoint'                : f'{clinical_constants_instance.STARD_PRIMARY_ENDPOINT_WEEKS} weeks'
                                                }

        if not in_range:
            validation['warnings'].append(f"12-week response rate {rate_12week:.1%} outside expected range {self.expected_response_rate_range}. STAR*D Level-1 ≈ 47%. Final response rate: {rate_final:.1%}")


    def _validate_missingness_decomposed(self, data: pd.DataFrame, validation: Dict):
        """
        Validate missingness patterns with decomposition into:
        1. Structural sparsity (by design: infrequent assessments)
        2. Excess missingness (dropout + MCAR beyond structural)
        3. Distinguishes intentional sparsity from problematic missingness
        """
        total_missingness                   = float(data.isna().sum().sum() / data.size)
        
        n_patients                          = data.shape[1]
        n_days                              = data.shape[0]
        
        # Expected structural sparsity
        expected_structural                 = get_expected_structural_sparsity(surveys_per_patient = clinical_constants_instance.TYPICAL_SURVEYS_PER_PATIENT,
                                                                               total_days          = n_days,
                                                                              )
        
        # Excess missingness beyond structural
        excess_missingness                  = total_missingness - expected_structural
        
        # Interpretation
        interpretation                      = self._interpret_missingness(excess_missingness)
        
        # Per-patient missingness distribution
        per_patient                         = data.isna().mean(axis = 0)
        
        validation['checks']['missingness'] = {'total_missingness'            : total_missingness,
                                               'structural_sparsity_expected' : expected_structural,
                                               'excess_missingness'           : excess_missingness,
                                               'excess_missingness_pct'       : (excess_missingness / expected_structural * 100) if expected_structural > 0 else 0.0,
                                               'mean_per_patient'             : float(per_patient.mean()),
                                               'std_per_patient'              : float(per_patient.std()),
                                               'min_per_patient'              : float(per_patient.min()),
                                               'max_per_patient'              : float(per_patient.max()),
                                               'interpretation'               : interpretation,
                                              }
        
        # Warning logic
        if not (clinical_constants_instance.EXCESS_MISSINGNESS_TOLERANCE_LOWER <= excess_missingness <= clinical_constants_instance.EXCESS_MISSINGNESS_TOLERANCE_UPPER):
            validation['warnings'].append(f"Excess missingness {excess_missingness:.2%} outside tolerance range "
                                          f"[{clinical_constants_instance.EXCESS_MISSINGNESS_TOLERANCE_LOWER:.1%}, {clinical_constants_instance.EXCESS_MISSINGNESS_TOLERANCE_UPPER:.1%}]. "
                                          f"Expected structural sparsity: {expected_structural:.1%}, observed total: {total_missingness:.1%}. "
                                          f"{interpretation}"
                                         )


    def _interpret_missingness(self, excess: float) -> str:
        """
        Interpret excess missingness clinically: provides actionable guidance based on missingness level
        """
        if (excess < clinical_constants_instance.EXCESS_MISSINGNESS_TOLERANCE_LOWER):
            return "Lower than expected - consider increasing MCAR or dropout rate for realism"
        
        elif (excess > clinical_constants_instance.EXCESS_MISSINGNESS_TOLERANCE_UPPER):
            return "Higher than expected - dropout may be too aggressive or MCAR too high"
        
        else:
            return "Within expected range for sparse longitudinal studies"


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

    status = "PASS" if validation['overall_valid'] else "⚠️  FAIL"
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
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

    print("\n" + "=" * 80)