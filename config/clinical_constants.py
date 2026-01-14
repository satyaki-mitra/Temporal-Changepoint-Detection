# Dependencies
from dataclasses import dataclass


@dataclass(frozen = True)
class ClinicalConstants:
    """
    Clinical Constants for PHQ-9 data generation
    """
    # PHQ-9 SCORE BOUNDARIES
    PHQ9_MIN_SCORE                        : float = 0.0
    PHQ9_MAX_SCORE                        : float = 27.0

    # CLINICAL TRIAL PARAMETERS (STAR*D Protocol)
    STARD_DROPOUT_RATE                    : float = 0.21   # Level 1 actual dropout
    STARD_RESPONSE_RATE                   : float = 0.47   # ≥50% symptom reduction
    STARD_REMISSION_RATE                  : float = 0.28   # PHQ-9 ≤5
    STARD_PRIMARY_ENDPOINT_WEEKS          : int   = 12     # Standard trial duration
    STARD_PRIMARY_ENDPOINT_DAYS           : int   = 84     # 12 weeks in days

    # TEMPORAL AUTOCORRELATION PARAMETERS
    PHQ9_TEST_RETEST_RELIABILITY          : float = 0.84   # 2-day interval
    EXPECTED_AUTOCORR_SPARSE_LOWER        : float = 0.30   # Adjusted for sparse sampling
    EXPECTED_AUTOCORR_SPARSE_UPPER        : float = 0.70   # More realistic for 14-30 day gaps
    MAX_AUTOCORR_GAP_DAYS                 : int   = 14     # Maximum gap for correlation pairs
    AUTOCORR_WEIGHT_HALFLIFE              : float = 10.0   # Exponential decay half-life (days)
    MAX_AUTOCORR_WINDOW_DAYS              : int   = 21     # Maximum temporal window

    # BASELINE SEVERITY PARAMETERS
    TYPICAL_RCT_BASELINE_MEAN             : float = 16.0   # Moderate-severe depression trials
    TYPICAL_RCT_BASELINE_STD              : float = 3.0    # Population heterogeneity
    BASELINE_VALIDATION_LOWER             : float = 13.0   # Relaxed for real-world variation
    BASELINE_VALIDATION_UPPER             : float = 19.0   # Upper bound for moderate-severe

    # TREATMENT RESPONSE PARAMETERS
    TYPICAL_RECOVERY_RATE_MEAN            : float = -0.075 # Points per day (negative = improvement)
    TYPICAL_RECOVERY_RATE_STD             : float = 0.03   # Treatment response heterogeneity
    MIN_CLINICALLY_MEANINGFUL_IMPROVEMENT : float = 3.0    # Conservative MCID estimate

    

    # Plateau Timing (weeks after treatment start)
    EARLY_RESPONDER_PLATEAU_WEEKS         : int   = 6
    GRADUAL_RESPONDER_PLATEAU_WEEKS       : int   = 10
    LATE_RESPONDER_PLATEAU_WEEKS          : int   = 16

    # MEASUREMENT NOISE & VARIABILITY
    PHQ9_MCID                             : float = 5.0     # Minimal clinically important difference
    MEASUREMENT_NOISE_STD                 : float = 2.5     # Must be < MCID
    MEASUREMENT_NOISE_MIN                 : float = 1.0     # Individual minimum
    MEASUREMENT_NOISE_MAX                 : float = 4.0     # Individual maximum

    # RELAPSE PARAMETERS 
    RELAPSE_PROBABILITY_DAILY             : float = 0.10    # Per observation day
    RELAPSE_MAGNITUDE_MEAN                : float = 3.5     # Exponential distribution mean
    RELAPSE_MAGNITUDE_GAMMA_SHAPE         : float = 2.0     # For gamma distribution
    RELAPSE_MAGNITUDE_LOGNORMAL_SIGMA     : float = 0.5     # For lognormal distribution

    # MISSINGNESS PARAMETERS
    DROPOUT_RATE_LOWER                    : float = 0.15    # Lower bound (best-case scenario)
    DROPOUT_RATE_UPPER                    : float = 0.30    # Upper bound (real-world studies)
    DROPOUT_EXPONENTIAL_SCALE_FACTOR      : float = 0.3     # Scale relative to study duration
    DROPOUT_MINIMUM_OFFSET_DAYS           : int   = 60      # Ensure some follow-up before dropout

    MCAR_MISSINGNESS_RATE                 : float = 0.08    # Random missed appointments
    MCAR_RATE_UPPER_BOUND                 : float = 0.15    # Maximum reasonable MCAR

    # SURVEY SCHEDULING PARAMETERS
    MIN_SURVEYS_PER_PATIENT               : int   = 10      # Minimum for adequate trajectory
    MAX_SURVEYS_PER_PATIENT               : int   = 20      # Maximum realistic frequency
    TYPICAL_SURVEYS_PER_PATIENT           : int   = 15      # For sparsity calculations

    # VALIDATION THRESHOLDS
    MIN_OBSERVATIONS_FOR_AUTOCORR         : int   = 3       # Minimum patient observations
    BOUNDARY_REFLECTION_FACTOR            : float = 0.5     # For soft boundary handling
    BOUNDARY_HIT_WARNING_THRESHOLD        : int   = 5       # Consecutive boundary hits

    # Missingness decomposition tolerances
    EXCESS_MISSINGNESS_TOLERANCE_LOWER    : float = -0.05   # Allow 5% below expected
    EXCESS_MISSINGNESS_TOLERANCE_UPPER    : float = 0.10    # Allow 10% above expected

    # Response rate validation (12-week endpoint)
    EXPECTED_RESPONSE_RATE_LOWER          : float = 0.40    # Conservative lower bound
    EXPECTED_RESPONSE_RATE_UPPER          : float = 0.70    # Optimistic upper bound

    # TRAJECTORY MODELING PARAMETERS
    AR_COEFFICIENT_MIN                    : float = 0.5     # Minimum autocorrelation
    AR_COEFFICIENT_MAX                    : float = 0.9     # Maximum autocorrelation
    AR_COEFFICIENT_DEFAULT                : float = 0.70    # Default value
    AR_COEFFICIENT_INDIVIDUAL_STD         : float = 0.05    # Individual variation

    # Plateau modeling
    PLATEAU_NOISE_REDUCTION_FACTOR        : float = 0.5     # Reduce noise during plateau
    PLATEAU_TRANSITION_WEEKS              : int   = 2       # Smooth transition period

    # Baseline clipping parameters
    BASELINE_CLIP_LOWER_STD_MULTIPLIER    : float = 2.0     # For symmetric clipping
    BASELINE_CLIP_UPPER_LIMIT             : float = 27.0    # Hard upper bound

    # SAMPLE SIZE CONSTRAINTS
    MIN_PATIENTS                          : int   = 100     # Minimum study size
    MAX_PATIENTS                          : int   = 5000    # Maximum for computational efficiency
    MIN_STUDY_DAYS                        : int   = 180     # 6 months minimum
    MAX_STUDY_DAYS                        : int   = 730     # 2 years maximum



# SINGLETON INSTANCE FOR PARAMETER CLASS
clinical_constants_instance    = ClinicalConstants()

# PHQ-9 Severity Thresholds
PHQ9_SEVERITY_THRESHOLDS       = {'minimal'           : (0, 4),
                                  'mild'              : (5, 9),
                                  'moderate'          : (10, 14),
                                  'moderately_severe' : (15, 19),
                                  'severe'            : (20, 27),                            
                                 }

# Response Pattern Probabilities (must sum to 1.0)
RESPONSE_PATTERN_PROBABILITIES = {'early_responder'   : 0.30,    # Respond within 4 weeks
                                  'gradual_responder' : 0.35,    # Respond by 8-12 weeks
                                  'late_responder'    : 0.15,    # Respond after 12 weeks
                                  'non_responder'     : 0.20     # <50% improvement
                                 }



# UTILITY FUNCTIONS
def get_severity_category(score: float) -> str:
    """
    Classify PHQ-9 score into severity category
    
    Arguments:
    ----------
        score { float } : PHQ-9 score [0, 27]
    
    Returns:
    --------
        { str } : Severity category name
    """
    if (not (clinical_constants_instance.PHQ9_MIN_SCORE <= score <= clinical_constants_instance.PHQ9_MAX_SCORE)):
        raise ValueError(f"Score {score} outside valid range [{clinical_constants_instance.PHQ9_MIN_SCORE}, {clinical_constants_instance.PHQ9_MAX_SCORE}]")
    
    for category, (lower, upper) in PHQ9_SEVERITY_THRESHOLDS.items():
        if (lower <= score <= upper):
            return category
    
    return 'unknown'


def validate_response_pattern_probabilities() -> bool:
    """
    Validate that response pattern probabilities sum to 1.0
    
    Returns:
    --------
        { bool } : True if valid, raises ValueError otherwise
    """
    total = sum(RESPONSE_PATTERN_PROBABILITIES.values())
    
    # Allow floating-point tolerance
    if not (0.99 <= total <= 1.01):  
        raise ValueError(f"Response pattern probabilities must sum to 1.0, got {total:.3f}. Current values: {RESPONSE_PATTERN_PROBABILITIES}")
    
    return True


def get_expected_structural_sparsity(surveys_per_patient: int, total_days: int) -> float:
    """
    Calculate expected structural sparsity (by design)
    
    Arguments:
    ----------
        surveys_per_patient { int } : Average surveys per patient
        
        total_days          { int } : Total study duration
    
    Returns:
    --------
                { float }           : Expected sparsity [0, 1]
    """
    if (surveys_per_patient > total_days):
        raise ValueError(f"Cannot schedule {surveys_per_patient} surveys in {total_days} days")
    
    return 1.0 - (surveys_per_patient / total_days)


def get_weeks_to_days(weeks: float) -> int:
    """
    Convert weeks to days (standard 7-day weeks)
    
    Arguments:
    ----------
        weeks { float } : Number of weeks
    
    Returns:
    --------
            { int }     : Number of days
    """
    return int(weeks * 7)


# Run validation on import
validate_response_pattern_probabilities()


# CONSTANTS SUMMARY FOR DOCUMENTATION
CONSTANTS_SUMMARY = {'PHQ-9 Scale'        : {'Range'           : f'[{clinical_constants_instance.PHQ9_MIN_SCORE}, {clinical_constants_instance.PHQ9_MAX_SCORE}]',
                                             'MCID'            : clinical_constants_instance.PHQ9_MCID,
                                             'Severity Levels' : len(PHQ9_SEVERITY_THRESHOLDS),
                                            },
                    'Clinical Benchmarks' : {'STAR*D Dropout'          : f'{clinical_constants_instance.STARD_DROPOUT_RATE:.1%}',
                                             'STAR*D Response Rate'    : f'{clinical_constants_instance.STARD_RESPONSE_RATE:.1%}',
                                             'Trial Duration'          : f'{clinical_constants_instance.STARD_PRIMARY_ENDPOINT_WEEKS} weeks',
                                             'Test-Retest Reliability' : clinical_constants_instance.PHQ9_TEST_RETEST_RELIABILITY,
                                            },
                    'Temporal Modeling'   : {'AR Coefficient Range' : f'[{clinical_constants_instance.AR_COEFFICIENT_MIN}, {clinical_constants_instance.AR_COEFFICIENT_MAX}]',
                                             'Max Correlation Gap'  : f'{clinical_constants_instance.MAX_AUTOCORR_GAP_DAYS} days',
                                             'Weight Half-Life'     : f'{clinical_constants_instance.AUTOCORR_WEIGHT_HALFLIFE} days',
                                            },
                    'Response Patterns'   : {'Early Responder'   : f'{RESPONSE_PATTERN_PROBABILITIES["early_responder"]:.0%}',
                                             'Gradual Responder' : f'{RESPONSE_PATTERN_PROBABILITIES["gradual_responder"]:.0%}',
                                             'Late Responder'    : f'{RESPONSE_PATTERN_PROBABILITIES["late_responder"]:.0%}',
                                             'Non-Responder'     : f'{RESPONSE_PATTERN_PROBABILITIES["non_responder"]:.0%}',
                                            }
                   }