# Dependencies
import warnings
from pathlib import Path
from typing import Tuple
from typing import Literal
from pydantic import Field
from pydantic import BaseModel
from pydantic import ValidationInfo
from pydantic import model_validator
from pydantic import field_validator
from config.clinical_constants import clinical_constants_instance


class DataGenerationConfig(BaseModel):
    """
    Validated configuration for synthetic PHQ-9 data generation: all parameters have clinical justification and literature-based constraints
    """
    # SAMPLE SIZE PARAMETERS
    total_patients            : int                                          = Field(default     = 1000,
                                                                                     ge          = clinical_constants_instance.MIN_PATIENTS,
                                                                                     le          = clinical_constants_instance.MAX_PATIENTS,
                                                                                     description = "Total number of patients in the study",
                                                                                    )
    
    total_days                : int                                          = Field(default     = 365,
                                                                                     ge          = clinical_constants_instance.MIN_STUDY_DAYS,
                                                                                     le          = clinical_constants_instance.MAX_STUDY_DAYS,
                                                                                     description = f"Study duration in days ({clinical_constants_instance.MIN_STUDY_DAYS//30} months to {clinical_constants_instance.MAX_STUDY_DAYS//365} years typical for depression trials)",
                                                                                    )
    
    maximum_surveys_attempted : int                                          = Field(default     = clinical_constants_instance.MAX_SURVEYS_PER_PATIENT,
                                                                                     ge          = 4,
                                                                                     le          = 52,
                                                                                     description = "Maximum PHQ-9 surveys per patient (clinical realistic: weekly to monthly = 12-52)",
                                                                                    )

    min_surveys_attempted     : int                                          = Field(default     = clinical_constants_instance.MIN_SURVEYS_PER_PATIENT,
                                                                                     ge          = 5,
                                                                                     le          = clinical_constants_instance.MAX_SURVEYS_PER_PATIENT,
                                                                                     description = "Minimum PHQ-9 surveys per patient (ensures adequate longitudinal coverage)",
                                                                                    )
    
    random_seed               : int                                          = Field(default     = 2023,
                                                                                     ge          = 0,
                                                                                     description = "Random seed for reproducibility"
                                                                                    )
    
    # TEMPORAL DEPENDENCY PARAMETERS
    ar_coefficient            : float                                        = Field(default     = clinical_constants_instance.AR_COEFFICIENT_DEFAULT,
                                                                                     ge          = clinical_constants_instance.AR_COEFFICIENT_MIN,
                                                                                     le          = clinical_constants_instance.AR_COEFFICIENT_MAX,
                                                                                     description = f"Autocorrelation coefficient for AR(1) model. Literature: Kroenke et al. (2001) test-retest r={clinical_constants_instance.PHQ9_TEST_RETEST_RELIABILITY}",
                                                                                    )
    
    baseline_mean_score       : float                                        = Field(default     = clinical_constants_instance.TYPICAL_RCT_BASELINE_MEAN,
                                                                                     ge          = 10.0,
                                                                                     le          = 20.0,
                                                                                     description = "Mean baseline PHQ-9 score. Literature: Moderate-severe depression trials typically 15-17",
                                                                                    )
    
    baseline_std_score        : float                                        = Field(default     = clinical_constants_instance.TYPICAL_RCT_BASELINE_STD,
                                                                                     ge          = 2.0,
                                                                                     le          = 6.0,
                                                                                     description = "Standard deviation of baseline scores (population heterogeneity)",
                                                                                    )
    
    recovery_rate_mean        : float                                        = Field(default     = clinical_constants_instance.TYPICAL_RECOVERY_RATE_MEAN,
                                                                                     ge          = -0.20,
                                                                                     le          = -0.03,
                                                                                     description = f"Mean daily recovery rate (negative = improvement). Literature: Adjusted for {clinical_constants_instance.STARD_PRIMARY_ENDPOINT_WEEKS}-week STAR*D endpoint",  
                                                                                    )
    
    recovery_rate_std         : float                                        = Field(default     = clinical_constants_instance.TYPICAL_RECOVERY_RATE_STD,
                                                                                     ge          = 0.01,
                                                                                     le          = 0.08,
                                                                                     description = "Std dev of recovery rates (treatment response heterogeneity)",
                                                                                    )
    
    noise_std                 : float                                        = Field(default     = clinical_constants_instance.MEASUREMENT_NOISE_STD,
                                                                                     ge          = clinical_constants_instance.MEASUREMENT_NOISE_MIN,
                                                                                     le          = clinical_constants_instance.MEASUREMENT_NOISE_MAX,
                                                                                     description = f"Daily measurement noise. Literature: Löwe et al. (2004) MCID ~{clinical_constants_instance.PHQ9_MCID} points, so noise < {clinical_constants_instance.MEASUREMENT_NOISE_MAX}",    
                                                                                    )
    
    relapse_probability       : float                                        = Field(default     = clinical_constants_instance.RELAPSE_PROBABILITY_DAILY,
                                                                                     ge          = 0.05,
                                                                                     le          = 0.20,
                                                                                     description = "Probability of relapse per observed PHQ-9 measurement",
                                                                                    )
    
    relapse_magnitude_mean    : float                                        = Field(default     = clinical_constants_instance.RELAPSE_MAGNITUDE_MEAN,
                                                                                     ge          = 2.0,
                                                                                     le          = 5.0,
                                                                                     description = "Mean score increase during relapse episodes",
                                                                                    )

    relapse_distribution      : Literal['exponential', 'gamma', 'lognormal'] = Field(default     = 'exponential',
                                                                                     description = "Distribution for relapse magnitude. Exponential (default) has heavy tail; gamma is more bounded; lognormal has very heavy tail",
                                                                                    )
    
    # MISSINGNESS MECHANISM PARAMETERS
    dropout_rate              : float                                        = Field(default     = clinical_constants_instance.STARD_DROPOUT_RATE,
                                                                                     ge          = 0.05,
                                                                                     le          = 0.35,
                                                                                     description = f"Overall study dropout rate. Literature: STAR*D Level 1 = {clinical_constants_instance.STARD_DROPOUT_RATE:.0%} (Rush et al., 2006)",
                                                                                    )
    
    mcar_missingness_rate     : float                                        = Field(default     = clinical_constants_instance.MCAR_MISSINGNESS_RATE,
                                                                                     ge          = 0.0,
                                                                                     le          = clinical_constants_instance.MCAR_RATE_UPPER_BOUND,
                                                                                     description = "Missing Completely At Random rate per observation",
                                                                                    )

    
    # VALIDATION PARAMETERS
    max_autocorr_gap_days     : int                                          = Field(default     = clinical_constants_instance.MAX_AUTOCORR_GAP_DAYS,
                                                                                     ge          = 1,
                                                                                     le          = 30,
                                                                                     description = "Maximum gap (days) for pairs to be included in autocorrelation estimation",
                                                                                    )

    autocorr_weight_halflife  : float                                        = Field(default     = clinical_constants_instance.AUTOCORR_WEIGHT_HALFLIFE,
                                                                                     ge          = 1.0,
                                                                                     le          = 30.0,
                                                                                     description = "Half-life (days) for exponential temporal weighting in autocorrelation",
                                                                                    )

    max_autocorr_window_days  : int                                          = Field(default     = clinical_constants_instance.MAX_AUTOCORR_WINDOW_DAYS,
                                                                                     ge          = 7,
                                                                                     le          = 60,
                                                                                     description = "Maximum temporal window (days) for weighted autocorrelation computation",
                                                                                    )

    # RESPONSE PATTERN MODELING (NEW)
    enable_response_patterns  : bool                                         = Field(default     = True,
                                                                                     description = "Enable heterogeneous response patterns (early/gradual/late/non-responders)",
                                                                                    )

    enable_plateau_logic      : bool                                         = Field(default     = True,
                                                                                     description = "Enable plateau phase after response stabilization",
                                                                                    )
                                            
    # OUTPUT PATHS
    output_data_path          : Path                                         = Field(default     = Path("data/raw/synthetic_phq9_data.csv"),
                                                                                     description = "Path to save generated synthetic data",
                                                                                    )
    
    validation_report_path    : Path                                         = Field(default     = Path("results/generation/validation_reports/validation_report.json"),
                                                                                     description = "Path to save data validation report",
                                                                                    )
    

    @model_validator(mode = 'after')
    def validate_survey_constraints(self):
        """
        Ensure min/max survey counts are consistent
        """
        if (self.min_surveys_attempted > self.maximum_surveys_attempted):
            raise ValueError(f"min_surveys_attempted ({self.min_surveys_attempted}) cannot exceed maximum_surveys_attempted ({self.maximum_surveys_attempted})")
        
        # Additional validation: ensure max surveys fit in study duration
        if (self.maximum_surveys_attempted > self.total_days):
            raise ValueError(f"maximum_surveys_attempted ({self.maximum_surveys_attempted}) exceeds total_days ({self.total_days}). Cannot schedule more surveys than days.")
        
        # Warn if min surveys might conflict with dropout: heuristic assuming weekly spacing
        min_days_needed         = self.min_surveys_attempted * 7  
        
        if (min_days_needed > (self.total_days * 0.5)):
            warnings.warn(f"min_surveys_attempted ({self.min_surveys_attempted}) may be too high given dropout_rate ({self.dropout_rate:.1%}). Some patients may drop out before completing minimum surveys.")
        
        return self


    @field_validator('ar_coefficient')
    @classmethod
    def validate_autocorrelation(cls, v):
        """
        Validate AR coefficient against literature
        """
        if not (clinical_constants_instance.AR_COEFFICIENT_MIN <= v <= clinical_constants_instance.AR_COEFFICIENT_MAX):
            raise ValueError(f"AR coefficient {v:.2f} outside literature range [{clinical_constants_instance.AR_COEFFICIENT_MIN}, {clinical_constants_instance.AR_COEFFICIENT_MAX}]." 
                             f"Reference: Kroenke et al. (2001) - PHQ-9 test-retest r={clinical_constants_instance.PHQ9_TEST_RETEST_RELIABILITY}"
                            )

        return v
    

    @field_validator('baseline_mean_score')
    @classmethod
    def validate_baseline(cls, v):
        """
        Validate baseline severity against clinical trials
        """
        if not (clinical_constants_instance.BASELINE_VALIDATION_LOWER <= v <= clinical_constants_instance.BASELINE_VALIDATION_UPPER):
            warnings.warn(f"Baseline mean {v:.1f} outside typical range [{clinical_constants_instance.BASELINE_VALIDATION_LOWER}-{clinical_constants_instance.BASELINE_VALIDATION_UPPER}]."
                          f"Most depression RCTs enroll moderate-severe patients (PHQ-9 {clinical_constants_instance.TYPICAL_RCT_BASELINE_MEAN-1}-{clinical_constants_instance.TYPICAL_RCT_BASELINE_MEAN+1})."
                         )

        return v
    

    @field_validator('recovery_rate_mean')
    @classmethod
    def validate_recovery_rate(cls, v, info: ValidationInfo):
        """
        Validate recovery rate against 12-week STAR*D expectations
        """
        baseline        = info.data.get('baseline_mean_score', clinical_constants_instance.TYPICAL_RCT_BASELINE_MEAN)

        expected_12week = baseline + (v * clinical_constants_instance.STARD_PRIMARY_ENDPOINT_DAYS)

        if (expected_12week < -5.0):
            raise ValueError(f"Recovery rate {v:.3f} too aggressive - would result in 12-week score of {expected_12week:.1f}. "
                             f"Consider recovery_rate_mean in range [-0.08, -0.04]."
                            )

        actual_12week_estimate = max(0.0, expected_12week)
        expected_reduction_pct = (baseline - actual_12week_estimate) / baseline * 100

        if not (clinical_constants_instance.EXPECTED_RESPONSE_RATE_LOWER * 100 <= expected_reduction_pct <= clinical_constants_instance.EXPECTED_RESPONSE_RATE_UPPER * 100):
            warnings.warn(f"Expected 12-week symptom reduction {expected_reduction_pct:.1f}% outside typical range "
                          f"[{clinical_constants_instance.EXPECTED_RESPONSE_RATE_LOWER*100:.0f}-"
                          f"{clinical_constants_instance.EXPECTED_RESPONSE_RATE_UPPER*100:.0f}%]. "
                          f"Theoretical 12-week score: {expected_12week:.1f}. "
                          f"Reference: Rush et al. (2006) STAR*D."
                         )

        return v
        

    @field_validator('noise_std')
    @classmethod
    def validate_noise(cls, v):
        """
        Validate noise level against MCID
        """
        if (v > (clinical_constants_instance.PHQ9_MCID - 1.5)):
            warnings.warn(f"Noise std {v:.1f} approaches MCID ({clinical_constants_instance.PHQ9_MCID} points)." 
                          f"Should be substantially smaller. Reference: Löwe et al. (2004)"
                         )

        return v
    

    @field_validator('dropout_rate')
    @classmethod
    def validate_dropout(cls, v):
        """
        Validate dropout rate against STAR*D and real-world studies
        """
        if not (clinical_constants_instance.DROPOUT_RATE_LOWER <= v <= clinical_constants_instance.DROPOUT_RATE_UPPER):
            warnings.warn(f"Dropout rate {v:.2%} outside typical range [{clinical_constants_instance.DROPOUT_RATE_LOWER:.0%}-{clinical_constants_instance.DROPOUT_RATE_UPPER:.0%}]. "
                          f"Reference: STAR*D Level 1 dropout = {clinical_constants_instance.STARD_DROPOUT_RATE:.0%} (Rush et al., 2006). "
                          f"Real-world studies: 18-30% (Fernandez et al., 2015)"
                         )
        
        return v
    

    class Config:
        validate_assignment = True
        extra               = 'forbid'  # Prevent typos in parameter names
        

    def get_summary(self) -> dict:
        """
        Get human-readable summary of configuration
        """
        return {'Study Design'   : {'Patients'            : self.total_patients,
                                    'Duration (days)'     : self.total_days,
                                    'Max surveys/patient' : self.maximum_surveys_attempted,
                                    'Min surveys/patient' : self.min_surveys_attempted,
                                   },
                'Temporal Model' : {'AR(1) coefficient' : f"{self.ar_coefficient:.2f}",
                                    'Baseline severity' : f"{self.baseline_mean_score:.1f} ± {self.baseline_std_score:.1f}",
                                    'Recovery rate'     : f"{self.recovery_rate_mean:.3f} points/day",
                                    'Measurement noise' : f"{self.noise_std:.1f} points",
                                   },
                'Response Patterns' : {'Enabled'              : self.enable_response_patterns,
                                       'Plateau logic'        : self.enable_plateau_logic,
                                      },
                'Missingness'    : {'Dropout rate' : f"{self.dropout_rate:.1%}",
                                    'MCAR rate'    : f"{self.mcar_missingness_rate:.1%}",
                                   },
                'Relapse'        : {'Distribution' : self.relapse_distribution,
                                    'Probability'  : f"{self.relapse_probability:.1%}",
                                    'Magnitude'    : f"{self.relapse_magnitude_mean:.1f} points",
                                   },
                'Output'         : {'Data'       : str(self.output_data_path),
                                    'Validation' : str(self.validation_report_path),
                                   }
               }



# Convenience functions
def validate_against_literature(config: DataGenerationConfig) -> dict:
    """
    Validate configuration parameters against clinical literature and returns a python dict with validation results and literature references
    
    Arguments:
    ----------
        config { DataGenerationConfig } : DataGenerationConfig instance
    
    Returns:
    --------
                     { dict }           : Dictionary containing:
                                          - valid      : bool (overall validation status)
                                          - warnings   : list of warning messages
                                          - references : dict of literature citations
    """
    validation = {'valid'      : True,
                  'warnings'   : [],
                  'references' : {},
                 }
    
    #  Baseline severity Check
    if not (clinical_constants_instance.BASELINE_VALIDATION_LOWER <= config.baseline_mean_score <= clinical_constants_instance.BASELINE_VALIDATION_UPPER):
        validation['warnings'].append(f"Baseline mean ({config.baseline_mean_score:.1f}) outside typical range [{clinical_constants_instance.BASELINE_VALIDATION_LOWER}-{clinical_constants_instance.BASELINE_VALIDATION_UPPER}] for moderate-severe depression trials.")
        validation['valid'] = False
    
    validation['references']['baseline'] = (f"Kroenke et al. (2001): PHQ-9 ≥15 indicates moderately severe depression. Typical RCT enrollment: PHQ-9 {clinical_constants_instance.TYPICAL_RCT_BASELINE_MEAN-1}-{clinical_constants_instance.TYPICAL_RCT_BASELINE_MEAN+1}")
    
    # Autocorrelation Check
    if not (clinical_constants_instance.AR_COEFFICIENT_MIN + 0.1 <= config.ar_coefficient <= clinical_constants_instance.AR_COEFFICIENT_MAX - 0.05):
        validation['warnings'].append(f"AR coefficient ({config.ar_coefficient:.2f}) outside optimal range [{clinical_constants_instance.AR_COEFFICIENT_MIN+0.1:.1f}-{clinical_constants_instance.AR_COEFFICIENT_MAX-0.05:.2f}] for depression symptom stability.")
    
    validation['references']['autocorrelation'] = (f"Kroenke et al. (2001): PHQ-9 test-retest reliability r={clinical_constants_instance.PHQ9_TEST_RETEST_RELIABILITY} (2-day interval)")
    
    # Recovery rate & response Check (12-week endpoint)
    expected_12week_score = (config.baseline_mean_score + config.recovery_rate_mean * clinical_constants_instance.STARD_PRIMARY_ENDPOINT_DAYS)
    reduction_pct         = ((config.baseline_mean_score - max(0, expected_12week_score)) / config.baseline_mean_score * 100)
    
    if not (clinical_constants_instance.EXPECTED_RESPONSE_RATE_LOWER * 100 <= reduction_pct <= clinical_constants_instance.EXPECTED_RESPONSE_RATE_UPPER * 100):
        validation['warnings'].append(f"Expected 12-week reduction ({reduction_pct:.1f}%) outside typical response rate [{clinical_constants_instance.EXPECTED_RESPONSE_RATE_LOWER*100:.0f}-{clinical_constants_instance.EXPECTED_RESPONSE_RATE_UPPER*100:.0f}%] for antidepressant trials.")
    
    validation['references']['response_rate'] = (f"Rush et al. (2006): STAR*D Level 1 showed {clinical_constants_instance.STARD_RESPONSE_RATE:.0%} response rate (≥50% symptom reduction) after {clinical_constants_instance.STARD_PRIMARY_ENDPOINT_WEEKS} weeks")
    
    # Noise level (MCID) Check
    if ((config.noise_std < clinical_constants_instance.MEASUREMENT_NOISE_MIN + 0.5) or (config.noise_std > clinical_constants_instance.MEASUREMENT_NOISE_MAX - 0.5)):
        validation['warnings'].append(f"Noise std ({config.noise_std:.1f}) should be {clinical_constants_instance.MEASUREMENT_NOISE_MIN+0.5:.1f}-{clinical_constants_instance.MEASUREMENT_NOISE_MAX-0.5:.1f} points to reflect realistic day-to-day variation.")
    
    validation['references']['mcid'] = (f"Löwe et al. (2004): Minimal clinically important difference approximately {clinical_constants_instance.PHQ9_MCID} points on PHQ-9. Daily noise should be smaller.")
    
    # Dropout rate Check
    if not (clinical_constants_instance.DROPOUT_RATE_LOWER <= config.dropout_rate <= clinical_constants_instance.DROPOUT_RATE_UPPER):
        validation['warnings'].append(f"Dropout rate ({config.dropout_rate:.2%}) outside typical range [{clinical_constants_instance.DROPOUT_RATE_LOWER:.0%}-{clinical_constants_instance.DROPOUT_RATE_UPPER:.0%}] for depression trials.")
    
    validation['references']['dropout'] = (f"Rush et al. (2006): STAR*D Level 1 dropout rate = {clinical_constants_instance.STARD_DROPOUT_RATE:.0%}. Real-world studies: 18-30% (Fernandez et al., 2015)")
    
    return validation