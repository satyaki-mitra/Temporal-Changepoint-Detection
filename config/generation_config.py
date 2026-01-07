# Dependencies
import warnings
from pathlib import Path
from typing import Tuple
from pydantic import Field
from pydantic import BaseModel
from pydantic import validator


class DataGenerationConfig(BaseModel):
    """
    Validated configuration for synthetic PHQ-9 data generation
    
    All parameters have clinical justification and literature-based constraints
    """
    # SAMPLE SIZE PARAMETERS
    total_patients            : int   = Field(default     = 1000,
                                              ge          = 100,
                                              le          = 5000,
                                              description = "Total number of patients in the study",
                                             )
    
    total_days                : int   = Field(default     = 365,
                                              ge          = 180,
                                              le          = 730,
                                              description = "Study duration in days (6 months to 2 years typical for depression trials)",
                                             )
    
    required_sample_count     : int   = Field(default     = 50,
                                              ge          = 20,
                                              le          = 100,
                                              description = "Number of observation days (sparse sampling)",
                                             )
    
    maximum_surveys_attempted : int   = Field(default     = 7,
                                              ge          = 4,
                                              le          = 10,
                                              description = "Maximum PHQ-9 surveys per patient (clinical realistic: 4-8)",
                                             )
    
    random_seed               : int   = Field(default     = 2023,
                                              ge          = 0,
                                              description = "Random seed for reproducibility"
                                             )
    
    # TEMPORAL DEPENDENCY PARAMETERS
    ar_coefficient            : float = Field(default     = 0.70,
                                              ge          = 0.5,
                                              le          = 0.9,
                                              description = "Autocorrelation coefficient for AR(1) model. Literature: Kroenke et al. (2001) test-retest r=0.84",
                                             )
    
    baseline_mean_score       : float = Field(default     = 16.0,
                                              ge          = 10.0,
                                              le          = 20.0,
                                              description = "Mean baseline PHQ-9 score. Literature: Moderate-severe depression trials typically 15-17",
                                             )
    
    baseline_std_score        : float = Field(default     = 4.0,
                                              ge          = 2.0,
                                              le          = 6.0,
                                              description = "Standard deviation of baseline scores (population heterogeneity)",
                                             )
    
    recovery_rate_mean        : float = Field(default     = -0.06,
                                              ge          = -0.20,
                                              le          = -0.05,
                                              description = "Mean daily recovery rate (negative = improvement). Literature: -0.10 results in ~50% reduction over 80 days (STAR*D)",  
                                             )
    
    recovery_rate_std         : float = Field(default     = 0.03,
                                              ge          = 0.01,
                                              le          = 0.08,
                                              description = "Std dev of recovery rates (treatment response heterogeneity)",
                                             )
    
    noise_std                 : float = Field(default     = 2.5,
                                              ge          = 1.0,
                                              le          = 4.0,
                                              description = "Daily measurement noise. Literature: Löwe et al. (2004) MCID ~5 points, so noise < 3",    
                                             )
    
    relapse_probability       : float = Field(default     = 0.10,
                                              ge          = 0.05,
                                              le          = 0.20,
                                              description = "Daily probability of symptom relapse (literature: ~10%)",
                                             )
    
    relapse_magnitude_mean    : float = Field(default     = 3.5,
                                              ge          = 2.0,
                                              le          = 5.0,
                                              description = "Mean score increase during relapse episodes",
                                             )
    
    # MISSINGNESS MECHANISM PARAMETERS
    dropout_rate              : float = Field(default     = 0.18,
                                              ge          = 0.05,
                                              le          = 0.25,
                                              description = "Overall study dropout rate. Literature: Fournier et al. (2010) meta-analysis ~13%",
                                             )
    
    mcar_missingness_rate     : float = Field(default     = 0.08,
                                              ge          = 0.0,
                                              le          = 0.15,
                                              description = "Missing Completely At Random rate per observation",
                                             )
                                            
    # OUTPUT PATHS
    output_data_path          : Path  = Field(default     = Path("data/raw/synthetic_phq9_data.csv"),
                                              description = "Path to save generated synthetic data",
                                             )
    
    validation_report_path    : Path  = Field(default     = Path("results/generation/validation_reports/validation_report.json"),
                                              description = "Path to save data validation report",
                                             )
    

    @validator('required_sample_count')
    def validate_sample_count(cls, v, values):
        """
        Ensure sample count is reasonable relative to total days
        """
        total_days = values.get('total_days', 365)
        
        if (v > total_days):
            raise ValueError(f"Sample count ({v}) cannot exceed total days ({total_days})")
        
        if (v < 0.1 * total_days):
            raise ValueError(f"Sample count ({v}) should be at least 10% of total days (minimum: {int(0.1 * total_days)})")
        
        return v
    

    @validator('ar_coefficient')
    def validate_autocorrelation(cls, v):
        """
        Validate AR coefficient against literature
        """
        if not (0.5 <= v <= 0.9):
            raise ValueError(f"AR coefficient {v:.2f} outside literature range [0.5, 0.9]. Reference: Kroenke et al. (2001) - PHQ-9 test-retest r=0.84")

        return v
    

    @validator('baseline_mean_score')
    def validate_baseline(cls, v):
        """
        Validate baseline severity against clinical trials
        """
        if not (14.0 <= v <= 18.0):
            # Warning, not error - allow flexibility    
            warnings.warn(f"Baseline mean {v:.1f} outside typical trial range [14-18]. Most depression RCTs enroll moderate-severe patients (PHQ-9 15-17).")

        return v
    

    @validator('recovery_rate_mean')
    def validate_recovery_rate(cls, v, values):
        """
        Ensure recovery rate will produce realistic response rates
        """
        baseline       = values.get('baseline_mean_score', 16.0)
        total_days     = values.get('total_days', 365)
        
        # Calculate expected final score
        expected_final = baseline + (v * total_days)
        
        if (expected_final < 0):
            raise ValueError(f"Recovery rate {v:.3f} too aggressive - would result in negative scores. Expected final score: {expected_final:.1f}")
        
        # Calculate expected response rate (≥50% reduction)
        expected_reduction_pct = (baseline - expected_final) / baseline * 100
        
        if not (30 <= expected_reduction_pct <= 70):
            warnings.warn(f"Expected symptom reduction {expected_reduction_pct:.1f}% outside typical range [40-60%] for antidepressant trials. Reference: Rush et al. (2006) STAR*D - 47% response rate")
        
        return v
    

    @validator('noise_std')
    def validate_noise(cls, v):
        """
        Validate noise level against MCID
        """
        if (v > 3.5):
            warnings.warn(f"Noise std {v:.1f} exceeds recommended maximum (3.5 points). Should be smaller than MCID ~5 points. Reference: Löwe et al. (2004)")

        return v
    

    @validator('dropout_rate')
    def validate_dropout(cls, v):
        """
        Validate dropout rate against meta-analyses
        """
        if not (0.10 <= v <= 0.20):
            warnings.warn(f"Dropout rate {v:.2%} outside typical range [10-20%]. Reference: Fournier et al. (2010) - meta-analysis dropout ~13%")
        
        return v
    
    class Config:
        validate_assignment = True

        # Prevent typos in parameter names
        extra               = 'forbid'  
        

    def get_summary(self) -> dict:
        """
        Get human-readable summary of configuration
        """
        return {'Study Design'   : {'Patients'            : self.total_patients,
                                    'Duration (days)'     : self.total_days,
                                    'Observation days'    : self.required_sample_count,
                                    'Max surveys/patient' : self.maximum_surveys_attempted,
                                   },
                'Temporal Model' : {'AR(1) coefficient' : f"{self.ar_coefficient:.2f}",
                                    'Baseline severity' : f"{self.baseline_mean_score:.1f} ± {self.baseline_std_score:.1f}",
                                    'Recovery rate'     : f"{self.recovery_rate_mean:.3f} points/day",
                                    'Measurement noise' : f"{self.noise_std:.1f} points",
                                   },
                'Missingness'    : {'Dropout rate' : f"{self.dropout_rate:.1%}",
                                    'MCAR rate'    : f"{self.mcar_missingness_rate:.1%}",
                                   },
                'Output'         : {'Data'       : str(self.output_data_path),
                                    'Validation' : str(self.validation_report_path),
                                   }
               }


def validate_against_literature(config: DataGenerationConfig) -> dict:
    """
    Validate configuration parameters against clinical literature
    
    Returns dict with validation results and literature references
    
    Arguments:
    ----------
        config { DataGenerationConfig } : DataGenerationConfig instance
    
    Returns:
    --------
                     { dict }           : Dictionary containing:
                                          - valid      : bool (overall validation status)
                                          - warnings   : list of warning messages
                                          - references : dict of literature citations
                                    
    References:
    -----------
    1. Kroenke et al. (2001): The PHQ-9: validity of a brief depression severity measure. J Gen Intern Med.
    2. Rush et al. (2006): Acute and longer-term outcomes in depressed outpatients requiring one or several treatment steps: STAR*D.
    3. Löwe et al. (2004): Monitoring depression treatment outcomes with the PHQ-9. Med Care.
    4. Fournier et al. (2010): Antidepressant drug effects and depression severity: a patient-level meta-analysis. JAMA.
    """
    validation = {'valid'      : True,
                  'warnings'   : [],
                  'references' : {},
                 }
    
    #  Baseline severity Check
    if not (14.0 <= config.baseline_mean_score <= 18.0):
        validation['warnings'].append(f"⚠️  Baseline mean ({config.baseline_mean_score:.1f}) outside typical range [14-18] for moderate-severe depression trials.")
        validation['valid'] = False
    
    validation['references']['baseline'] = ("Kroenke et al. (2001): PHQ-9 ≥15 indicates moderately severe depression. Typical RCT enrollment: PHQ-9 15-17")
    
    # Autocorrelation Check
    if not (0.6 <= config.ar_coefficient <= 0.85):
        validation['warnings'].append(f"⚠️  AR coefficient ({config.ar_coefficient:.2f}) outside typical range [0.6-0.85] for depression symptom stability.")
    
    validation['references']['autocorrelation'] = ("Kroenke et al. (2001): PHQ-9 test-retest reliability r=0.84 (2-day interval)")
    
    # Recovery rate & response Check
    expected_final_score = (config.baseline_mean_score + config.recovery_rate_mean * config.total_days)
    reduction_pct        = ((config.baseline_mean_score - expected_final_score) / config.baseline_mean_score * 100)
    
    if not (40 <= reduction_pct <= 60):
        validation['warnings'].append(f"⚠️  Expected reduction ({reduction_pct:.1f}%) outside typical response rate [40-60%] for antidepressant trials.")
    
    validation['references']['response_rate'] = ("Rush et al. (2006): STAR*D Level 1 showed 47% response rate (≥50% symptom reduction) after 12-14 weeks")
    
    # Noise level (MCID) Check
    if ((config.noise_std < 1.5) or (config.noise_std > 3.5)):
        validation['warnings'].append(f"⚠️  Noise std ({config.noise_std:.1f}) should be 2-3 points to reflect realistic day-to-day variation.")
    
    validation['references']['mcid'] = ("Löwe et al. (2004): Minimal clinically important difference approximately 5 points on PHQ-9. Daily noise should be smaller.")
    
    # Dropout rate Check
    if not (0.10 <= config.dropout_rate <= 0.20):
        validation['warnings'].append(f"⚠️  Dropout rate ({config.dropout_rate:.2%}) outside typical range [10-20%] for depression trials.")
    
    validation['references']['dropout'] = ("Fournier et al. (2010): Patient-level meta-analysis of antidepressant trials - average dropout rate 13%")
    
    return validation
