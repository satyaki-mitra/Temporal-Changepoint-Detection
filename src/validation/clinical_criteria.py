# Dependencies
from typing import Dict
from typing import Tuple
from dataclasses import dataclass


@dataclass
class ClinicalBenchmark:
    """
    Literature-based benchmarks for PHQ-9 data validation
    """
    # Kroenke et al. (2001) - PHQ-9 Validation
    AUTOCORRELATION_RANGE     : Tuple[float, float] = (0.70, 0.90)      # Test-retest r=0.84
    
    # Rush et al. (2006) - STAR*D
    RESPONSE_RATE_RANGE       : Tuple[float, float] = (0.40, 0.60)      # 47% in Level-1
    REMISSION_RATE_RANGE      : Tuple[float, float] = (0.28, 0.40)      # 28-37% typical
    
    # Typical RCT enrollment
    BASELINE_SEVERITY_RANGE   : Tuple[float, float] = (14.0, 18.0)      # Moderate-severe
    
    # Löwe et al. (2004) - MCID
    MCID_THRESHOLD            : float               = 5.0               # Minimal clinically important difference
    MIN_IMPROVEMENT           : float               = 3.0               # Minimum meaningful change
    
    # Fournier et al. (2010) - Meta-analysis
    DROPOUT_RATE_RANGE        : Tuple[float, float] = (0.10, 0.20)      # ~13% average
    
    # Cuijpers et al. (2014) - Response timing
    EARLY_RESPONSE_DAYS       : int                 = 14                # First 2 weeks indicator
    ACUTE_PHASE_DAYS          : int                 = 84                # 12 weeks
    
    # Score distribution properties
    EXPECTED_SKEWNESS_RANGE   : Tuple[float, float] = (-0.5, 0.5)       # Near-normal at baseline
    EXPECTED_KURTOSIS_RANGE   : Tuple[float, float] = (-1.0, 3.0)       # Not too heavy-tailed
    
    # Relapse characteristics (Monroe & Harkness, 2011)
    RELAPSE_RATE_RANGE        : Tuple[float, float] = (0.20, 0.40)      # 20-40% within 1 year
    RELAPSE_SEVERITY_INCREASE : float               = 5.0               # Typical relapse magnitude
    
    # Temporal pattern expectations
    EXPECTED_TRAJECTORY       : str                 = "linear_decline"  # Population-level
    TYPICAL_SLOPE_RANGE       : Tuple[float, float] = (-0.10, -0.03)    # Points/day


@dataclass
class StatisticalCriteria:
    """
    Statistical quality criteria
    """
    MIN_OBSERVATIONS           : int = 3000    # Minimum for reliable analysis
    MAX_MISSINGNESS            : float = 0.98  # Maximum acceptable sparsity
    
    MIN_PATIENTS_WITH_FOLLOWUP : int = 500     # Require adequate longitudinal data
    MIN_SURVEYS_PER_PATIENT    : int = 2       # Minimum for trajectory
    
    # Distributional checks
    MAX_OUTLIER_RATE           : float = 0.05  # Max 5% outliers (>3 SD)
    MIN_VARIANCE               : float = 1.0   # Ensure non-degenerate data