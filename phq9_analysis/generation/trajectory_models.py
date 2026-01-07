# Dependencies
import numpy as np
from typing import List
from typing import Optional
from dataclasses import field
from dataclasses import dataclass


@dataclass
class PatientTrajectory:
    """
    Store patient-specific trajectory parameters by encapsulating all individual characteristics
    that determine a patient's depression symptom trajectory over time
    
    Attributes:
    -----------
        baseline            : Initial PHQ-9 score at treatment start (0-27)
       
        recovery_rate       : Daily score change rate (negative = improvement)
       
        noise_std           : Individual measurement variability (day-to-day fluctuation)
       
        ar_coefficient      : Temporal autocorrelation strength (0.5-0.9)
       
        last_score          : Most recent observed score (for AR process)
       
        relapse_history     : List of days when relapses occurred
       
        treatment_start_day : Day when treatment began (default: 1)
    
    Clinical Notes:
    ---------------
    - baseline       : Captures initial severity heterogeneity
    - recovery_rate  : Models treatment response heterogeneity
    - noise_std      : Reflects individual symptom stability
    - ar_coefficient : Day-to-day symptom persistence
    """
    
    baseline            : float
    recovery_rate       : float
    noise_std           : float
    ar_coefficient      : float
    last_score          : Optional[float] = None
    relapse_history     : List[int]       = field(default_factory = list)
    treatment_start_day : int = 1
    

    def __post_init__(self):
        """
        Validate trajectory parameters
        """
        if not (0 <= self.baseline <= 27):
            raise ValueError(f"Baseline {self.baseline} outside valid PHQ-9 range [0, 27]")
        
        if not (0.5 <= self.ar_coefficient <= 0.9):
            raise ValueError(f"AR coefficient {self.ar_coefficient} outside realistic range [0.5, 0.9]. Literature: Kroenke et al. (2001) test-retest r=0.84")
        
        if (self.noise_std < 0):
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}")
    

    def get_expected_score_at_day(self, day: int) -> float:
        """
        Calculate expected PHQ-9 score at a given day based on treatment trajectory
        
        Arguments:
        ----------
            day { int }  : Day number (1 to study duration)
        

        Returns:
        --------
            { float }    : Expected score (without noise or autocorrelation)
        

        Formula:
        --------
            E[Y_t] = baseline + recovery_rate * (t - treatment_start)
        """
        days_since_treatment = day - self.treatment_start_day
        expected_score       = self.baseline + (self.recovery_rate * days_since_treatment)
        
        # Prevent negative expected scores
        return max(expected_score, 0.0)
    

    def update_last_score(self, score: float):
        """
        Update the last observed score for AR process
        """
        self.last_score = score

    
    def add_relapse(self, day: int):
        """
        Record a relapse event
        """
        self.relapse_history.append(day)

    
    def get_summary(self) -> dict:
        """
        Get human-readable summary of trajectory
        """
        return {'baseline'       : f"{self.baseline:.1f}",
                'recovery_rate'  : f"{self.recovery_rate:.3f} points/day",
                'noise_std'      : f"{self.noise_std:.2f}",
                'ar_coefficient' : f"{self.ar_coefficient:.2f}",
                'n_relapses'     : len(self.relapse_history),
                'last_score'     : f"{self.last_score:.1f}" if self.last_score else "None",
               }



class AR1Model:
    """
    First-order autoregressive model for PHQ-9 score generation: Y_t = α*Y_{t-1} + (1-α)*μ_t + ε_t + relapse_t
    
    This model combines:
    - Temporal autocorrelation (α*Y_{t-1})
    - Treatment-driven trend ((1-α)*μ_t)
    - Daily measurement noise (ε_t)
    - Occasional relapses (relapse_t)
    
    Clinical Justification:
    -----------------------
    - AR(1) structure captures day-to-day symptom stability
    - α ≈ 0.7 matches PHQ-9 test-retest reliability (Kroenke et al.)
    - Treatment trend allows gradual improvement
    - Noise reflects natural symptom fluctuation
    - Relapses model temporary worsening episodes
    """
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize AR1 model
        
        Arguments:
        ----------
            random_seed { int } : Seed for reproducibility (optional)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
    

    def generate_score(self, trajectory: PatientTrajectory, day: int, relapse_probability: float = 0.10, relapse_magnitude_mean: float = 3.5) -> float:
        """
        Generate PHQ-9 score for a specific day using AR(1) model

        Process:
        --------
        - Calculate expected score from treatment trend
        - Apply AR(1) autocorrelation if previous score exists
        - Add daily measurement noise
        - Add relapse component (with probability)
        - Clip to valid PHQ-9 range
        
        Arguments:
        ----------
            trajectory             { PatientTrajectory } : Patient's trajectory parameters
            
            day                          { int }         : Current day number
            
            relapse_probability         { float }        : Probability of relapse on this day (0-1)
            
            relapse_magnitude_mean      { float }        : Mean score increase during relapse
        
        Returns:
        --------
                            { float }                    : PHQ-9 score clipped to valid range [0, 27]
        """
        # Expected score based on treatment progression
        expected_score = trajectory.get_expected_score_at_day(day = day)
        
        # Apply AR(1) process
        if trajectory.last_score is not None:
            # Weighted average of previous score and expected trend
            score = (trajectory.ar_coefficient * trajectory.last_score + (1 - trajectory.ar_coefficient) * expected_score)
        
        else:
            # First observation: start at expected value
            score = expected_score
        
        # Add daily measurement noise
        noise  = np.random.normal(0, trajectory.noise_std)
        score += noise
        
        # Add relapse component
        if (np.random.random() < relapse_probability):
            # Relapse magnitude follows exponential distribution
            relapse_magnitude = np.random.exponential(scale=relapse_magnitude_mean)
            score            += relapse_magnitude

            trajectory.add_relapse(day)
        
        # Clip to valid PHQ-9 range
        score = np.clip(score, 0, 27)
        
        # Update trajectory
        trajectory.update_last_score(score)
        
        return score

    
    @staticmethod
    def calculate_autocorrelation(scores: np.ndarray, lag: int = 1) -> float:
        """
        Calculate autocorrelation for a series of scores: r(k) = Cov(Y_t, Y_{t-k}) / Var(Y_t)
        
        Arguments:
        ----------
            scores { np.ndarray } : Array of PHQ-9 scores (no NaNs)

            lag       { int }     : Lag for autocorrelation (default: 1 for AR(1))
        
        Returns:
        --------
                 { float }        : Autocorrelation coefficient at specified lag
        """
        if (len(scores) <= lag):
            return np.nan
        
        # Remove mean
        y           = scores - np.mean(scores)
        
        # Calculate autocorrelation
        numerator   = np.sum(y[:-lag] * y[lag:])
        denominator = np.sum(y ** 2)
        
        if (denominator == 0):
            return np.nan
        
        return numerator / denominator



def initialize_patient_trajectories(n_patients: int, baseline_mean: float, baseline_std: float, recovery_rate_mean: float, recovery_rate_std: float, 
                                    noise_std: float, ar_coefficient: float, random_seed: Optional[int] = None) -> dict:
    """
    Initialize trajectories for all patients in the study

    Clinical Notes:
    ---------------
    - Baseline follows normal distribution (moderate-severe range)
    - Recovery rate follows normal (some respond faster than others)
    - Noise follows gamma distribution (ensures positive values)
    - AR coefficient has small variation (symptom stability varies slightly)
    
    Arguments:
    ----------
        n_patients          { int }  : Number of patients
        
        baseline_mean      { float } : Mean baseline PHQ-9 score

        baseline_std       { float } : Std dev of baseline scores

        recovery_rate_mean { float } : Mean recovery rate

        recovery_rate_std  { float } : Std dev of recovery rates

        noise_std          { float } : Daily measurement noise
        
        ar_coefficient     { float } : AR(1) coefficient
        
        random_seed         { int }  : Random seed for reproducibility
    
    Returns:
    --------
                { dict }             : Dictionary mapping patient_id to PatientTrajectory
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    trajectories = dict()
    
    for patient_id in range(1, n_patients + 1):
        # Sample baseline severity (normal distribution, clipped to [0, 27])
        baseline                 = np.clip(np.random.normal(baseline_mean, baseline_std), 0, 27)
        
        # Sample individual recovery rate
        recovery_rate            = np.random.normal(recovery_rate_mean, recovery_rate_std)
        
        # Sample individual noise level (gamma distribution ensures positive): Shape parameter = 4 | scale = noise_std/4
        individual_noise         = np.clip(np.random.gamma(shape = 4, scale = noise_std/4), 1.0, 4.0)
        
        # Sample AR coefficient (slight variation around population mean)
        ar_coef                  = np.clip(np.random.normal(ar_coefficient, 0.05), 0.5, 0.9)
        
        # Create trajectory
        trajectory               = PatientTrajectory(baseline       = baseline,
                                                     recovery_rate  = recovery_rate,
                                                     noise_std      = individual_noise,
                                                     ar_coefficient = ar_coef,
                                                    )
        
        trajectories[patient_id] = trajectory
    
    return trajectories