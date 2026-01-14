# Dependencies
import numpy as np
from typing import List
from typing import Optional
from dataclasses import field
from dataclasses import dataclass
from numpy.random import SeedSequence
from config.clinical_constants import clinical_constants_instance
from config.clinical_constants import RESPONSE_PATTERN_PROBABILITIES


@dataclass
class PatientTrajectory:
    """
    Store patient-specific trajectory parameters by encapsulating all individual characteristics that determine a patient's depression symptom trajectory over time

    Attributes:
    -----------
        baseline            : Initial PHQ-9 score at treatment start (0-27)

        recovery_rate       : Daily score change rate (negative = improvement)
        
        noise_std           : Individual measurement variability (day-to-day fluctuation)
        
        ar_coefficient      : Temporal autocorrelation strength (0.5-0.9)
        
        last_score          : Most recent observed score (for AR process)
        
        last_day            : Most recent day with observation
        
        relapse_history     : List of days when relapses occurred
        
        treatment_start_day : Day when treatment began (default: 1)

        response_pattern    : Response category (early/gradual/late/non-responder)

        plateau_start_day   : Day when symptom improvement plateaus

        in_plateau_phase    : Boolean indicating if currently in plateau

    Clinical Notes:
    ---------------
    - baseline         : Captures initial severity heterogeneity
    - recovery_rate    : Models treatment response heterogeneity
    - noise_std        : Reflects individual symptom stability
    - ar_coefficient   : Day-to-day symptom persistence
    - response_pattern : Captures heterogeneous response timing
    - plateau logic    : Models symptom stabilization after response
    """
    baseline            : float
    recovery_rate       : float
    noise_std           : float
    ar_coefficient      : float
    rng                 : np.random.Generator
    last_score          : Optional[float]     = None
    last_day            : Optional[int]       = None
    relapse_history     : List[int]           = field(default_factory = list)
    treatment_start_day : int                 = 1
    response_pattern    : str                 = 'gradual_responder'
    plateau_start_day   : Optional[int]       = None
    in_plateau_phase    : bool                = False


    def __post_init__(self):
        """
        Validate trajectory parameters
        """
        if not (clinical_constants_instance.PHQ9_MIN_SCORE <= self.baseline <= clinical_constants_instance.PHQ9_MAX_SCORE):
            raise ValueError(f"Baseline {self.baseline} outside valid PHQ-9 range [{clinical_constants_instance.PHQ9_MIN_SCORE}, {clinical_constants_instance.PHQ9_MAX_SCORE}]")

        if not (clinical_constants_instance.AR_COEFFICIENT_MIN <= self.ar_coefficient <= clinical_constants_instance.AR_COEFFICIENT_MAX):
            raise ValueError(f"AR coefficient {self.ar_coefficient} outside realistic range [{clinical_constants_instance.AR_COEFFICIENT_MIN}, {clinical_constants_instance.AR_COEFFICIENT_MAX}]. Literature: Kroenke et al. (2001) test-retest r={clinical_constants_instance.PHQ9_TEST_RETEST_RELIABILITY}")

        if (self.noise_std < 0):
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}")

        # Validate response pattern
        valid_patterns = list(RESPONSE_PATTERN_PROBABILITIES.keys())
        if self.response_pattern not in valid_patterns:
            raise ValueError(f"Invalid response_pattern '{self.response_pattern}'. Must be one of {valid_patterns}")


    def get_expected_score_at_day(self, day: int, enable_plateau: bool = True) -> float:
        """
        Calculate expected PHQ-9 score at a given day based on treatment trajectory

        Formula (without plateau):
        --------------------------
            E[Y_t] = baseline + recovery_rate * (t - treatment_start)

        Formula (with plateau):
        ----------------------
            If t < plateau_start  : E[Y_t] = baseline + recovery_rate * (t - treatment_start)
            If t >= plateau_start : E[Y_t] = plateau_score (constant)
        
        Arguments:
        ----------
            day            { int }  : Calendar day (1-indexed)
            
            enable_plateau { bool } : Whether to apply plateau logic
        
        Returns:
        --------
            { float } : Expected score [0, 27]
        """
        days_since_treatment = day - self.treatment_start_day

        # Check if in plateau phase
        if enable_plateau and (self.plateau_start_day is not None) and (day >= self.plateau_start_day):
            # In plateau: return stabilized score
            plateau_score  = self.baseline + (self.recovery_rate * (self.plateau_start_day - self.treatment_start_day))
            expected_score = plateau_score
        
        else:
            # Pre-plateau: linear trajectory
            expected_score = self.baseline + (self.recovery_rate * days_since_treatment)

        clipped_score = np.clip(expected_score, clinical_constants_instance.PHQ9_MIN_SCORE, clinical_constants_instance.PHQ9_MAX_SCORE)

        return clipped_score


    def check_and_enter_plateau(self, day: int):
        """
        Check if patient should enter plateau phase based on response pattern. Plateau timing generally depends on response pattern:
        - Early responders   : Plateau at 6 weeks
        - Gradual responders : Plateau at 10 weeks
        - Late responders    : Plateau at 16 weeks
        - Non-responders     : No plateau (continue slow trajectory)
        
        Arguments:
        ----------
            day { int } : Current day
        """
        if (self.in_plateau_phase or (self.plateau_start_day is not None)):
            # Already in plateau
            return  

        weeks_since_start = (day - self.treatment_start_day) / 7.0

        # Determine plateau timing based on response pattern
        if (self.response_pattern == 'early_responder'):
            plateau_weeks = clinical_constants_instance.EARLY_RESPONDER_PLATEAU_WEEKS
        
        elif (self.response_pattern == 'gradual_responder'):
            plateau_weeks = clinical_constants_instance.GRADUAL_RESPONDER_PLATEAU_WEEKS
        
        elif (self.response_pattern == 'late_responder'):
            plateau_weeks = clinical_constants_instance.LATE_RESPONDER_PLATEAU_WEEKS
        
        else: 
            # Non-responders don't plateau
            return  

        # Enter plateau if time reached
        if (weeks_since_start >= plateau_weeks):
            self.plateau_start_day = day
            self.in_plateau_phase  = True


    def get_effective_noise(self, day: int, enable_plateau: bool = True) -> float:
        """
        Get effective noise standard deviation, reduced during plateau phase: during plateau, symptom variability decreases (stabilization)
        
        Arguments:
        ----------
            day            { int }  : Current day
            
            enable_plateau { bool } : Whether to apply plateau logic
        
        Returns:
        --------
                  { float }         : Effective noise standard deviation
        """
        if not enable_plateau:
            return self.noise_std

        if (self.in_plateau_phase and (self.plateau_start_day is not None)):
            # Smooth transition into plateau over 2 weeks
            weeks_in_plateau = (day - self.plateau_start_day) / 7.0

            if (weeks_in_plateau >= clinical_constants_instance.PLATEAU_TRANSITION_WEEKS):
                # Full plateau: reduced noise
                return self.noise_std * clinical_constants_instance.PLATEAU_NOISE_REDUCTION_FACTOR
            
            else:
                # Transition phase: linearly interpolate noise reduction
                transition_progress = weeks_in_plateau / clinical_constants_instance.PLATEAU_TRANSITION_WEEKS
                noise_reduction     = 1.0 - (transition_progress * (1.0 - clinical_constants_instance.PLATEAU_NOISE_REDUCTION_FACTOR))
                return self.noise_std * noise_reduction
        
        else:
            return self.noise_std


    def update_last_observation(self, score: float, day: int):
        """
        Update the most recent observed score and day
        """
        self.last_score = score
        self.last_day   = day


    def add_relapse(self, day: int):
        """
        Record a relapse event on the specified day
        
        Arguments:
        ----------
            day { int } : Day when relapse occurred (1-indexed)
        """
        self.relapse_history.append(day)


    def get_summary(self) -> dict:
        """
        Get human-readable summary of trajectory
        """
        return {'baseline'         : f"{self.baseline:.1f}",
                'recovery_rate'    : f"{self.recovery_rate:.3f} points/day",
                'noise_std'        : f"{self.noise_std:.2f}",
                'ar_coefficient'   : f"{self.ar_coefficient:.2f}",
                'response_pattern' : self.response_pattern,
                'plateau_start'    : f"Day {self.plateau_start_day}" if self.plateau_start_day else "None",
                'in_plateau'       : self.in_plateau_phase,
                'n_relapses'       : len(self.relapse_history),
                'last_score'       : f"{self.last_score:.1f}" if self.last_score is not None else "None",
                'last_day'         : self.last_day,
               }



class AR1Model:
    """
    First-order autoregressive model for PHQ-9 score generation: Y_t = α^Δt * Y_{t-Δt} + (1 - α^Δt) * μ_t + ε_t + relapse_t

    This formulation handles irregular observation gaps, incorporates response pattern heterogeneity and plateau logic
    """
    def __init__(self):
        """
        Initialize AR(1) model        
        """
        pass


    def generate_score(self, trajectory: PatientTrajectory, day: int, relapse_probability: float = 0.10, relapse_magnitude_mean: float = 3.5, 
                       relapse_distribution: str = "exponential", enable_plateau: bool = True) -> float:
        """
        Generate PHQ-9 score for a specific day using a gap-aware AR(1) process:

        - This method is trajectory-stateful (state stored in PatientTrajectory)
        - Checks and enters plateau phase automatically
        - Reduces noise during plateau (symptom stabilization)
        - Soft boundary reflection to prevent artificial concentration at 0/27
        
        Arguments:
        ----------
            trajectory             { PatientTrajectory } : PatientTrajectory instance (modified in-place)
            
            day                           { int }        : Calendar day for score generation (1-indexed)
            
            relapse_probability          { float }       : Daily probability of relapse event
            
            relapse_magnitude_mean       { float }       : Mean magnitude of relapse (exponential distribution)

            relapse_distribution          { str }        : Distribution for relapse magnitude
                                                           - Exponential (default) has heavy tail; 
                                                           - Gamma is more bounded; 
                                                           - Lognormal has very heavy tail

            enable_plateau                { bool }       : Whether to apply plateau logic
        
        Returns:
        --------
                          { float }                      : Generated PHQ-9 score (float, clipped to [0, 27])
        """
        # Check if should enter plateau phase
        if enable_plateau:
            trajectory.check_and_enter_plateau(day)

        expected_score = trajectory.get_expected_score_at_day(day, enable_plateau = enable_plateau)

        # Gap-aware AR(1)
        if ((trajectory.last_score is not None) and (trajectory.last_day is not None)):
            delta_days      = day - trajectory.last_day
            effective_alpha = trajectory.ar_coefficient ** delta_days
            score           = (effective_alpha * trajectory.last_score + (1 - effective_alpha) * expected_score)
        
        else:
            score = expected_score

        # Measurement noise (reduced during plateau)
        effective_noise = trajectory.get_effective_noise(day, enable_plateau = enable_plateau)
        score          += trajectory.rng.normal(0, effective_noise)

        # Relapse event with configurable distribution
        if (trajectory.rng.random() < relapse_probability):
            if (relapse_distribution == 'exponential'):
                magnitude = trajectory.rng.exponential(scale = relapse_magnitude_mean)
            
            elif (relapse_distribution == 'gamma'):
                magnitude = trajectory.rng.gamma(shape = clinical_constants_instance.RELAPSE_MAGNITUDE_GAMMA_SHAPE, 
                                                 scale = relapse_magnitude_mean / clinical_constants_instance.RELAPSE_MAGNITUDE_GAMMA_SHAPE,
                                                )
            
            elif (relapse_distribution == 'lognormal'):
                sigma     = clinical_constants_instance.RELAPSE_MAGNITUDE_LOGNORMAL_SIGMA   
                mu        = np.log(relapse_magnitude_mean) - (sigma**2 / 2)
                magnitude = trajectory.rng.lognormal(mean  = mu, 
                                                     sigma = sigma,
                                                    )
            
            else:
                magnitude = trajectory.rng.exponential(scale = relapse_magnitude_mean)
            
            remaining_capacity = clinical_constants_instance.PHQ9_MAX_SCORE - score
            scaled_magnitude   = magnitude * (remaining_capacity / clinical_constants_instance.PHQ9_MAX_SCORE)

            score             += scaled_magnitude

            trajectory.add_relapse(day)

        # Soft boundary reflection before hard clipping
        if (score < clinical_constants_instance.PHQ9_MIN_SCORE):
            score = abs(score) * clinical_constants_instance.BOUNDARY_REFLECTION_FACTOR

        elif (score > clinical_constants_instance.PHQ9_MAX_SCORE):
            score = clinical_constants_instance.PHQ9_MAX_SCORE - (score - clinical_constants_instance.PHQ9_MAX_SCORE) * clinical_constants_instance.BOUNDARY_REFLECTION_FACTOR

        # Hard clip to valid PHQ-9 range
        score = float(np.clip(score, clinical_constants_instance.PHQ9_MIN_SCORE, clinical_constants_instance.PHQ9_MAX_SCORE))

        # Update trajectory state
        trajectory.update_last_observation(score, day)

        return score


    @staticmethod
    def calculate_autocorrelation(scores: np.ndarray, lag: int = 1) -> float:
        """
        Calculate autocorrelation for a series of scores
        """
        if (len(scores) <= lag):
            return np.nan

        y     = scores - np.mean(scores)
        denom = np.sum(y ** 2)

        if (denom == 0):
            return np.nan

        return np.sum(y[:-lag] * y[lag:]) / denom



# Convenience functions
def assign_response_pattern(rng: np.random.Generator) -> str:
    """
    Assign response pattern based on clinical probabilities
    
    Arguments:
    ----------
        rng { np.random.Generator } : Random Number Generator instance 
    
    Returns:
    --------
              { str }               : Response pattern ('early_responder', 'gradual_responder', 'late_responder', 'non_responder')
    """
    patterns = list(RESPONSE_PATTERN_PROBABILITIES.keys())
    probs    = list(RESPONSE_PATTERN_PROBABILITIES.values())

    return rng.choice(patterns, p = probs)


def adjust_recovery_rate_for_pattern(base_rate: float, pattern: str) -> float:
    """
    Adjust recovery rate based on response pattern: response patterns have different trajectories:

    - Early responders   : Faster initial improvement
    - Gradual responders : Steady improvement (baseline)
    - Late responders    : Slower initial improvement
    - Non-responders     : Minimal improvement
    
    Arguments:
    ----------
        base_rate { float } : Base recovery rate (e.g., -0.06)
        
        pattern   { str }   : Response pattern
    
    Returns:
    --------
            { float }       : Adjusted recovery rate
    """
    if (pattern == 'early_responder'):
        # 30% faster
        return base_rate * 1.3  

    elif (pattern == 'gradual_responder'):
        # Baseline
        return base_rate  

    elif (pattern == 'late_responder'):
        # 30% slower
        return base_rate * 0.7  

    elif (pattern == 'non_responder'):
        # Minimal improvement
        return base_rate * 0.3 
    
    else:
        return base_rate


def initialize_patient_trajectories(n_patients: int, baseline_mean: float, baseline_std: float, recovery_rate_mean: float, recovery_rate_std: float,
                                    noise_std: float, ar_coefficient: float, random_seed: Optional[int] = None, 
                                    enable_response_patterns: bool = True) -> dict:
    """
    Initialize trajectories for all patients in the study: assigns heterogeneous response patterns (early/gradual/late/non-responders)
    """
    trajectories = dict()

    # Initialize SeedSequence
    ss           = SeedSequence(random_seed)
    child_seeds  = ss.spawn(n_patients)

    for patient_id in range(1, n_patients + 1):
        patient_rng  = np.random.default_rng(child_seeds[patient_id - 1])
        
        # Sample baseline with reflection-based clipping (NEW)
        raw_baseline = patient_rng.normal(baseline_mean, baseline_std)
        
        # Symmetric reflection at boundaries
        if (raw_baseline < clinical_constants_instance.PHQ9_MIN_SCORE):
            baseline = clinical_constants_instance.PHQ9_MIN_SCORE + abs(raw_baseline - clinical_constants_instance.PHQ9_MIN_SCORE) * clinical_constants_instance.BOUNDARY_REFLECTION_FACTOR

        elif (raw_baseline > clinical_constants_instance.PHQ9_MAX_SCORE):
            baseline = clinical_constants_instance.PHQ9_MAX_SCORE - (raw_baseline - clinical_constants_instance.PHQ9_MAX_SCORE) * clinical_constants_instance.BOUNDARY_REFLECTION_FACTOR

        else:
            # Within bounds: soft clipping to preserve mean
            baseline = np.clip(a     = raw_baseline,
                               a_min = max(clinical_constants_instance.PHQ9_MIN_SCORE, baseline_mean - clinical_constants_instance.BASELINE_CLIP_LOWER_STD_MULTIPLIER * baseline_std),
                               a_max = clinical_constants_instance.BASELINE_CLIP_UPPER_LIMIT
                              )

        # Assign response pattern
        if enable_response_patterns:
            response_pattern = assign_response_pattern(patient_rng)

        else:
            response_pattern = 'gradual_responder'

        # Base recovery rate
        base_recovery_rate = patient_rng.normal(loc   = recovery_rate_mean, 
                                                scale = recovery_rate_std,
                                               )

        # Adjust for response pattern
        if enable_response_patterns:
            recovery_rate = adjust_recovery_rate_for_pattern(base_recovery_rate, response_pattern)
        
        else:
            recovery_rate = base_recovery_rate

        # Individual noise (gamma distribution for positive skew)
        individual_noise         = np.clip(a     = patient_rng.gamma(shape = 4, scale = noise_std / 4),
                                           a_min = 1.0,
                                           a_max = 4.0,
                                          )

        # Individual AR coefficient
        ar_coef                  = np.clip(a     = patient_rng.normal(ar_coefficient, clinical_constants_instance.AR_COEFFICIENT_INDIVIDUAL_STD),
                                           a_min = clinical_constants_instance.AR_COEFFICIENT_MIN,
                                           a_max = clinical_constants_instance.AR_COEFFICIENT_MAX,
                                          )

        trajectories[patient_id] = PatientTrajectory(baseline         = baseline,
                                                     recovery_rate    = recovery_rate,
                                                     noise_std        = individual_noise,
                                                     ar_coefficient   = ar_coef,
                                                     response_pattern = response_pattern,
                                                     rng              = patient_rng,
                                                    )

    return trajectories