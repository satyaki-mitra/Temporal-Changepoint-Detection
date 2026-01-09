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
    that determine a patient's depression symptom trajectory over time.

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
    last_day            : Optional[int]   = None
    relapse_history     : List[int]       = field(default_factory = list)
    treatment_start_day : int             = 1


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
        Calculate expected PHQ-9 score at a given day based on treatment trajectory.

        Formula:
        --------
            E[Y_t] = baseline + recovery_rate * (t - treatment_start)
        """
        days_since_treatment = day - self.treatment_start_day
        expected_score       = self.baseline + (self.recovery_rate * days_since_treatment)
        clipped_score        = np.clip(expected_score, 0.0, 27.0)

        return clipped_score


    def update_last_observation(self, score: float, day: int):
        """
        Update the most recent observed score and day.
        """
        self.last_score = score
        self.last_day   = day


    def add_relapse(self, day: int):
        """
        Record a relapse event.
        """
        self.relapse_history.append(day)


    def get_summary(self) -> dict:
        """
        Get human-readable summary of trajectory.
        """
        return {'baseline'       : f"{self.baseline:.1f}",
                'recovery_rate'  : f"{self.recovery_rate:.3f} points/day",
                'noise_std'      : f"{self.noise_std:.2f}",
                'ar_coefficient' : f"{self.ar_coefficient:.2f}",
                'n_relapses'     : len(self.relapse_history),
                'last_score'     : f"{self.last_score:.1f}" if self.last_score is not None else "None",
                'last_day'       : self.last_day,
               }



class AR1Model:
    """
    First-order autoregressive model for PHQ-9 score generation:

        Y_t = α^Δt * Y_{t-Δt} + (1 - α^Δt) * μ_t + ε_t + relapse_t

    This formulation correctly handles irregular observation gaps.
    """
    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)


    def generate_score(self, trajectory: PatientTrajectory, day: int, relapse_probability: float = 0.10, relapse_magnitude_mean: float = 3.5) -> float:
        """
        Generate PHQ-9 score for a specific day using a gap-aware AR(1) process.
        """
        expected_score = trajectory.get_expected_score_at_day(day)

        # Gap-aware AR(1)
        if ((trajectory.last_score is not None) and (trajectory.last_day is not None)):
            delta_days      = day - trajectory.last_day
            effective_alpha = trajectory.ar_coefficient ** delta_days

            score           = (effective_alpha * trajectory.last_score + (1 - effective_alpha) * expected_score)

        else:
            score = expected_score

        # Measurement noise
        score += np.random.normal(0, trajectory.noise_std)

        # Relapse event
        if (np.random.random() < relapse_probability):
            score += np.random.exponential(scale = relapse_magnitude_mean)
            trajectory.add_relapse(day)

        # Clip to valid PHQ-9 range
        score = float(np.clip(score, 0, 27))

        # Update trajectory state
        trajectory.update_last_observation(score, day)

        return score


    @staticmethod
    def calculate_autocorrelation(scores: np.ndarray, lag: int = 1) -> float:
        """
        Calculate autocorrelation for a series of scores.
        """
        if (len(scores) <= lag):
            return np.nan

        y     = scores - np.mean(scores)
        denom = np.sum(y ** 2)

        if (denom == 0):
            return np.nan

        return np.sum(y[:-lag] * y[lag:]) / denom



# Convenience functions
def initialize_patient_trajectories(n_patients: int, baseline_mean: float, baseline_std: float, recovery_rate_mean: float, recovery_rate_std: float,
                                    noise_std: float, ar_coefficient: float, random_seed: Optional[int] = None) -> dict:
    """
    Initialize trajectories for all patients in the study.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    trajectories = dict()

    for patient_id in range(1, n_patients + 1):
        baseline                 = np.clip(a     = np.random.normal(baseline_mean, baseline_std),
                                           a_min = 0,
                                           a_max = 27
                                          )

        recovery_rate            = np.random.normal(loc   = recovery_rate_mean,
                                                    scale = recovery_rate_std,
                                                   )

        individual_noise         = np.clip(a     = np.random.gamma(shape = 4, scale = noise_std / 4),
                                           a_min = 1.0,
                                           a_max = 4.0
                                          )

        ar_coef                  = np.clip(a     = np.random.normal(ar_coefficient, 0.05),
                                           a_min = 0.5,
                                           a_max = 0.9,
                                          )

        trajectories[patient_id] = PatientTrajectory(baseline       = baseline,
                                                     recovery_rate  = recovery_rate,
                                                     noise_std      = individual_noise,
                                                     ar_coefficient = ar_coef,
                                                    )

    return trajectories