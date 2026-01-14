# Dependencies
import numpy as np
from typing import Dict
from typing import Optional
from scipy.ndimage import gaussian_filter1d
from src.detection.statistical_tests import BOCPDStatisticalValidator


class BOCPDDetector:
    """
    Online Bayesian Change Point Detection (BOCPD)

    Implements Adams & MacKay (2007) with:
    - Constant hazard function h = 1 / λ
    - Gaussian likelihood (only)
    - Numerically stable log-space recursion
    - Online mean and variance updates (Welford)

    This class:
    -----------
    - Assumes hazard λ is already chosen
    - Does NOT perform hazard tuning
    """
    def __init__(self, config, logger: Optional = None):
        """
        Initialize BOCPD detector

        Arguments:
        ----------
            config { ChangePointDetectionConfig } : Unified detection configuration

            logger   { logging.Logger | None }    : Optional logger
        """
        self.config = config
        self.logger = logger


    def detect(self, signal: np.ndarray, hazard_lambda: Optional[float] = None) -> Dict:
        """
        Run BOCPD on aggregated signal

        Arguments:
        ----------
            signal        { np.ndarray } : 1D aggregated statistic (e.g., daily CV)

            hazard_lambda   { float }    : Expected run length (λ)

        Returns:
        --------
                    { dict }             : Detection + validation outputs
        """
        T             = len(signal)

        if (T < 5):
            return {'method'                   : 'bocpd',
                    'algorithm'                : 'BOCPD',
                    'variant'                  : f"bocpd_{self.config.likelihood_model}",
                    'hazard_lambda'            : hazard_lambda,
                    'hazard_rate'              : None,
                    'hazard_source'            : 'auto' if self.config.auto_tune_hazard else 'fixed',
                    'hazard_range_used'        : self.config.hazard_range if self.config.auto_tune_hazard else None,
                    'signal_length'            : int(T),
                    'max_run_length_used'      : None,
                    'posterior_smoothing'      : int(self.config.posterior_smoothing),
                    'cp_posterior'             : None,
                    'run_length_posterior'     : None,
                    'change_points'            : [],
                    'change_points_normalized' : [],
                    'n_changepoints'           : 0,
                    'validation'               : {'overall_significant' : False,
                                                  'rejection_reason'    : f'Signal too short for BOCPD (length={T}, minimum=5)',
                                                 }
                   }

        # Get the Run Length       
        max_r         = min(self.config.max_run_length, T)

        # Log-space run-length posterior
        log_R         = np.full((T, max_r), -np.inf)
        log_R[0, 0]   = 0.0

        cp_posterior  = np.zeros(T)

        # Hazard rate
        hazard_lambda = hazard_lambda if hazard_lambda is not None else self.config.hazard_lambda
        hazard_rate   = min(1.0 / hazard_lambda, 0.99)

        # Empirical Bayes prior
        mu0           = float(np.mean(signal))
        var0          = max(float(np.var(signal)), 1e-8)

        means         = np.full(max_r, mu0)
        vars_         = np.full(max_r, var0)
        counts        = np.ones(max_r)

        step          = max(1, T // 20)

        for t in range(1, T):
            if (self.logger and (t % step == 0)):
                self.logger.info(f"BOCPD progress: {t}/{T}")

            valid_r        = min(t, max_r - 1)
            x              = signal[t]

            # Compute sufficient statistics before rolling and update means and variances for all active run lengths
            updated_means  = np.copy(means)
            updated_vars   = np.copy(vars_)
            updated_counts = np.copy(counts)

            # Welford's online update for run lengths [0, valid_r)
            for r in range(valid_r):
                n_old             = counts[r]
                n_new             = n_old + 1
                
                delta             = x - means[r]
                updated_means[r]  = means[r] + delta / n_new
                
                delta2            = x - updated_means[r]
                updated_vars[r]   = (n_old * vars_[r] + delta * delta2) / n_new
                
                updated_counts[r] = n_new

            # log-space posterior update : predictive variance (adding observation noise)
            pred_var                = vars_[:valid_r + 1] + 1e-8

            # Log predictive likelihood
            log_pred                = (- 0.5 * np.log(2 * np.pi * pred_var) - 0.5 * (x - means[:valid_r + 1]) ** 2 / pred_var)

            # Log growth probabilities (no change point)
            log_growth              = (log_R[t-1, :valid_r] + log_pred[:valid_r] + np.log(1.0 - hazard_rate))

            # Log change point probability
            log_cp                  = (np.log(hazard_rate) + np.logaddexp.reduce(log_R[t-1, :valid_r + 1] + log_pred))

            # Update run-length posterior
            log_R[t, 1:valid_r + 1] = log_growth
            log_R[t, 0]             = log_cp

            # Normalize in log-space
            log_norm                = np.logaddexp.reduce(log_R[t, :valid_r + 1])
            log_R[t, :valid_r + 1] -= log_norm

            # Change point posterior (probability of r = 0)
            cp_posterior[t]         = np.exp(log_R[t, 0])

            # Roll sufficient statistics after updates
            means                   = np.roll(updated_means, 1)
            vars_                   = np.roll(updated_vars, 1)
            counts                  = np.roll(updated_counts, 1)

            # Reset r = 0 to prior
            means[0]                = mu0
            vars_[0]                = var0
            counts[0]               = 1

            # Ensure numerical stability
            vars_                   = np.maximum(vars_, 1e-8)

        # Optional smoothing
        if (self.config.posterior_smoothing > 1):
            cp_posterior = gaussian_filter1d(cp_posterior,
                                             sigma = self.config.posterior_smoothing,
                                             mode  = 'nearest',
                                            )

        # Validation
        validator  = BOCPDStatisticalValidator(posterior_threshold = self.config.cp_posterior_threshold,
                                               persistence         = self.config.bocpd_persistence,
                                              )
                                              
        validation = validator.validate(cp_posterior)
 
        return {'method'               : 'bocpd',
                'algorithm'            : 'BOCPD',
                'variant'              : f"bocpd_{self.config.likelihood_model}",
                'hazard_lambda'        : hazard_lambda,
                'hazard_rate'          : hazard_rate,
                'signal_length'        : int(T),
                'max_run_length_used'  : int(max_r),
                'posterior_smoothing'  : int(self.config.posterior_smoothing),
                'cp_posterior'         : cp_posterior,
                'run_length_posterior' : np.exp(log_R),
                'change_points'        : validation.get('normalized_positions', []),
                'n_changepoints'       : validation['n_changepoints'],
                'validation'           : validation,
               }