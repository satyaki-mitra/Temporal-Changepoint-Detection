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


    def detect(self, signal: np.ndarray) -> Dict:
        """
        Run BOCPD on aggregated signal

        Returns:
        --------
            { dict } : Detection + validation outputs
        """
        T             = len(signal)
        max_r         = self.config.max_run_length

        # Log-space run-length posterior
        log_R         = np.full((T, max_r), -np.inf)
        log_R[0, 0]   = 0.0

        cp_posterior  = np.zeros(T)

        # Hazard rate
        hazard_lambda = self.config.hazard_lambda
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

            valid_r                  = min(t, max_r - 1)
            x                        = signal[t]

            pred_var                 = vars_[:valid_r + 1] + 1e-8
            log_pred                 = (- 0.5 * np.log(2 * np.pi * pred_var) - 0.5 * (x - means[:valid_r + 1]) ** 2 / pred_var)

            log_pred                -= np.max(log_pred)

            log_growth               = (log_R[t-1, :valid_r] + log_pred[:valid_r] + np.log(1.0 - hazard_rate))

            log_cp                   = (np.log(hazard_rate) + np.logaddexp.reduce(log_R[t-1, :valid_r + 1] + log_pred))

            log_R[t, 1:valid_r + 1]  = log_growth
            log_R[t, 0]              = log_cp
            log_R[t]                -= np.logaddexp.reduce(log_R[t])

            cp_posterior[t]          = np.exp(log_R[t, 0])

            # Update sufficient statistics
            new_means                = np.roll(means, 1)
            new_vars                 = np.roll(vars_, 1)
            new_counts               = np.roll(counts, 1)

            new_means[0]             = mu0
            new_vars[0]              = var0
            new_counts[0]            = 1

            delta                    = x - means[:valid_r]
            new_means[1:valid_r + 1] = means[:valid_r] + delta / (counts[:valid_r] + 1)

            delta2                   = x - new_means[1:valid_r + 1]
            new_vars[1:valid_r + 1]  = ((counts[:valid_r] * vars_[:valid_r] + delta * delta2) / (counts[:valid_r] + 1))

            means                    = new_means
            vars_                    = np.maximum(new_vars, 1e-8)
            counts                   = new_counts

        # Optional smoothing
        if (self.config.posterior_smoothing > 1):
    
            cp_posterior = gaussian_filter1d(cp_posterior,
                                             sigma = self.config.posterior_smoothing,
                                             mode  = 'nearest',
                                            )

        # Validation
        validator  = BOCPDStatisticalValidator(posterior_threshold = self.config.cp_posterior_threshold,
                                               persistence         = 3,
                                              )
                                              
        validation = validator.validate(cp_posterior)
 
        return {'method'               : 'bocpd',
                'variant'              : f"bocpd_{self.config.likelihood_model}",
                'hazard_lambda'        : hazard_lambda,
                'hazard_rate'          : hazard_rate,
                'cp_posterior'         : cp_posterior,
                'run_length_posterior' : np.exp(log_R),
                'n_changepoints'       : validation['n_changepoints'],
                'validation'           : validation,
               }