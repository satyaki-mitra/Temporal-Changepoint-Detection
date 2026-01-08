# Dependencies
import numpy as np
from typing import Dict
from typing import Optional
from phq9_analysis.detection.statistical_tests import BOCPDStatisticalValidator


class BOCPDDetector:
    """
    Online Bayesian Change Point Detection (BOCPD)

    Implements Adams & MacKay (2007) with:
    - Constant hazard function
    - Gaussian likelihood
    - Run-length posterior
    """
    def __init__(self, config):
        """
        Initialize BOCPD detector

        Arguments:
        ----------
            config { ChangePointDetectionConfig } : Unified detection configuration
        """
        self.config = config


    def detect(self, signal: np.ndarray) -> Dict:
        """
        Run BOCPD on aggregated signal

        Arguments:
        ----------
            signal { np.ndarray } : Aggregated 1D statistic

        Returns:
        --------
                 { dict }         : BOCPD detection results
        """
        T            = len(signal)
        max_r        = self.config.max_run_length

        # Run-length posterior
        R            = np.zeros((T, max_r))
        R[0, 0]      = 1.0
        cp_posterior = np.zeros(T)

        hazard       = 1.0 / self.config.hazard_lambda

        mu0          = self.config.prior_mean
        var0         = self.config.prior_variance

        means        = np.full(max_r, mu0)
        vars_        = np.full(max_r, var0)

        for t in range(1, T):
            x               = signal[t]

            # Predictive likelihood (Gaussian)
            pred_var        = vars_ + 1e-6
            pred_prob       = (1.0 / np.sqrt(2 * np.pi * pred_var)* np.exp(-0.5 * (x - means) ** 2 / pred_var))

            # Growth probabilities
            growth          = R[t-1, :max_r-1] * pred_prob[:max_r-1] * (1 - hazard)

            # Change point probability
            cp_prob         = np.sum(R[t-1, :] * pred_prob * hazard)

            R[t, 1:]        = growth
            R[t, 0]         = cp_prob

            # Normalize
            R[t]           /= np.sum(R[t])

            cp_posterior[t] = R[t, 0]

            # Update sufficient statistics
            means           = np.roll(means, 1)
            vars_           = np.roll(vars_, 1)

            means[0]        = mu0
            vars_[0]        = var0
            means[1:]      += (x - means[1:]) / np.arange(1, max_r)

        # Optional posterior smoothing
        if (self.config.posterior_smoothing > 1):
            w            = self.config.posterior_smoothing
            cp_posterior = np.convolve(a    = cp_posterior,
                                       v    = np.ones(w) / w,
                                       mode = 'same',
                                      )

        return {'method'               : 'bocpd',
                'variant'              : f"bocpd_{self.config.likelihood_model}",
                'run_length_posterior' : R,
                'cp_posterior'         : cp_posterior,
               }


    def validate(self, cp_posterior: np.ndarray) -> Dict:
        """
        Validate BOCPD change points via posterior persistence.

        Uses:
        - Posterior threshold
        - Consecutive persistence criterion
        """
        validator               = BOCPDStatisticalValidator(posterior_threshold = self.config.cp_posterior_threshold,
                                                            persistence         = 3,
                                                           )

        validated_change_points = validator.validate(cp_posterior = cp_posterior)

        return validated_change_points