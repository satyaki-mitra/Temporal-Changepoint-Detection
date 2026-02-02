# Dependencies
import numpy as np
from typing import Dict, Optional
from scipy.ndimage import gaussian_filter1d
from scipy.stats import t as student_t
from src.detection.statistical_tests import BOCPDStatisticalValidator


class BOCPDDetector:
    """
    Online Bayesian Change Point Detection (BOCPD) - FIXED VERSION
    
    Implements Adams & MacKay (2007) with:
    - Student-t likelihood (df=3-5) for heavy-tailed data
    - Improved hazard defaults (λ=100, range=[50,500])
    - Posterior threshold (0.15)
    - Numerical stability
    """
    def __init__(self, config, logger: Optional = None):
        """
        Initialize BOCPD detector
        
        Arguments:
        ----------
            config { ChangePointDetectionConfig } : Unified detection configuration

            logger { logging.Logger | None }      : Optional logger
        """
        self.config = config
        self.logger = logger
        
        # Validate likelihood model
        if config.likelihood_model not in {'gaussian', 'student_t'}:
            raise ValueError(f"Unsupported likelihood: {config.likelihood_model}. Use 'gaussian' or 'student_t'")

    def detect(self, signal: np.ndarray, hazard_lambda: Optional[float] = None) -> Dict:
        """
        Run BOCPD on aggregated signal
        
        Arguments:
        ----------
            signal        { np.ndarray } : 1D aggregated statistic (e.g., daily CV)

            hazard_lambda    { float }   : Expected run length (λ)
            
        Returns:
        --------
                    { dict }             : Detection + validation outputs
        """
        T = len(signal)
        
        if (T < 5):
            return self._return_insufficient_data(T             = T, 
                                                  hazard_lambda = hazard_lambda,
                                                 )
        
        # Setup
        max_r         = min(self.config.max_run_length, T)
        log_R         = np.full((T, max_r), -np.inf)
        log_R[0, 0]   = 0.0
        cp_posterior  = np.zeros(T)
        
        # Hazard rate
        hazard_lambda = hazard_lambda if hazard_lambda is not None else self.config.hazard_lambda
        hazard_rate   = min(1.0 / hazard_lambda, 0.99)
        
        # Empirical Bayes prior
        mu0           = float(np.mean(signal))
        var0          = max(float(np.var(signal)), 1e-8)
        
        # Sufficient statistics
        means         = np.full(max_r, mu0)
        vars_         = np.full(max_r, var0)
        counts        = np.ones(max_r)
        
        # Progress logging
        step          = max(1, T // 20)
        
        # BOCPD recursion
        for t in range(1, T):
            if self.logger and (t % step == 0):
                self.logger.info(f"BOCPD progress: {t}/{T}")
            
            valid_r        = min(t, max_r - 1)
            x              = signal[t]
            
            # Welford's online update for sufficient statistics
            updated_means  = np.copy(means)
            updated_vars   = np.copy(vars_)
            updated_counts = np.copy(counts)
            
            for r in range(valid_r):
                n_old             = counts[r]
                n_new             = n_old + 1
                delta             = x - means[r]
                updated_means[r]  = means[r] + delta / n_new
                delta2            = x - updated_means[r]
                updated_vars[r]   = (n_old * vars_[r] + delta * delta2) / n_new
                updated_counts[r] = n_new
            
            # Compute predictive log-likelihood (Student-t or Gaussian)
            log_pred                 = self._compute_predictive_log_likelihood(x      = x,
                                                                               means  = means[:valid_r + 1],
                                                                               vars_  = vars_[:valid_r + 1],
                                                                               counts = counts[:valid_r + 1],
                                                                              )
            
            # Log growth probabilities (no change point)
            log_growth              = log_R[t-1, :valid_r] + log_pred[:valid_r] + np.log(1.0 - hazard_rate)
            
            # Predictive likelihood under a NEW segment (r_t = 0)
            log_pred_cp             = self._compute_predictive_log_likelihood(x      = x,
                                                                              means  = np.array([mu0]),
                                                                              vars_  = np.array([var0]),
                                                                              counts = np.array([1.0]),
                                                                             )[0]
            
            # Log change point probability
            log_cp                  = np.log(hazard_rate) + log_pred_cp
            
            # Update run-length posterior
            log_R[t, 1:valid_r + 1] = log_growth
            log_R[t, 0]             = log_cp
            
            # Normalize in log-space
            log_norm                = np.logaddexp.reduce(log_R[t, :valid_r + 1])
            log_R[t, :valid_r + 1] -= log_norm
            
            # Change point posterior = P(r_t = 0)
            cp_posterior[t]         = np.exp(log_cp - log_norm)
            
            # Roll sufficient statistics
            means                   = np.roll(updated_means, 1)
            vars_                   = np.roll(updated_vars, 1)
            counts                  = np.roll(updated_counts, 1)
            
            # Reset r=0 to prior
            means[0]                = mu0
            vars_[0]                = var0
            counts[0]               = 1
            
            # Numerical stability
            vars_                   = np.maximum(vars_, 1e-8)
        
        # Optional smoothing
        if (self.config.posterior_smoothing > 1):
            cp_posterior = gaussian_filter1d(cp_posterior,
                                             sigma = self.config.posterior_smoothing,
                                             mode  = 'nearest',
                                            )

        # Adaptive BOCPD posterior threshold 
        mu                 = float(cp_posterior.mean())
        sigma              = max(float(cp_posterior.std()), 1e-6)
        
        # Compute data adaptive threshold
        adaptive_threshold = mu + (self.config.cp_posterior_threshold_multiplier * sigma)
        adaptive_threshold = float(np.clip(adaptive_threshold, 0.01, 0.99))
        
        # Diagnostics
        self._log_diagnostics(cp_posterior        = cp_posterior, 
                              adaptive_threshold  = adaptive_threshold,
                              hazard_lambda       = hazard_lambda, 
                              hazard_rate         = hazard_rate,
                             )
        
        # Validation
        validator  = BOCPDStatisticalValidator(posterior_threshold = adaptive_threshold,
                                               persistence         = self.config.bocpd_persistence,
                                              )

        validation = validator.validate(cp_posterior)

        # Reset-on-CP (semantic, not recursive) 
        if (self.config.reset_on_cp and (validation['n_changepoints'] > 0)):
            if self.logger:
                self.logger.info(f"Reset-on-CP enabled: {validation['n_changepoints']} change points detected")
                
        return {'method'               : 'bocpd',
                'algorithm'            : 'BOCPD',
                'variant'              : f"bocpd_{self.config.likelihood_model}",
                'likelihood_model'     : self.config.likelihood_model,
                'likelihood_df'        : getattr(self.config, 'likelihood_df', None),
                'hazard_lambda'        : hazard_lambda,
                'hazard_rate'          : hazard_rate,
                'signal_length'        : int(T),
                'max_run_length_used'  : int(max_r),
                'posterior_smoothing'  : int(self.config.posterior_smoothing),
                'cp_posterior'         : cp_posterior,
                'run_length_posterior' : np.exp(log_R),
                'adaptive_threshold'   : adaptive_threshold,
                'change_points'        : validation.get('normalized_positions', []),
                'n_changepoints'       : validation['n_changepoints'],
                'validation'           : validation,
               }
    

    def _compute_predictive_log_likelihood(self, x: float, means: np.ndarray, vars_: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """
        Compute predictive log-likelihood using Student-t or Gaussian
        
        For Student-t:
        - More robust to outliers
        - Heavy tails handle CV spikes from relapses
        - df=3-5 recommended for CV data
        """
        if (self.config.likelihood_model == 'student_t'):
            # Student-t predictive distribution
            df       = self.config.likelihood_df
            pred_var = vars_ * (1.0 + 1.0 / counts) + 1e-8
            
            # Student-t log PDF
            log_pred = student_t.logpdf(x,
                                        df    = df,
                                        loc   = means,
                                        scale = np.sqrt(pred_var),
                                       )
            
        else:
            # Gaussian predictive distribution (original)
            pred_var  = vars_ * (1.0 + 1.0 / counts) + 1e-8
            log_pred  = -0.5 * np.log(2 * np.pi * pred_var)
            log_pred -= 0.5 * (x - means) ** 2 / pred_var
        
        return log_pred
    

    def _log_diagnostics(self, cp_posterior: np.ndarray, adaptive_threshold: float, hazard_lambda: float, hazard_rate: float):
        """
        Diagnostics with warnings
        """
        if not self.logger:
            return
        
        max_post  = cp_posterior.max()
        mean_post = cp_posterior.mean()
        threshold = adaptive_threshold
        n_above   = (cp_posterior > threshold).sum()
        
        self.logger.info(f"BOCPD diagnostics:")
        self.logger.info(f"  Likelihood: {self.config.likelihood_model}")
        if (self.config.likelihood_model == 'student_t'):
            self.logger.info(f"  Student-t df: {self.config.likelihood_df}")
        
        self.logger.info(f"  Hazard λ: {hazard_lambda:.1f}, rate: {hazard_rate:.4f}")
        self.logger.info(f"  Max CP posterior: {max_post:.4f}")
        self.logger.info(f"  Mean CP posterior: {mean_post:.4f}")
        self.logger.info(f"  Std CP posterior: {cp_posterior.std():.4f}")
        self.logger.info(f"  Threshold: {threshold:.4f}")
        self.logger.info(f"  Points above threshold: {n_above}")
        
        # Warnings for potential issues
        if (max_post < threshold):
            self.logger.warning(f"Max posterior ({max_post:.4f}) < threshold ({threshold:.4f}) - no CPs will be detected!")

        if (hazard_lambda < 30):
            self.logger.warning(f"λ={hazard_lambda:.1f} is very aggressive (expects CP every {hazard_lambda:.0f} days)")
        
        # Show top posteriors
        top_indices = np.argsort(cp_posterior)[-10:][::-1]
        self.logger.info(f"  Top 10 posteriors: {cp_posterior[top_indices]}")
        self.logger.info(f"  At time indices: {top_indices.tolist()}")
    

    def _return_insufficient_data(self, T: int, hazard_lambda: Optional[float]) -> Dict:
        """
        Return structure for insufficient data
        """
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
                'validation'               : {'overall_significant': False,
                                              'rejection_reason': f'Signal too short for BOCPD (length={T}, minimum=5)'
                                             },
               }
