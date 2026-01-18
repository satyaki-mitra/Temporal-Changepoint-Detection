# Dependencies
import numpy as np
import pandas as pd
from typing import Dict
from typing import Optional


class BOCPDHazardTuner:
    """
    BOCPD hazard (λ) tuner using various methods for tuning λ

    Supported methods:
    ------------------
    - 'heuristic'     : Deterministic, fast, production-safe: multiple heuristic strategies (uniform, autocorr, variance)
    - 'predictive_ll' : Time-series CV using Gaussian plug-in predictive likelihood

    Notes:
    ------
    - λ is the expected run length
    - Hazard rate h = 1 / λ
    - This tuner ONLY selects λ; it does NOT perform detection
    """
    def __init__(self, config, method: str = 'heuristic'):
        """
        Initialize hazard tuner

        Arguments:
        ----------
            config { ChangePointDetectionConfig } : Detection configuration
            
            method           { str }              : Tuning method
        """
        if method not in {'heuristic', 'predictive_ll'}:
            raise ValueError(f"Unknown hazard tuning method: {method}")

        self.config = config
        self.method = method


    def tune(self, signal: np.ndarray) -> Dict:
        """
        Tune hazard parameter λ (expected run length)

        Returns:
        --------
            { dict } : Canonical hazard tuning result
        """
        if (signal.ndim != 1):
            raise ValueError("signal must be 1D")

        if (len(signal) < 20):
            raise ValueError(f"Signal too short for hazard tuning (length={len(signal)}, minimum=20)")

        if (self.method == 'heuristic'):
            return self._tune_heuristic(signal = signal)

        return self._tune_predictive_ll(signal = signal)


    # HEURISTIC TUNER (DEFAULT, FAST)
    def _tune_heuristic(self, signal: np.ndarray, target_n_cps: Optional[int] = None, strategy: str = 'autocorr') -> Dict:
        """
        Heuristic hazard tuning

        FIXED: Proper target_n_cps calculation and bounds enforcement

        Strategies:
        -----------
        - 'uniform'  : Assume uniform spacing → λ ≈ T / (k + 1)
        - 'autocorr' : Use autocorrelation to estimate run length
        - 'variance' : Use variance changepoints as proxy
        """
        T             = len(signal)
        lo, hi        = self.config.hazard_range

        if (strategy == 'uniform'):
            # Better default target based on signal length
            if target_n_cps is None:
                # For 365-day signal, expect 3-6 change points → λ ≈ 60-120
                target_n_cps = max(3, min(6, T // 60))

            # Ensure lambda is within bounds
            raw_lambda    = T / (target_n_cps + 1)
            hazard_lambda = float(np.clip(raw_lambda, lo, hi))

            notes         = f"Uniform spacing: λ ≈ T/(k+1) = {T}/({target_n_cps}+1) = {raw_lambda:.1f}, clipped to [{lo}, {hi}] → {hazard_lambda:.1f}"

        elif (strategy == 'autocorr'):
            # More robust autocorrelation-based heuristic
            if (len(signal) > 5):
                # Use lag-1 autocorrelation
                acf_1         = np.corrcoef(signal[:-1], signal[1:])[0, 1]
                acf_1         = np.clip(acf_1, -0.99, 0.99)

                # Expected run length based on autocorrelation decay
                # If ACF is high (e.g., 0.9), changes are rare → high λ
                # If ACF is low (e.g., 0.1), changes are common → low λ
                if (acf_1 > 0):
                    raw_lambda = -1.0 / np.log(acf_1) if acf_1 > 0.01 else T / 4
                
                else:
                    raw_lambda = T / 4

                hazard_lambda = float(np.clip(raw_lambda, lo, hi))
                notes         = f"Autocorrelation-based: ACF(1)={acf_1:.3f} → raw λ={raw_lambda:.1f}, clipped to [{lo}, {hi}] → {hazard_lambda:.1f}"
            
            else:
                # Fallback to uniform
                hazard_lambda = float(np.clip(T / 4, lo, hi))
                notes         = "Autocorr failed (signal too short) - fallback to uniform"

        elif (strategy == 'variance'):
            # Variance-based heuristic
            if (len(signal) >= 10):
                rolling_var = pd.Series(signal).rolling(window=5, center=True).var().fillna(method='bfill').fillna(method='ffill')
                var_diff    = np.abs(np.diff(rolling_var))
                
                # Find rough change points
                threshold   = np.percentile(var_diff, 75)
                rough_cps   = np.where(var_diff > threshold)[0]

                if (len(rough_cps) > 1):
                    # Calculate average segment length
                    segments      = np.diff([0] + list(rough_cps) + [T])
                    avg_segment   = float(np.median(segments))  # Median for robustness
                    raw_lambda    = avg_segment
                    hazard_lambda = float(np.clip(raw_lambda, lo, hi))
                    notes         = f"Variance-based: {len(rough_cps)} rough CPs → median segment={avg_segment:.1f}, clipped to [{lo}, {hi}] → {hazard_lambda:.1f}"
                
                else:
                    # Fallback: assume ~4-5 change points
                    hazard_lambda = float(np.clip(T / 5, lo, hi))
                    notes         = "Variance-based: no clear variance changes - fallback to T/5"
            else:
                hazard_lambda = float(np.clip(T / 4, lo, hi))
                notes         = "Variance-based failed (signal too short) - fallback"

        else:
            raise ValueError(f"Unknown heuristic strategy: {strategy}")

        # Validation check
        if (hazard_lambda < 10):
            notes += f" WARNING: λ={hazard_lambda:.1f} is very low (expecting CP every {hazard_lambda:.0f} days)"

        return {'optimal_hazard_lambda' : hazard_lambda,
                'optimal_hazard_rate'   : 1.0 / hazard_lambda,
                'lambdas_tested'        : [hazard_lambda],
                'scores'                : None,
                'criterion'             : f'heuristic_{strategy}',
                'method'                : 'heuristic',
                'strategy'              : strategy,
                'notes'                 : notes,
               }


    # PREDICTIVE LOG-LIKELIHOOD TUNER (OFFLINE / ANALYSIS) 
    @staticmethod
    def _gaussian_predictive_ll(train: np.ndarray, test: np.ndarray) -> float:
        """
        Gaussian plug-in predictive log-likelihood
        """
        if (len(train) < 2):
            return -np.inf

        mu  = np.mean(train)
        var = np.var(train) + 1e-8

        ll  = -0.5 * np.sum((test - mu) ** 2 / var)
        ll -= 0.5 * len(test) * np.log(2 * np.pi * var)

        return float(ll)


    def _tune_predictive_ll(self, signal: np.ndarray, n_lambdas: int = 15, train_ratio: float = 0.7, n_folds: int = 3, horizon: int = 10) -> Dict:
        """
        Tune λ via time-series cross-validated predictive log-likelihood

        Important:
        ----------
        - Uses Gaussian plug-in predictive likelihood
        - Does NOT use full BOCPD posterior predictive
        - Intended for offline analysis, NOT default production
        """
        low, high  = self.config.hazard_range

        # Log-space grid for better coverage
        lambdas    = np.logspace(np.log10(low), np.log10(high), n_lambdas)

        scores     = list()

        for lam in lambdas:
            fold_lls = list()

            for fold in range(n_folds):
                train_end = int(len(signal) * (train_ratio + 0.05 * fold))

                if (train_end >= len(signal) - horizon):
                    continue

                train = signal[:train_end]
                test  = signal[train_end:train_end + horizon]

                if ((len(train) < 5) or (len(test) < 1)):
                    continue

                ll    = self._gaussian_predictive_ll(train = train, 
                                                     test  = test,
                                                    )

                fold_lls.append(ll)

            # Handle case where no valid folds
            if fold_lls:
                scores.append(float(np.mean(fold_lls)))

            else:
                scores.append(-np.inf)

        # Check if all scores are invalid
        if (all(s == -np.inf for s in scores)):
            # Fallback to heuristic
            heuristic_result           = self._tune_heuristic(signal, strategy='uniform')
            heuristic_result['notes']  = 'Predictive LL failed - fallback to heuristic'
            heuristic_result['method'] = 'predictive_ll_fallback'
            
            return heuristic_result

        # Exclude -inf scores from consideration
        valid_indices = [i for i, s in enumerate(scores) if s > -np.inf]
        
        if not valid_indices:
            # All scores invalid - fallback
            heuristic_result           = self._tune_heuristic(signal, strategy='uniform')
            heuristic_result['notes']  = 'All predictive LL scores invalid - fallback to heuristic'
            heuristic_result['method'] = 'predictive_ll_fallback'
            
            return heuristic_result

        valid_scores  = [scores[i] for i in valid_indices]
        valid_lambdas = [lambdas[i] for i in valid_indices]

        idx           = int(np.argmax(valid_scores))
        best_lambda   = float(valid_lambdas[idx])
        best_score    = valid_scores[idx]

        notes         = f'Gaussian plug-in predictive likelihood (approximate), best LL={best_score:.2f} at λ={best_lambda:.1f}'

        # ADDED: Warning for extreme values
        if (best_lambda < 20):
            notes += f" WARNING: Selected λ={best_lambda:.1f} is low (expecting CP every {best_lambda:.0f} days)"

        return {'optimal_hazard_lambda' : best_lambda,
                'optimal_hazard_rate'   : 1.0 / best_lambda,
                'lambdas_tested'        : lambdas.tolist(),
                'scores'                : scores,
                'best_score'            : best_score,
                'criterion'             : 'held_out_gaussian_predictive_ll',
                'method'                : 'predictive_ll',
                'n_folds'               : n_folds,
                'train_ratio'           : train_ratio,
                'horizon'               : horizon,
                'notes'                 : notes,
               }