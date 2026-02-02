# Dependencies
import numpy as np
import pandas as pd
from typing import Dict
from typing import Optional


class BOCPDHazardTuner:
    """
    BOCPD hazard (λ) tuner
    
    Supported methods:
    - 'heuristic'     : Fast, production-safe (DEFAULT)
    - 'predictive_ll' : Cross-validated (use for analysis)
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
            return self._tune_heuristic(signal=signal)
        
        return self._tune_predictive_ll(signal=signal)


    # HEURISTIC TUNER (DEFAULT, FAST) - FIXED
    def _tune_heuristic(self, signal: np.ndarray, target_n_cps: Optional[int] = None, strategy: str = 'variance') -> Dict:
        """
        Heuristic hazard tuning with default values
        
        Strategies:
        - 'variance' : Use variance changes (DEFAULT, RECOMMENDED for CV)
        - 'autocorr' : Use autocorrelation decay 
        - 'uniform'  : Assume uniform spacing
        """
        T      = len(signal)
        lo, hi = self.config.hazard_range
        
        # Enforce minimum hazard: never allow λ < 50 days
        lo     = max(lo, 50.0)  
        
        if (strategy == 'autocorr'):
            # Robust autocorrelation heuristic
            if (len(signal) > 5):
                # Compute lag-1 autocorrelation
                acf_1 = np.corrcoef(signal[:-1], signal[1:])[0, 1]
                acf_1 = np.clip(acf_1, -0.99, 0.99)
                
                # Expected run length from ACF decay
                # High ACF (0.9) → slow change → high λ
                # Low ACF (0.1) → fast change → low λ
                if (acf_1 > 0.01):
                    # Exponential decay model: λ = -1/log(ρ)
                    raw_lambda = -1.0 / np.log(max(acf_1, 0.01))
                    # Scale by signal length (longer signals → expect more CPs)
                    raw_lambda = min(raw_lambda, T / 3)  # At least 3 CPs expected

                else:
                    # Very low autocorrelation → moderate λ
                    raw_lambda = T / 5
                
                hazard_lambda = float(np.clip(raw_lambda, lo, hi))
                notes         = (f"Autocorr-based: ACF(1)={acf_1:.3f} → raw λ={raw_lambda:.1f}, clipped to [{lo}, {hi}] → {hazard_lambda:.1f}")
            
            else:
                # Fallback for short signals
                hazard_lambda = float(np.clip(T / 4, lo, hi))
                notes         = "Signal too short for autocorr - fallback to T/4"
        
        elif (strategy == 'uniform'):
            # Default target
            if target_n_cps is None:
                # For 365-day signal: expect 3-6 CPs → λ ≈ 60-120
                target_n_cps = max(3, min(6, T // 60))
            
            raw_lambda    = T / (target_n_cps + 1)
            hazard_lambda = float(np.clip(raw_lambda, lo, hi))
            notes         = (f"Uniform spacing: T/(k+1) = {T}/({target_n_cps}+1) = {raw_lambda:.1f}, clipped to [{lo}, {hi}] → {hazard_lambda:.1f}")
        
        elif (strategy == 'variance'):
            # Variance-based heuristic 
            if (len(signal) >= 10):
                rolling_var = (pd.Series(signal).rolling(window = 5, center = True).var().bfill().ffill())
                var_diff    = np.abs(np.diff(rolling_var))
                threshold   = np.percentile(var_diff, 75)
                rough_cps   = np.where(var_diff > threshold)[0]
                
                if (len(rough_cps) > 1):
                    segments      = np.diff([0] + list(rough_cps) + [T])
                    avg_segment   = float(np.median(segments))
                    raw_lambda    = avg_segment
                    hazard_lambda = float(np.clip(raw_lambda, lo, hi))
                    notes         = (f"Variance-based: {len(rough_cps)} rough CPs → median segment={avg_segment:.1f}, clipped to [{lo}, {hi}] → {hazard_lambda:.1f}")
                
                else:
                    hazard_lambda = float(np.clip(T / 5, lo, hi))
                    notes         = "Variance-based: no clear changes - fallback to T/5"
            
            else:
                hazard_lambda = float(np.clip(T / 4, lo, hi))
                notes         = "Signal too short for variance - fallback"
        
        else:
            raise ValueError(f"Unknown heuristic strategy: {strategy}")
        
        # Clinical plausibility check
        if (hazard_lambda < 50):
            notes += f"- WARNING: λ={hazard_lambda:.1f} < 50 (very aggressive)"
        
        if (hazard_lambda > 200):
            notes += f"- INFO: λ={hazard_lambda:.1f} > 200 (conservative, <2 CPs/year expected)"
        
        return {'optimal_hazard_lambda' : hazard_lambda,
                'optimal_hazard_rate'   : 1.0 / hazard_lambda,
                'lambdas_tested'        : [hazard_lambda],
                'scores'                : None,
                'criterion'             : f'heuristic_{strategy}',
                'method'                : 'heuristic',
                'strategy'              : strategy,
                'notes'                 : notes,
               }


    # PREDICTIVE LOG-LIKELIHOOD TUNER 
    @staticmethod
    def _gaussian_predictive_ll(train: np.ndarray, test: np.ndarray) -> float:
        """
        Gaussian plug-in predictive log-likelihood
        """
        if (len(train) < 2):
            return -np.inf
        
        mu   = np.mean(train)
        var  = np.var(train) + 1e-8
        
        ll   = -0.5 * np.sum((test - mu) ** 2 / var)
        ll  -= 0.5 * len(test) * np.log(2 * np.pi * var)
        
        return float(ll)


    def _tune_predictive_ll(self, signal: np.ndarray, n_lambdas: int = 20, train_ratio: float = 0.7, n_folds: int = 3, horizon: int = 10) -> Dict:
        """
        Tune λ via time-series cross-validated predictive LL
        
        Features:
        ---------
        - Lambda values tested (20 vs 15)
        - Handling of invalid scores
        - Clinical plausibility checks
        """
        low, high = self.config.hazard_range
        
        # Enforce minimum
        low       = max(low, 30.0)
        
        # Log-space grid for better coverage
        lambdas   = np.logspace(np.log10(low), np.log10(high), n_lambdas)
        scores    = list()
        
        for lam in lambdas:
            fold_lls = []
            
            for fold in range(n_folds):
                train_end = int(len(signal) * (train_ratio + 0.05 * fold))
                
                if (train_end >= (len(signal) - horizon)):
                    continue
                
                train = signal[:train_end]
                test  = signal[train_end:train_end + horizon]
                
                if ((len(train) < 5) or (len(test) < 1)):
                    continue
                
                ll = self._gaussian_predictive_ll(train=train, test=test)
                fold_lls.append(ll)
            
            if fold_lls:
                scores.append(float(np.mean(fold_lls)))

            else:
                scores.append(-np.inf)
        
        # Fallback handling
        valid_indices = [i for i, s in enumerate(scores) if (s > -np.inf)]
        
        if not valid_indices:
            # All scores invalid - fallback to heuristic
            heuristic_result           = self._tune_heuristic(signal, strategy='autocorr')
            heuristic_result['notes']  = 'Predictive LL failed - fallback to heuristic'
            heuristic_result['method'] = 'predictive_ll_fallback'
            
            return heuristic_result
        
        valid_scores  = [scores[i] for i in valid_indices]
        valid_lambdas = [lambdas[i] for i in valid_indices]
        
        idx           = int(np.argmax(valid_scores))
        best_lambda   = float(valid_lambdas[idx])
        best_score    = valid_scores[idx]
        
        notes         = f'Gaussian predictive LL: best={best_score:.2f} at λ={best_lambda:.1f}'
        
        # Clinical checks
        if (best_lambda < 30):
            notes += f"- WARNING: λ={best_lambda:.1f} < 30 (very aggressive)"
        
        if best_lambda > 200:
            notes += f"- INFO: λ={best_lambda:.1f} > 200 (conservative)"
        
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
