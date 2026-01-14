# Dependencies
import numpy as np
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
    def _tune_heuristic(self, signal: np.ndarray, target_n_cps: Optional[int] = None, strategy: str = 'uniform') -> Dict:
        """
        Heuristic hazard tuning

        REFINED: Added multiple heuristic strategies

        Strategies:
        -----------
        - 'uniform'  : Assume uniform spacing → λ ≈ T / (k + 1)
        - 'autocorr' : Use autocorrelation to estimate run length
        - 'variance' : Use variance changepoints as proxy
        """
        T             = len(signal)
        lo, hi        = self.config.hazard_range

        if (strategy == 'uniform'):
            # Original uniform spacing assumption
            if target_n_cps is None:
                target_n_cps = 3

            hazard_lambda = float(np.clip(T / (target_n_cps + 1), lo, hi))

            notes         = f"Uniform spacing: λ ≈ T/(k+1) = {T}/({target_n_cps}+1)"

        elif (strategy == 'autocorr'):
            # Autocorrelation-based heuristic: λ ≈ 1 / (1 - ACF(lag=1))
            if (len(signal) > 5):
                acf_1         = np.corrcoef(signal[:-1], signal[1:])[0, 1]
                acf_1         = np.clip(acf_1, -0.99, 0.99)

                hazard_lambda = 1.0 / max(1.0 - acf_1, 0.01)
                hazard_lambda = float(np.clip(hazard_lambda, lo, hi))

                notes         = f"Autocorrelation-based: ACF(1)={acf_1:.3f} → λ={hazard_lambda:.1f}"
            
            else:
                # Fallback to uniform
                hazard_lambda = float(np.clip(T / 4, lo, hi))
                notes         = "Autocorr failed (signal too short) - fallback to uniform"

        elif (strategy == 'variance'):
            # Variance-based heuristic: estimate segment lengths from variance changes
            rolling_var = pd.Series(signal).rolling(window=5, center=True).var().fillna(method='bfill').fillna(method='ffill')
            var_diff    = np.abs(np.diff(rolling_var))
            
            # Find rough change points
            threshold   = np.percentile(var_diff, 75)
            rough_cps   = np.where(var_diff > threshold)[0]

            if (len(rough_cps) > 0):
                avg_segment   = np.mean(np.diff([0] + list(rough_cps) + [T]))
                hazard_lambda = float(np.clip(avg_segment, lo, hi))
                notes         = f"Variance-based: {len(rough_cps)} rough CPs → avg segment={avg_segment:.1f}"
            
            else:
                hazard_lambda = float(np.clip(T / 4, lo, hi))
                notes         = "Variance-based failed (no variance changes) - fallback"

        else:
            raise ValueError(f"Unknown heuristic strategy: {strategy}")

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


    def _tune_predictive_ll(self, signal: np.ndarray, n_lambdas: int = 20, train_ratio: float = 0.7, n_folds: int = 3, horizon: int = 10) -> Dict:
        """
        Tune λ via time-series cross-validated predictive log-likelihood

        Important:
        ----------
        - Uses Gaussian plug-in predictive likelihood
        - Does NOT use full BOCPD posterior predictive
        - Intended for offline analysis, NOT default production
        """
        low, high  = self.config.hazard_range

        # Use fewer lambdas for faster tuning
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

        idx         = int(np.argmax(scores))
        best_lambda = float(lambdas[idx])
        best_score  = scores[idx]

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
                'notes'                 : f'Gaussian plug-in predictive likelihood (approximate), best LL={best_score:.2f}',
               }