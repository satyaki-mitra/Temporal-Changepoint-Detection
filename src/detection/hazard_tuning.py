# Dependencies
import numpy as np
from typing import Dict
from src.detection.bocpd_detector import BOCPDDetector


class BOCPDHazardTuner:
    """
    BOCPD hazard (λ) tuner using various methods for tuning λ

    Supported methods:
    ------------------
    - 'heuristic'     : Deterministic, fast, production-safe
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
            raise ValueError("Signal too short for hazard tuning")

        if (self.method == 'heuristic'):
            return self._tune_heuristic(signal = signal)

        return self._tune_predictive_ll(signal = signal)


    # HEURISTIC TUNER (DEFAULT, FAST)
    def _tune_heuristic(self, signal: np.ndarray, target_n_cps: int = 3) -> Dict:
        """
        Heuristic hazard tuning

        Assumption:
        -----------
        If k change points are expected over T timesteps, average segment length λ ≈ T / (k + 1)
        """
        T             = len(signal)
        lo, hi        = self.config.hazard_range

        hazard_lambda = float(np.clip(T / (target_n_cps + 1), lo, hi))

        return {'optimal_hazard_lambda' : hazard_lambda,
                'optimal_hazard_rate'   : 1.0 / hazard_lambda,
                'lambdas_tested'        : [hazard_lambda],
                'scores'                : None,
                'criterion'             : 'signal_length_heuristic',
                'method'                : 'heuristic',
                'notes'                 : f"λ ≈ T/(k+1) = {T}/({target_n_cps}+1)",
               }


    # PREDICTIVE LOG-LIKELIHOOD TUNER (OFFLINE / ANALYSIS)
    @staticmethod
    def _gaussian_predictive_ll(train: np.ndarray, test: np.ndarray) -> float:
        """
        Gaussian plug-in predictive log-likelihood
        """
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
        lo, hi  = self.config.hazard_range
        lambdas = np.logspace(np.log10(lo), np.log10(hi), n_lambdas)

        scores  = list()

        for lam in lambdas:
            fold_lls = list()

            for fold in range(n_folds):
                train_end = int(len(signal) * (train_ratio + 0.05 * fold))

                if (train_end >= len(signal) - horizon):
                    continue

                train = signal[:train_end]
                test  = signal[train_end:train_end + horizon]

                ll    = self._gaussian_predictive_ll(train = train, 
                                                     test  = test,
                                                    )

                fold_lls.append(ll)

            scores.append(float(np.mean(fold_lls)) if fold_lls else -np.inf)

        idx         = int(np.argmax(scores))
        best_lambda = float(lambdas[idx])

        return {'optimal_hazard_lambda' : best_lambda,
                'optimal_hazard_rate'   : 1.0 / best_lambda,
                'lambdas_tested'        : lambdas.tolist(),
                'scores'                : scores,
                'criterion'             : 'held_out_gaussian_predictive_ll',
                'method'                : 'predictive_ll',
                'notes'                 : 'Gaussian plug-in predictive likelihood (approximate)',
               }