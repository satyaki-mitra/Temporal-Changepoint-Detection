# Dependencies
import numpy as np
from scipy import stats
from typing import List
from typing import Dict
from typing import Tuple
from typing import Literal
from typing import Optional
from scipy.stats import false_discovery_control


# Base Utilities
def calculate_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Cohen's d = (mean_after - mean_before) / pooled_std

    Interpretation:
    ----------------
    |d| < 0.2   : Negligible
    0.2–0.5     : Small
    0.5–0.8     : Medium
    ≥ 0.8       : Large
    """
    if ((len(group1) < 2) or (len(group2) < 2)):
        return 0.0

    mean1, mean2 = np.mean(group1), np.mean(group2)
    var1, var2   = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_std   = np.sqrt(((len(group1) - 1) * var1 + (len(group2) - 1) * var2) / max(len(group1) + len(group2) - 2, 1))

    if (pooled_std < 1e-10):
        return 0.0

    return float((mean2 - mean1) / pooled_std)



# PELT — Frequentist Validation
class PELTStatisticalValidator:
    """
    Frequentist validation of offline change points (PELT)

    Each detected change point is validated using:
    - Before vs After windowed hypothesis tests
    - Multiple testing correction
    - Effect size (Cohen's d)
    """
    def __init__(self, alpha: float = 0.05, correction_method: Literal['bonferroni', 'fdr_bh', 'none'] = 'fdr_bh', effect_size_threshold: float = 0.3, window_size: Optional[int] = None):
        """
        Arguments:
        ----------
            alpha                  { float }   : Significance level

            correction_method      { Literal } : Multiple testing correction
            
            effect_size_threshold  { float }   : Minimum Cohen's d
            
            window_size            { int }     : If provided, limits before/after windows
        """
        self.alpha                 = alpha
        self.correction_method     = correction_method
        self.effect_size_threshold = effect_size_threshold
        self.window_size           = window_size


    def validate_all_changepoints(self, signal: np.ndarray, change_points: List[int]) -> Dict:
        """
        Validate all detected change points using frequentist tests

        Strategy:
        ----------
        - Compare BEFORE vs AFTER windows around each change point
        - Apply multiple testing correction
        - Report statistical + practical significance
        """
        valid_cps = [cp for cp in change_points if (0 < cp < len(signal))]

        if not valid_cps:
            return {'n_changepoints'      : 0,
                    'n_significant'       : 0,
                    'tests'               : {},
                    'overall_significant' : False,
                   }

        test_results = dict()
        p_values     = list()

        for idx, cp in enumerate(valid_cps):
            before = signal[:cp]
            after  = signal[cp:]

            if self.window_size:
                before = before[-self.window_size:]
                after  = after[:self.window_size]

            if ((len(before) < 5) or (len(after) < 5)):
                continue

            result                              = self._test_single_cp(before        = before, 
                                                                       after         = after, 
                                                                       cp_idx        = cp,
                                                                       signal_length = len(signal),
                                                                      )
            test_results[f'CP_{idx+1}_at_{cp}'] = result

            p_values.append(result['p_value'])

        if p_values:
            corrected = self._apply_correction(p_values)
            
            for key, p_corr in zip(test_results.keys(), corrected):
                test_results[key]['p_value_corrected']            = float(p_corr)
                test_results[key]['significant_after_correction'] = (p_corr < self.alpha)

        n_sig = sum(r.get('significant_after_correction', False) for r in test_results.values())

        return {'n_changepoints'      : len(valid_cps),
                'n_significant'       : n_sig,
                'tests'               : test_results,
                'overall_significant' : n_sig > 0,
                'alpha'               : self.alpha,
                'correction_method'   : self.correction_method,
                'summary'             : {'fraction_significant' : n_sig / max(len(valid_cps), 1),
                                         'mean_effect_size'     : float(np.mean([t['cohens_d'] for t in test_results.values()])) if test_results else 0.0,
                                         'max_effect_size'      : float(np.max([abs(t['cohens_d']) for t in test_results.values()])) if test_results else 0.0,
                                        }
               }


    def _test_single_cp(self, before: np.ndarray, after: np.ndarray, cp_idx: int, signal_length: int) -> Dict:
        """
        Perform hypothesis test for a single change point
        """
        test_name, stat, p_value = self._select_test(group1 = before, 
                                                     group2 = after,
                                                    )

        cohens_d                 = calculate_effect_size(before, after)

        mean_diff                = np.mean(after) - np.mean(before)

        return {'change_point_index'  : int(cp_idx),
                'test_name'           : test_name,
                'test_statistic'      : float(stat),
                'p_value'             : float(p_value),
                'cohens_d'            : float(cohens_d),
                'before_mean'         : float(np.mean(before)),
                'after_mean'          : float(np.mean(after)),
                'mean_difference'     : float(mean_diff),
                'significant'         : p_value < self.alpha,
                'meaningful_effect'   : abs(cohens_d) >= self.effect_size_threshold,
                'cp_index'            : cp_idx,
                'window_size'         : self.window_size,
                'normalized_position' : (cp_idx / signal_length),
               }

    
    def _select_test(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[str, float, float]:
        """
        Select appropriate statistical test
        """
        if ((len(np.unique(group1)) == 1) or (len(np.unique(group2)) == 1)):
            stat, p = stats.ttest_ind(group1, group2)
            
            return 'T-Test', stat, p

        if ((len(group1) < 20) or (len(group2) < 20)):
            stat, p = stats.ttest_ind(group1, group2)
            
            return 'T-Test', stat, p

        stat, p = stats.ranksums(group1, group2)

        return 'Wilcoxon Rank-Sum', stat, p


    def _apply_correction(self, p_values: List[float]) -> np.ndarray:
        """
        Apply multiple testing correction
        """
        if (self.correction_method == 'bonferroni'):
            return np.minimum(np.array(p_values) * len(p_values), 1.0)

        if (self.correction_method == 'fdr_bh'):
            try:
                return false_discovery_control(p_values, method = 'bh')

            except Exception:
                return self._fdr_bh_manual(p_values)

        return np.array(p_values)

    
    def _fdr_bh_manual(self, p_values: List[float]) -> np.ndarray:
        """
        Manual Benjamini-Hochberg correction
        """
        p         = np.array(p_values)
        idx       = np.argsort(p)
        sorted_p  = p[idx]

        corrected = sorted_p * len(p) / (np.arange(len(p)) + 1)
        corrected = np.minimum.accumulate(corrected[::-1])[::-1]

        out       = np.empty_like(corrected)
        out[idx]  = corrected

        return out



# BOCPD — Bayesian Validation
class BOCPDStatisticalValidator:
    """
    Bayesian validation for BOCPD detections

    Validation is based on:
    - Posterior probability mass
    - Persistence of CP posterior peaks
    """
    def __init__(self, posterior_threshold: float = 0.6, persistence: int = 3):
        """
        Arguments:
        ----------
        posterior_threshold { float } : Minimum posterior probability

        persistence          { int }  : Required consecutive timesteps above threshold
        """
        self.posterior_threshold = posterior_threshold
        self.persistence         = persistence


    def validate(self, cp_posterior: np.ndarray) -> Dict:
        """
        Validate BOCPD change points from posterior probabilities
        """
        cp_flags   = cp_posterior >= self.posterior_threshold

        persistent = np.convolve(cp_flags.astype(int), np.ones(self.persistence, dtype = int), mode = 'same') >= self.persistence

        detected   = np.where(persistent)[0]

        return {'n_changepoints'      : len(detected),
                'indices'             : detected.tolist(),
                'posterior_threshold' : self.posterior_threshold,
                'persistence'         : self.persistence,
                'overall_significant' : (len(detected) > 0),
                'summary'             : {'mean_posterior_at_cp' : float(np.mean(cp_posterior[detected])) if len(detected) else 0.0,
                                         'coverage_ratio'       : len(detected) / max(len(cp_posterior), 1),
                                        }
               }



# Unified Convenience Function
def validate_change_points(method: Literal['pelt', 'bocpd'], signal: Optional[np.ndarray] = None, change_points: Optional[List[int]] = None, 
                           cp_posterior: Optional[np.ndarray] = None, **kwargs) -> Dict:
    """
    Unified validation entry point
    """
    if (method == 'pelt'):
        validator = PELTStatisticalValidator(**kwargs)
        
        return validator.validate_all_changepoints(signal, change_points)

    if (method == 'bocpd'):
        validator = BOCPDStatisticalValidator(**kwargs)

        return validator.validate(cp_posterior)

    raise ValueError(f"Unknown validation method: {method}")
