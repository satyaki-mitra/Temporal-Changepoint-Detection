# Dependencies
import numpy as np
import pandas as pd
from typing import List
from typing import Dict
from scipy import stats
from typing import Tuple
from typing import Literal
from scipy.stats import false_discovery_control


class StatisticalValidator:
    """
    Validate change points using statistical hypothesis testing
    """
    def __init__(self, alpha: float = 0.05, correction_method: Literal['bonferroni', 'fdr_bh', 'none'] = 'fdr_bh', effect_size_threshold: float = 0.3):
        """
        Initialize validator
        
        Arguments:
        ----------
            alpha                  { float }  : Significance level

            correction_method     { Literal } : Multiple testing correction
            
            effect_size_threshold  { float }  : Minimum Cohen's d for meaningful change
        """
        self.alpha                 = alpha
        self.correction_method     = correction_method
        self.effect_size_threshold = effect_size_threshold

    
    def validate_all_changepoints(self, signal: np.ndarray, change_points: List[int]) -> Dict:
        """
        Validate all detected change points
        
        - Tests each change point by comparing data before vs after, not adjacent segments
        
        Arguments:
        ----------
            signal        { np.ndarray } : 1D array (e.g., daily CV values)

            change_points    { list }    : List of change point indices (includes end)
        
        Returns:
        --------
                    { dict }             : Dictionary with test results for each change point
        """
        # Remove end point if present
        actual_cps = [cp for cp in change_points if (cp < len(signal))]
        
        if (len(actual_cps) == 0):
            return {'n_changepoints'      : 0,
                    'tests'               : {},
                    'overall_significant' : False,
                   }
        
        # Test each change point
        test_results = dict()
        p_values     = list()
        
        for i, cp_idx in enumerate(actual_cps):
            # Compare before vs after the change point
            before = signal[:cp_idx]
            after  = signal[cp_idx:]
            
            if (len(before) < 2) or (len(after) < 2):
                # Need at least 2 points in each group
                continue  
            
            # Perform test
            test_result                           = self._test_single_changepoint(before = before, 
                                                                                  after  = after, 
                                                                                  cp_idx = cp_idx,
                                                                                 )
            
            test_results[f'CP_{i+1}_at_{cp_idx}'] = test_result

            p_values.append(test_result['p_value'])
        
        # Apply multiple testing correction
        if (len(p_values) > 0):
            corrected_p_values = self._apply_correction(p_values = p_values)
            
            # Update results with corrected p-values
            for i, (key, p_corrected) in enumerate(zip(test_results.keys(), corrected_p_values)):
                test_results[key]['p_value_corrected']            = p_corrected
                test_results[key]['significant_after_correction'] = p_corrected < self.alpha
        
        # Overall assessment
        n_significant = sum(1 for result in test_results.values() if result.get('significant_after_correction', False))
        
        return {'n_changepoints'      : len(actual_cps),
                'n_significant'       : n_significant,
                'tests'               : test_results,
                'overall_significant' : n_significant > 0,
                'correction_method'   : self.correction_method,
                'alpha'               : self.alpha,
               }

    
    def _test_single_changepoint(self, before: np.ndarray, after: np.ndarray, cp_idx: int) -> Dict:
        """
        Test single change point: before vs after
        
        Arguments:
        ----------
            before { np.ndarray } : Data before change point

            after  { np.ndarray } : Data after change point
            
            cp_idx     { int }    : Change point index
        
        Returns:
        --------
                  { dict }        : Test results dictionary
        """
        # Choose test based on data properties
        test_name, test_stat, p_value = self._select_and_run_test(group1 = before, 
                                                                  group2 = after,
                                                                 )
        
        # Calculate effect size (Cohen's d)
        cohens_d                      = calculate_effect_size(before, after)
        
        # Descriptive statistics
        before_mean                   = float(np.mean(before))
        after_mean                    = float(np.mean(after))
        mean_diff                     = after_mean - before_mean
        
        # Interpretation
        is_significant                = (p_value < self.alpha)
        has_meaningful_effect         = (abs(cohens_d) >= self.effect_size_threshold)
        
        if is_significant and has_meaningful_effect:
            interpretation = (f"Significant change detected (p={p_value:.4f}, d={cohens_d:.3f}). Mean {'increased' if mean_diff > 0 else 'decreased'} by {abs(mean_diff):.3f}.")

        elif is_significant:
            interpretation = (f"Statistically significant (p={p_value:.4f}) but small effect size (d={cohens_d:.3f}). May not be clinically meaningful.")

        else:
            interpretation = (f"No significant change detected (p={p_value:.4f}).")
        
        return {'change_point_index' : int(cp_idx),
                'test_name'          : test_name,
                'test_statistic'     : float(test_stat),
                'p_value'            : float(p_value),
                'cohens_d'           : float(cohens_d),
                'before_mean'        : before_mean,
                'after_mean'         : after_mean,
                'mean_difference'    : mean_diff,
                'before_n'           : len(before),
                'after_n'            : len(after),
                'significant'        : is_significant,
                'meaningful_effect'  : has_meaningful_effect,
                'interpretation'     : interpretation,
               }

    
    def _select_and_run_test(self, group1: np.ndarray, group2: np.ndarray) -> Tuple[str, float, float]:
        """
        Select and run appropriate statistical test
        
        Arguments:
        ----------
            group1 { np.ndarray } : First group of data

            group2 { np.ndarray } : Second group of data
        
        Returns:
        --------
                { tuple }         : A python tuple containing:
                                    - test_name
                                    - test_statistic
                                    - p_value
        """
        # Check for constant values
        if ((len(np.unique(group1)) == 1) or (len(np.unique(group2)) == 1)):
            # Use t-test for constant groups
            test_stat, p_value = stats.ttest_ind(group1, group2)

            return 'T-Test', test_stat, p_value
        
        # Check sample sizes
        if ((len(group1) < 5) or (len(group2) < 5)):
            # Use t-test for small samples
            test_stat, p_value = stats.ttest_ind(group1, group2)

            return 'T-Test', test_stat, p_value
        
        # Default: Wilcoxon rank-sum (Mann-Whitney U)
        test_stat, p_value = stats.ranksums(group1, group2)
        
        return 'Wilcoxon Rank-Sum', test_stat, p_value
    

    def _apply_correction(self, p_values: List[float]) -> np.ndarray:
        """
        Apply multiple testing correction
        
        Arguments:
        ----------
            p_values { list } : List of p-values
        
        Returns:
        --------
            { np.ndarray }    : Array of corrected p-values
        """
        if (self.correction_method == 'bonferroni'):
            # Bonferroni correction
            return np.array(p_values) * len(p_values)
        
        elif (self.correction_method == 'fdr_bh'):
            # Benjamini-Hochberg FDR correction
            try:
                return false_discovery_control(p_values, method = 'bh')

            except:
                # Fallback if scipy version doesn't have this
                return self._fdr_bh_manual(p_values)
        
        else:  
            # 'none'
            return np.array(p_values)

    
    def _fdr_bh_manual(self, p_values: List[float]) -> np.ndarray:
        """
        Manual Benjamini-Hochberg FDR correction
        
        Arguments:
        ----------
            p_values { list } : List of p-values
        
        Returns:
        --------
            { np.ndarray }    : Corrected p-values
        """
        p_array                  = np.array(p_values)
        n                        = len(p_array)
        
        # Sort p-values
        sorted_indices           = np.argsort(p_array)
        sorted_p                 = p_array[sorted_indices]
        
        # BH correction
        corrected                = sorted_p * n / (np.arange(n) + 1)
        
        # Ensure monotonicity
        for i in range(n - 2, -1, -1):
            corrected[i] = min(corrected[i], corrected[i + 1])
        
        # Restore original order
        restored                 = np.empty_like(corrected)
        restored[sorted_indices] = corrected
        
        return restored



def calculate_effect_size(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size
     
    - Cohen's d = (mean1 - mean2) / pooled_std
    
    - Interpretation:
    - |d| < 0.2: Negligible
    - 0.2 ≤ |d| < 0.5: Small
    - 0.5 ≤ |d| < 0.8: Medium
    - |d| ≥ 0.8: Large
    
    Arguments:
    ----------
        group1 { np.ndarray } : First group

        group2 { np.ndarray } : Second group
    
    Returns:
    --------
            { float }         : Cohen's d
    """
    mean1      = np.mean(group1)
    mean2      = np.mean(group2)
    
    var1       = np.var(group1, ddof=1)
    var2       = np.var(group2, ddof=1)
    
    n1         = len(group1)
    n2         = len(group2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if (pooled_std == 0):
        return 0.0
    
    cohens_d   = (mean2 - mean1) / pooled_std
    
    return cohens_d


def test_change_point_significance(signal: np.ndarray, change_points: List[int], alpha: float = 0.05, correction: str = 'fdr_bh') -> Dict:
    """
    Convenience function to test change point significance
    
    Arguments:
    ----------
        signal        { np.ndarray } : 1D signal array

        change_points   { list }     : List of change point indices
        
        alpha           { float }    : Significance level
        
        correction       { str }     : Multiple testing correction method
    
    Returns:
    --------
                { dict }             : Validation results dictionary
    """
    validator = StatisticalValidator(alpha             = alpha,
                                     correction_method = correction,
                                    )
    
    return validator.validate_all_changepoints(signal, change_points)