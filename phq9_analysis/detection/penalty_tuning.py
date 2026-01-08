# Dependencies
import numpy as np
import ruptures as rpt
from typing import Dict
from typing import List
from typing import Tuple
from typing import Optional
import matplotlib.pyplot as plt


class PenaltyTuner:
    """
    Tune PELT penalty parameter using Bayesian Information Criterion (BIC)

    References:
    ----------
    - Killick et al. (2012): Optimal detection of changepoints
    - Truong et al. (2020): Selective review of offline change point methods

    This class is intentionally algorithm-specific (PELT-only)
    """
    def __init__(self, cost_model: str = 'l1', min_size: int = 5, jump: int = 1):
        """
        Initialize penalty tuner.

        Arguments:
        ----------
            cost_model { str } : Cost function ('l1', 'l2', 'rbf', 'ar')

            min_size   { int } : Minimum segment size

            jump       { int } : Jump parameter for PELT
        """
        self.cost_model = cost_model
        self.min_size   = min_size
        self.jump       = jump


    def tune(self, signal: np.ndarray, penalty_range: Tuple[float, float] = (0.1, 10.0), n_penalties: int = 50) -> Dict:
        """
        Tune PELT penalty parameter using BIC

        Arguments:
        ----------
            signal         { np.ndarray } : 1D signal (e.g., daily CV values)

            penalty_range     { tuple }   : (min_penalty, max_penalty)

            n_penalties        { int }    : Number of penalty values to test

        Returns:
        --------
                     { dict }             : Tuning results dictionary containing:
                                            - optimal_penalty
                                            - penalties_tested
                                            - n_changepoints
                                            - bic_scores
                                            - optimal_n_changepoints
        """
        if (signal.ndim != 1):
            raise ValueError("signal must be a 1D array")

        if (len(signal) < self.min_size * 2):
            raise ValueError(f"Signal too short for reliable penalty tuning (length={len(signal)}, min_size={self.min_size})")

        penalties  = np.logspace(start = np.log10(penalty_range[0]),
                                 stop  = np.log10(penalty_range[1]),
                                 num   = n_penalties,
                                )

        bic_scores = list()
        n_cps_list = list()

        for penalty in penalties:
            algo = rpt.Pelt(model    = self.cost_model,
                            min_size = self.min_size,
                            jump     = self.jump,
                           )

            algo.fit(signal)
            change_points = algo.predict(pen = penalty)

            n_cps         = len(change_points) - 1
            n_cps_list.append(n_cps)

            bic           = self._calculate_bic(signal        = signal, 
                                                change_points = change_points,
                                               )
            bic_scores.append(bic)

        optimal_idx       = int(np.argmin(bic_scores))
        optimal_penalty   = float(penalties[optimal_idx])

        return {'optimal_penalty'        : optimal_penalty,
                'penalties_tested'       : penalties.tolist(),
                'n_changepoints'         : n_cps_list,
                'bic_scores'             : bic_scores,
                'optimal_n_changepoints' : n_cps_list[optimal_idx],
                'cost_model'             : self.cost_model,
                'min_segment_size'       : self.min_size,
                'jump'                   : self.jump,
                'criterion'              : 'bic',
                'signal_length'          : int(len(signal)),
               }

    
    def _calculate_bic(self, signal: np.ndarray, change_points: List[int]) -> float:
        """
        Calculate Bayesian Information Criterion

        Formula:
        --------
        BIC = n * log(σ²) + K * log(n)

        Where:
        - n = number of observations
        - σ² = residual variance across segments
        - K = number of change points

        Arguments:
        ----------
            signal        { np.ndarray } : Original signal

            change_points    { list }    : Change point indices (including end)

        Returns:
        --------
                    { float }            : BIC score (lower is better)
        """
        n              = len(signal)
        K              = max(len(change_points) - 1, 0)

        total_residual = 0.0
        boundaries     = [0] + change_points

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            segment    = signal[start:end]

            if (len(segment) < 2):
                continue

            mean            = np.mean(segment)
            residuals       = segment - mean
            total_residual += np.sum(residuals ** 2)

        sigma_squared = total_residual / max(n, 1)
        sigma_squared = max(sigma_squared, 1e-10)

        return float(n * np.log(sigma_squared) + K * np.log(n))


    # Visualization
    def plot_tuning_curve(self, penalties: List[float], bic_scores: List[float], n_changepoints: List[int], optimal_penalty: float, save_path: Optional[str] = None):
        """
        Plot BIC and number of change points vs penalty

        Arguments:
        ----------
            penalties       { list }  : Penalty values tested

            bic_scores      { list }  : Corresponding BIC scores

            n_changepoints  { list }  : Number of change points per penalty

            optimal_penalty { float } : Selected optimal penalty

            save_path       { str }   : Optional save path
        """
        fig, (ax1, ax2) = plt.subplots(nrows   = 1, 
                                       ncols   = 2, 
                                       figsize = (16, 6),
                                      )

        # BIC curve
        ax1.plot(penalties, bic_scores, marker = 'o', linewidth = 2)
        ax1.axvline(optimal_penalty, color = 'red', linestyle = '--', linewidth = 2)
        ax1.set_xscale('log')
        ax1.set_xlabel('Penalty')
        ax1.set_ylabel('BIC')
        ax1.set_title('BIC vs Penalty')
        ax1.grid(True, alpha = 0.3)

        # Change points count
        ax2.plot(penalties, n_changepoints, marker = 'o', linewidth = 2)
        ax2.axvline(optimal_penalty, color = 'red', linestyle = '--', linewidth = 2)
        ax2.set_xscale('log')
        ax2.set_xlabel('Penalty')
        ax2.set_ylabel('Number of Change Points')
        ax2.set_title('Change Points vs Penalty')
        ax2.grid(True, alpha = 0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(fname       = save_path, 
                        dpi         = 300, 
                        bbox_inches = 'tight',
                       )

            plt.close()

        else:
            plt.show()


# Convenience function
def tune_penalty_bic(signal: np.ndarray, cost_model: str = 'l1', min_size: int = 5, jump: int = 1, penalty_range: Tuple[float, float] = (0.1, 10.0),
                     plot: bool = False, save_path: Optional[str] = None) -> Tuple[float, Dict]:
    """
    Convenience wrapper for BIC-based PELT penalty tuning

    Arguments:
    ----------
        signal        { np.ndarray } : 1D signal array

        cost_model        { str }    : PELT cost function

        min_size          { int }    : Minimum segment size

        jump              { int }    : PELT jump parameter

        penalty_range    { tuple }   : Penalty search range

        plot             { bool }    : Whether to plot tuning curve

        save_path         { str }    : Optional plot save path

    Returns:
    --------
                { tuple }            : (optimal_penalty, tuning_results_dict)
    """
    tuner   = PenaltyTuner(cost_model = cost_model,
                           min_size   = min_size,
                           jump       = jump,
                          )

    results = tuner.tune(signal        = signal,
                         penalty_range = penalty_range,
                        )

    if plot:
        tuner.plot_tuning_curve(penalties       = results['penalties_tested'],
                                bic_scores      = results['bic_scores'],
                                n_changepoints  = results['n_changepoints'],
                                optimal_penalty = results['optimal_penalty'],
                                save_path       = save_path,
                               )

    return results['optimal_penalty'], results