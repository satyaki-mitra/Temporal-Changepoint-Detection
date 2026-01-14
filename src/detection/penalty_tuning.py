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
    """
    def __init__(self, cost_model: str = 'l1', min_size: int = 5, jump: int = 1):
        """
        Initialize penalty tuner

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
        """
        if (signal.ndim != 1):
            raise ValueError("signal must be a 1D array")

        if (len(signal) < self.min_size * 2):
            raise ValueError(f"Signal too short for reliable penalty tuning (length={len(signal)}, min_size={self.min_size})")

        penalties        = np.logspace(start = np.log10(penalty_range[0]),
                                       stop  = np.log10(penalty_range[1]),
                                       num   = n_penalties,
                                      )

        bic_scores       = list()
        n_cps_list       = list()
        valid_penalties  = list()
        failed_penalties = dict()

        for penalty in penalties:
            try:
                algo          = rpt.Pelt(model    = self.cost_model,
                                         min_size = self.min_size,
                                         jump     = self.jump,
                                        )

                algo.fit(signal)
                change_points = algo.predict(pen = penalty)

                n_cps         = max(len(change_points) - 1, 0)

                bic           = self._calculate_bic(signal        = signal,
                                                    change_points = change_points,
                                                   )

                valid_penalties.append(float(penalty))
                n_cps_list.append(int(n_cps))
                bic_scores.append(float(bic))

            except Exception as exc:
                failed_penalties[float(penalty)] = str(exc)
                continue

        if not valid_penalties:
            raise RuntimeError("Penalty tuning failed: all penalty values caused PELT failure")

        optimal_idx     = int(np.argmin(bic_scores))
        optimal_penalty = float(valid_penalties[optimal_idx])

        return {'optimal_penalty'        : optimal_penalty,
                'penalties_tested'       : valid_penalties,
                'n_changepoints'         : n_cps_list,
                'bic_scores'             : bic_scores,
                'optimal_n_changepoints' : n_cps_list[optimal_idx],
                'failed_penalties'       : failed_penalties,
                'cost_model'             : self.cost_model,
                'min_segment_size'       : self.min_size,
                'jump'                   : self.jump,
                'criterion'              : 'bic',
                'signal_length'          : int(len(signal)),
               }


    def _calculate_bic(self, signal: np.ndarray, change_points: List[int]) -> float:
        """
        Calculate Bayesian Information Criterion

        BIC = n * log(σ²) + p * log(n)
        - where p = 2 * n_segments (each segment has mean + variance)
        """
        n              = len(signal)
    
        # Ensure boundaries start at 0 and end at n exactly once
        boundaries     = [0] + [cp for cp in change_points if ((cp != 0) and (cp != n))]
        boundaries.append(n)

        # p = 2 * n_segments (not just K)
        n_segments     = len(boundaries) - 1

        # Each segment: mean + variance
        p              = 2 * n_segments 

        total_residual = 0.0

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            segment = signal[start:end]
            
            if (len(segment) < 2):
                continue

            mean            = np.mean(segment)
            total_residual += np.sum((segment - mean) ** 2)
        
        # Calculate BIC Score
        sigma_squared = max(total_residual / max(n, 1), 1e-10)
        bic_score     = float(n * np.log(sigma_squared) + p * np.log(n))

        return bic_score


    # Visualization
    def plot_tuning_curve(self, penalties: List[float], bic_scores: List[float], n_changepoints: List[int], optimal_penalty: float, save_path: Optional[str] = None):
        """
        Plot BIC and number of change points vs penalty
        """
        fig, (ax1, ax2) = plt.subplots(nrows   = 1,
                                       ncols   = 2,
                                       figsize = (16, 6),
                                      )

        ax1.plot(penalties, 
                 bic_scores, 
                 marker    = 'o', 
                 linewidth = 2,
                )

        ax1.axvline(optimal_penalty, 
                    color     = 'red', 
                    linestyle = '--', 
                    linewidth = 2,
                    label     = f'Optimal = {optimal_penalty:.2f}',
                   )

        ax1.set_xscale('log')
        ax1.set_xlabel('Penalty')
        ax1.set_ylabel('BIC')
        ax1.set_title('BIC vs Penalty (FIXED: p = 2 × n_segments)')
        ax1.grid(True, alpha = 0.3)
        ax1.legend()

        ax2.plot(penalties, 
                 n_changepoints, 
                 marker    = 'o', 
                 linewidth = 2,
                )

        ax2.axvline(optimal_penalty, 
                    color     = 'red', 
                    linestyle = '--', 
                    linewidth = 2,
                   )

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