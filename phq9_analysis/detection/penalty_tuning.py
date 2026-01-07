# Dependencies
import numpy as np
import pandas as pd
import ruptures as rpt
from typing import List
from typing import Tuple
from typing import Optional
import matplotlib.pyplot as plt


class PenaltyTuner:
    """
    Tune PELT penalty parameter using BIC criterion
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

    
    def tune(self, signal: np.ndarray, penalty_range: Tuple[float, float] = (0.1, 10.0), n_penalties: int = 50) -> Tuple[float, List[float], List[int], List[float]]:
        """
        Find optimal penalty using BIC
        
        Arguments:
        ----------
            signal        { np.ndarray } : 1D array of values (e.g., daily CV)

            penalty_range    { tuple }   : (min_penalty, max_penalty) to test
            
            n_penalties       { int }    : Number of penalty values to test
        
        Returns:
        --------
                    { tuple }            : A python tuple containing:
                                           - optimal_penalty
                                           - penalties_tested
                                           - n_changepoints
                                           - bic_scores
        """
        # Generate penalty values (log scale)
        penalties           = np.logspace(np.log10(penalty_range[0]),
                                          np.log10(penalty_range[1]),
                                          n_penalties
                                         )
                            
        n_obs               = len(signal)
        bic_scores          = list()
        n_changepoints_list = list()
        
        for penalty in penalties:
            # Fit PELT
            algo          = rpt.Pelt(model    = self.cost_model,
                                     min_size = self.min_size,
                                     jump     = self.jump,
                                    )

            algo.fit(signal)
            change_points = algo.predict(pen = penalty)
            
            # Number of change points (excluding end point)
            K             = len(change_points) - 1

            n_changepoints_list.append(K)
            
            # Calculate BIC
            bic           = self._calculate_bic(signal        = signal, 
                                                change_points = change_points,
                                               )

            bic_scores.append(bic)
        
        # Find optimal penalty (minimum BIC)
        optimal_idx       = np.argmin(bic_scores)
        optimal_penalty   = penalties[optimal_idx]
        
        return (optimal_penalty, penalties.tolist(), n_changepoints_list, bic_scores)

    
    def _calculate_bic(self, signal: np.ndarray, change_points: List[int]) -> float:
        """
        Calculate Bayesian Information Criterion: BIC = n*log(σ²) + K*log(n)
        
        Arguments:
        ----------
            signal        { np.ndarray } : Original signal

            change_points    { list }    : List of change point indices
        
        Returns:
        --------
                    { float }            : BIC score (lower is better)
        """
        n              = len(signal)

        # Number of change points
        K              = len(change_points) - 1  
        
        # Calculate residual variance for each segment
        boundaries     = [0] + change_points
        total_residual = 0
        
        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            segment    = signal[start:end]
            
            if (len(segment) > 0):
                segment_mean    = np.mean(segment)
                residuals       = segment - segment_mean
                total_residual += np.sum(residuals ** 2)
        
        # Residual variance
        sigma_squared = total_residual / n if (n > 0) else 1e-10
        
        # Prevent log(0)
        sigma_squared = max(sigma_squared, 1e-10)
        
        # BIC formula
        bic           = n * np.log(sigma_squared) + K * np.log(n)
        
        return bic
    

    def plot_tuning_curve(self, penalties: List[float], bic_scores: List[float], n_changepoints: List[int], optimal_penalty: float, save_path: Optional[str] = None):
        """
        Plot BIC curve and change point count vs penalty
        
        Arguments:
        ----------
            penalties       { list }  : List of penalty values tested
            
            bic_scores      { list }  : Corresponding BIC scores
            
            n_changepoints  { list }  : Number of change points for each penalty
            
            optimal_penalty { float } : Optimal penalty value
            
            save_path       { str }   : Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(nrows   = 1, 
                                       ncols   = 2, 
                                       figsize = (16, 6),
                                      )
        
        # Plot BIC vs Penalty
        ax1.plot(penalties, 
                 bic_scores, 
                 marker     = 'b-', 
                 linewidth  = 2, 
                 label      = 'BIC',
                )

        ax1.axvline(optimal_penalty,
                    color     = 'red',
                    linestyle = '--',
                    linewidth = 2,
                    label     = f'Optimal: {optimal_penalty:.3f}',
                   )

        ax1.set_xlabel('Penalty Value', fontsize = 12)
        ax1.set_ylabel('BIC Score', fontsize = 12)
        ax1.set_title('BIC vs Penalty Parameter', fontsize = 14, pad = 15)
        ax1.set_xscale('log')
        ax1.grid(True, alpha = 0.3)
        ax1.legend(fontsize = 11)
        
        # Plot Number of change points vs Penalty
        ax2.plot(penalties, 
                 n_changepoints, 
                 marker     = 'g-o', 
                 linewidth  = 2,
                 markersize = 6,
                )

        ax2.axvline(optimal_penalty,
                    color     = 'red',
                    linestyle = '--',
                    linewidth = 2,
                    alpha     = 0.7,
                   )

        optimal_k = n_changepoints[penalties.index(optimal_penalty)]

        ax2.scatter([optimal_penalty],
                    [optimal_k],
                    color  = 'red',
                    s      = 200,
                    zorder = 5,
                    label  = f'K={optimal_k} at optimal penalty',
                   )

        ax2.set_xlabel('Penalty Value', fontsize = 12)
        ax2.set_ylabel('Number of Change Points', fontsize = 12)
        ax2.set_title('Change Points vs Penalty', fontsize = 14, pad = 15)
        ax2.set_xscale('log')
        ax2.grid(True, alpha = 0.3)
        ax2.legend(fontsize = 11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(fname       = save_path, 
                        dpi         = 300, 
                        bbox_inches = 'tight',
                       )

            plt.close()

        else:
            plt.show()



def tune_penalty_bic(signal: np.ndarray, cost_model: str = 'l1', min_size: int = 5, penalty_range: Tuple[float, float] = (0.1, 10.0), plot: bool = False, save_path: Optional[str] = None) -> Tuple[float, dict]:
    """
    Convenience function to tune penalty using BIC
    
    Arguments:
    ----------
        signal        { np.ndarray } : 1D signal array
        
        cost_model        { str }    : Cost function
        
        min_size          { int }    : Minimum segment size
        
        penalty_range    { tuple }   : Range of penalties to test
        
        plot             { bool }    : Whether to plot tuning curve
        
        save_path         { str }    : Path to save plot
    
    Returns:
    --------
                { tuple }            : A python tuple containing:
                                       - optimal_penalty
                                       - tuning_results_dict
    """
    tuner                                         = PenaltyTuner(cost_model = cost_model, 
                                                                 min_size   = min_size,
                                                                )
    
    optimal_penalty, penalties, n_cps, bic_scores = tuner.tune(signal        = signal, 
                                                               penalty_range = penalty_range,
                                                              )
    
    results                                       = {'optimal_penalty'        : optimal_penalty,
                                                     'penalties_tested'       : penalties,
                                                     'n_changepoints'         : n_cps,
                                                     'bic_scores'             : bic_scores,
                                                     'optimal_n_changepoints' : n_cps[penalties.index(optimal_penalty)],
                                                    }
    
    if plot:
        tuner.plot_tuning_curve(penalties       = penalties,
                                bic_scores      = bic_scores,
                                n_changepoints  = n_cps,
                                optimal_penalty = optimal_penalty,
                                save_path       = save_path,
                               )
                        
    return optimal_penalty, results
