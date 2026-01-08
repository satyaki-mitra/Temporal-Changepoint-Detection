# Dependencies
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List
from typing import Dict
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt


class DetectionVisualizer:
    """
    Unified visualization engine for change point detection

    Supports:
    - Offline (PELT) visualizations
    - Online Bayesian (BOCPD) posterior plots
    - Side-by-side detector comparison
    - Shared x-axis alignment across detectors
    """
    def __init__(self, figsize: tuple = (15, 10), dpi: int = 300, style: str = 'default'):
        """
        Initialize visualizer

        Arguments:
        ----------
            figsize { tuple } : Default figure size
            
            dpi      { int }  : DPI for saved figures
            
            style    { str }  : Matplotlib style
        """
        self.figsize = figsize
        self.dpi     = dpi

        plt.style.use(style)

    # Aggregated Statistic Plot
    def plot_aggregated_cv(self, aggregated_data: pd.Series, pelt_cps: Optional[List[int]] = None, bocpd_cps: Optional[List[int]] = None, save_path: Optional[Path] = None):
        """
        Plot aggregated CV with optional PELT and BOCPD change points
        """
        fig, ax = plt.subplots(figsize = self.figsize)
        x       = np.arange(len(aggregated_data))

        ax.plot(x, 
                aggregated_data.values,
                color     = 'black', 
                linewidth = 2, 
                label     = 'Aggregated CV',
               )

        if pelt_cps:
            for i, cp in enumerate(pelt_cps):
                ax.axvline(cp, 
                           color     = 'red', 
                           linestyle = '--', 
                           alpha     = 0.7,
                           label     = 'PELT CP' if (i == 0) else None,
                          )

        if bocpd_cps:
            ax.scatter(bocpd_cps,
                       aggregated_data.iloc[bocpd_cps],
                       color  = 'blue', 
                       s      = 80, 
                       marker = 'D',
                       label  = 'BOCPD CP',
                      )

        ax.set_xlabel('Time Index')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Aggregated CV with Detected Change Points')
        ax.grid(True, alpha = 0.3)
        ax.legend()

        plt.tight_layout()

        self._save_or_show(save_path)


    # Raw Scatter + Segmentation (PELT)
    def plot_change_points_scatter(self, raw_data: pd.DataFrame, aggregated_data: pd.Series, change_points: List[int], smoothing_window: int = 10, save_path: Optional[Path] = None):
        """
        Scatter plot of raw PHQ-9 scores with PELT segmentation
        """
        fig, ax = plt.subplots(figsize = (20, 12))
        x       = np.arange(len(raw_data))

        for idx, (_, row) in enumerate(raw_data.iterrows()):
            vals = row.dropna()

            ax.scatter([idx]*len(vals), 
                       vals, 
                       color = 'gray', 
                       alpha = 0.4, 
                       s     = 25,
                      )

        smoothed = raw_data.mean(axis = 1).rolling(smoothing_window, 
                                                   center      = True, 
                                                   min_periods = smoothing_window,
                                                  ).mean()

        ax.plot(x, 
                smoothed, 
                color     = 'blue', 
                linewidth = 2, 
                label     = 'Smoothed Mean',
               )

        for cp in change_points:
            ax.axvline(cp, 
                       color     = 'red', 
                       linestyle = '--', 
                       alpha     = 0.6,
                      )

        ax.set_ylim(0, 27)
        ax.set_xlabel('Time Index')
        ax.set_ylabel('PHQ-9 Score')
        ax.set_title('Raw PHQ-9 Scores with PELT Segments')
        ax.grid(True, alpha = 0.3)
        ax.legend()

        plt.tight_layout()

        self._save_or_show(save_path)


    # Validation Diagnostics (PELT)
    def plot_validation_diagnostics(self, aggregated_data: pd.Series, pelt_cps: List[int], validation_results: Dict, save_path: Optional[Path] = None):
        """
        Diagnostic plots for frequentist validation of PELT change points
        """
        fig, axes = plt.subplots(nrows   = 2, 
                                 ncols   = 2, 
                                 figsize = (18, 12),
                                )

        x         = np.arange(len(aggregated_data))

        # CV + CPs
        ax        = axes[0, 0]

        ax.plot(x, 
                aggregated_data.values, 
                color     = 'black', 
                linewidth = 2,
               )

        for cp in pelt_cps:
            ax.axvline(cp, 
                       color     = 'red', 
                       linestyle = '--', 
                       alpha     = 0.7,
                      )

        ax.set_title('CV with Change Points')

        # P-values
        ax    = axes[0, 1]
        tests = validation_results.get('tests', {})

        if tests:
            labels = list(tests.keys())
            pvals  = [t['p_value_corrected'] for t in tests.values()]

            ax.bar(labels, pvals)

            ax.axhline(0.05, color = 'red', linestyle = '--')

            ax.set_title('Corrected p-values')
            ax.set_xticklabels(labels, rotation = 45)

        # Effect sizes
        ax = axes[1, 0]

        if tests:
            ds = [t['cohens_d'] for t in tests.values()]

            ax.barh(labels, ds)
            ax.axvline(0.3, linestyle = '--', color = 'orange')
            ax.axvline(0.5, linestyle = '--', color = 'green')

            ax.set_title("Effect Size (Cohen's d)")

        # Segment means
        ax         = axes[1, 1]
        boundaries = [0] + pelt_cps + [len(aggregated_data)]
        means      = [aggregated_data.iloc[boundaries[i]:boundaries[i+1]].mean() for i in range(len(boundaries)-1)]
        
        ax.plot(means, marker = 'o')
        
        ax.set_title('Segment Means')

        plt.tight_layout()

        self._save_or_show(save_path)


    # BOCPD Posterior Visualizations
    def plot_bocpd_posterior(self, run_length_posterior: np.ndarray, cp_posterior: Optional[np.ndarray] = None, save_path: Optional[Path] = None):
        """
        Plot BOCPD run-length posterior heatmap and CP posterior
        """
        fig, axes = plt.subplots(nrows       = 2, 
                                 ncols       = 1, 
                                 figsize     = (16, 10), 
                                 gridspec_kw = {'height_ratios': [3, 1]},
                                )

        sns.heatmap(run_length_posterior.T,
                    cmap = 'viridis',
                    ax   = axes[0],
                    cbar = True,
                   )

        axes[0].set_title('BOCPD Run-Length Posterior')
        axes[0].set_ylabel('Run Length')

        if cp_posterior is not None:
            axes[1].plot(cp_posterior, color = 'blue')

            axes[1].axhline(0.6, color = 'red', linestyle = '--')

            axes[1].set_title('Change Point Posterior Probability')

        axes[1].set_xlabel('Time Index')

        plt.tight_layout()

        self._save_or_show(save_path)


    # Penalty Tuning (PELT)
    def plot_penalty_tuning(self, penalties: List[float], bic_scores: List[float], n_changepoints: List[int], optimal_penalty: float, save_path: Optional[Path] = None):
        """
        Plot penalty tuning diagnostics for PELT
        """
        fig, axes = plt.subplots(nrows   = 1, 
                                 ncols   = 2, 
                                 figsize = (16, 6),
                                )

        axes[0].plot(penalties, bic_scores)

        axes[0].axvline(optimal_penalty, linestyle = '--', color = 'red')

        axes[0].set_xscale('log')
        axes[0].set_title('BIC vs Penalty')

        axes[1].plot(penalties, n_changepoints)

        axes[1].axvline(optimal_penalty, linestyle = '--', color = 'red')

        axes[1].set_xscale('log')
        axes[1].set_title('Change Points vs Penalty')

        plt.tight_layout()

        self._save_or_show(save_path)


    # Model Comparison Grid (NEW)
    def plot_model_comparison_grid(self, aggregated_data: pd.Series, model_results: Dict[str, Dict], n_cols: int = 2, save_path: Optional[Path] = None):
        """
        Matrix-style visual comparison of change point detections across models

        Parameters:
        ----------
            aggregated_data { pd.Series } : Aggregated CV time series

            model_results     { dict }    : Dictionary mapping model identifiers to their detection results 

            n_cols             { int }    : Number of columns in grid
        """
        n_models  = len(model_results)

        # +1 for reference row
        n_rows    = int(np.ceil(n_models / n_cols)) + 1  

        fig, axes = plt.subplots(nrows   = n_rows, 
                                 ncols   = n_cols,
                                 figsize = (self.figsize[0], self.figsize[1] * n_rows),
                                 sharex  = True,
                                )

        axes      = np.atleast_2d(axes)
        x         = np.arange(len(aggregated_data))

        # Reference row (Aggregated CV)
        for col in range(n_cols):
            ax = axes[0, col]

            ax.plot(x, 
                    aggregated_data.values, 
                    color     = 'black', 
                    linewidth = 2,
                   )
            
            ax.set_title('Aggregated CV (Reference)')
            ax.grid(True, alpha = 0.3)

        # Per-model plots
        for idx, (model_id, result) in enumerate(model_results.items()):
            row = idx // n_cols + 1
            col = idx % n_cols
            ax  = axes[row, col]

            ax.plot(x, 
                    aggregated_data.values, 
                    color = 'gray', 
                    alpha = 0.6,
                   )

            cps = result.get('change_points', [])
            
            for cp in cps:
                ax.axvline(cp,
                           color     = result.get('color', 'red'),
                           linestyle = result.get('linestyle', '--'),
                           linewidth = 2,
                           alpha     = 0.8,
                          )

            ax.set_title(model_id)

            ax.grid(True, alpha = 0.3)

        # Hide unused axes
        for i in range(n_models + 1, n_rows * n_cols):
            axes.flat[i].axis('off')

        plt.tight_layout()

        self._save_or_show(save_path)


    # Utilities
    def _save_or_show(self, save_path: Optional[Path]):
        """
        Save figure if path provided, else show interactively
        """
        if save_path:
            plt.savefig(fname       = save_path, 
                        dpi         = self.dpi, 
                        bbox_inches = 'tight',
                       )

            plt.close()
            
        else:
            plt.show()