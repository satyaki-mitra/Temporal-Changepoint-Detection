# Dependencies
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict
from typing import List 
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt


class DetectionVisualizer:
    """
    Model-aware visualization engine for change point detection

    Design principles:
    ------------------
    - Detector-agnostic (PELT / BOCPD treated uniformly)
    - Model-first API (each model has an ID and metadata)
    - Fair visual comparison (all CPs shown as vertical lines)
    - Production-safe (no implicit assumptions)
    """
    def __init__(self, figsize: tuple = (16, 10), dpi: int = 300, style: str = "default"):
        """
        Initialize the Visualizer Engine
        """
        self.figsize = figsize
        self.dpi     = dpi

        plt.style.use(style)


    def plot_aggregated_cv_with_all_models(self, aggregated_data: pd.Series, model_results: Dict[str, Dict], save_path: Optional[Path] = None,):
        """
        Plot aggregated CV with change points from all models overlaid

        Each model is expected to provide:
        ---------------------------------
        - change_points : List[int]
        - color         : str
        - linestyle     : str
        """
        fig, ax = plt.subplots(figsize = self.figsize)
        x       = np.arange(len(aggregated_data))

        # Base signal
        ax.plot(x,
                aggregated_data.values,
                color     = "black",
                linewidth = 2.5,
                label     = "Aggregated CV",
               )

        # Overlay CPs from each model
        for model_id, result in model_results.items():
            cps       = result.get("change_points", [])
            color     = result.get("color", "red")
            linestyle = result.get("linestyle", "--")

            for i, cp in enumerate(cps):
                ax.axvline(cp,
                           color     = color,
                           linestyle = linestyle,
                           alpha     = 0.8,
                           linewidth = 2,
                           label     = model_id if (i == 0) else None,
                          )

        ax.set_title("Aggregated CV with Change Points (All Models)")
        ax.set_xlabel("Time Index")
        ax.set_ylabel("Coefficient of Variation")
        ax.grid(True, alpha = 0.3)
        ax.legend(loc = "upper right", fontsize = 9)

        plt.tight_layout()

        self._save_or_show(save_path = save_path)


    # MODEL COMPARISON GRID (SIDE-BY-SIDE)
    def plot_model_comparison_grid(self, aggregated_data: pd.Series, model_results: Dict[str, Dict], n_cols: int = 3, save_path: Optional[Path] = None):
        """
        Side-by-side comparison grid for all models

        Each subplot:
        -------------
        - Aggregated CV
        - Model-specific change points
        - Model metadata in title
        """
        n_models  = len(model_results)
        n_rows    = int(np.ceil(n_models / n_cols))

        fig, axes = plt.subplots(nrows   = n_rows,
                                 ncols   = n_cols,
                                 figsize = (self.figsize[0], self.figsize[1] * n_rows / 2),
                                 sharex  = True,
                                )

        axes      = np.atleast_2d(axes)
        x         = np.arange(len(aggregated_data))

        for idx, (model_id, result) in enumerate(model_results.items()):
            row, col = divmod(idx, n_cols)
            ax       = axes[row, col]

            ax.plot(x,
                    aggregated_data.values,
                    color     = "black",
                    linewidth = 2,
                    alpha     = 0.8,
                   )

            cps       = result.get("change_points", [])
            color     = result.get("color", "red")
            linestyle = result.get("linestyle", "--")

            for cp in cps:
                ax.axvline(cp, 
                           color     = color, 
                           linestyle = linestyle, 
                           alpha     = 0.9,
                          )

            n_cps = len(cps)
            ax.set_title(f"{model_id}  |  CPs={n_cps}", fontsize = 10)

            ax.grid(True, alpha = 0.3)

        # Hide unused axes
        for i in range(n_models, n_rows * n_cols):
            axes.flat[i].axis("off")

        plt.tight_layout()

        self._save_or_show(save_path = save_path)


    # BOCPD POSTERIOR DIAGNOSTICS
    def plot_bocpd_posterior(self, run_length_posterior: np.ndarray, cp_posterior: np.ndarray, posterior_threshold: float, save_path: Optional[Path] = None):
        """
        Plot BOCPD diagnostics:

        - Run-length posterior heatmap
        - Change-point posterior probability
        """
        fig, axes = plt.subplots(nrows       = 2,
                                 ncols       = 1,
                                 figsize     = (16, 10),
                                 gridspec_kw = {"height_ratios": [3, 1]},
                                )

        sns.heatmap(run_length_posterior.T,
                    cmap   = "viridis",
                    ax     = axes[0],
                    cbar   = True,
                   )

        axes[0].set_title("BOCPD Run-Length Posterior")
        axes[0].set_ylabel("Run Length")

        axes[1].plot(cp_posterior, color = "blue", linewidth = 2)
        
        axes[1].axhline(posterior_threshold,
                        color     = "red",
                        linestyle = "--",
                        linewidth = 2,
                        label     = "Posterior Threshold",
                       )

        axes[1].set_title("BOCPD Change-Point Posterior")
        axes[1].set_xlabel("Time Index")
        axes[1].legend()

        plt.tight_layout()

        self._save_or_show(save_path = save_path)


    # UTILITY FUNCTION
    def _save_or_show(self, save_path: Optional[Path]):
        if save_path:
            plt.savefig(fname       = save_path,
                        dpi         = self.dpi,
                        bbox_inches = "tight",
                       )

            plt.close()

        else:
            plt.show()