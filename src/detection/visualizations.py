# Dependencies
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict
from typing import List 
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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


    @staticmethod
    def _normalize_change_points(change_points: List, signal_length: int, method: str) -> List[int]:
        """
        Convert change points to integer indices regardless of input format

        PELT  :  change_points are already integers [24, 58, 135]
        BOCPD : change_points are normalized floats [0.0657, 0.1589, 0.3698]

        Arguments:
        ----------
            change_points  { list }  : Change point positions (int or float)

            signal_length  { int }   : Total signal length
            
            method         { str }   : Detection method ('pelt' or 'bocpd')

        Returns:
        --------
                          { list }  : Integer indices
        """
        if not change_points:
            return []

        # Check if already integers (PELT format)
        if all(isinstance(cp, (int, np.integer)) for cp in change_points):
            return [int(cp) for cp in change_points]

        # Convert normalized positions to indices (BOCPD format)
        if all(isinstance(cp, (float, np.floating)) for cp in change_points):
            if all(0 <= cp <= 1 for cp in change_points):
                # Normalized positions - convert to indices
                return [int(cp * signal_length) for cp in change_points]

        # Fallback: assume they're already indices
        return [int(cp) for cp in change_points]


    def plot_aggregated_cv_with_all_models(self, aggregated_data: pd.Series, model_results: Dict[str, Dict], save_path: Optional[Path] = None):
        """
        Plot aggregated CV with change points from all models overlaid

        Each model is expected to provide:
        ---------------------------------
        - change_points : List[int] or List[float]  (handles both formats)
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
                zorder    = 1,
               )

        # Overlay CPs from each model
        for model_id, result in model_results.items():
            cps           = result.get("change_points", [])
            color         = result.get("color", "red")
            linestyle     = result.get("linestyle", "--")
            method        = result.get("method", "unknown")
            signal_length = result.get("signal_length", len(aggregated_data))

            # Normalize change points to integer indices
            cps_indices   = self._normalize_change_points(cps, signal_length, method)

            for i, cp in enumerate(cps_indices):
                ax.axvline(cp,
                           color     = color,
                           linestyle = linestyle,
                           alpha     = 0.8,
                           linewidth = 2,
                           label     = model_id if (i == 0) else None,
                           zorder    = 2,
                          )

        ax.set_title("Aggregated CV with Change Points (All Models)", fontsize=16, pad=20)
        ax.set_xlabel("Time Index", fontsize=13)
        ax.set_ylabel("Coefficient of Variation", fontsize = 13)
        ax.grid(True, alpha = 0.3, linestyle = ':')
        ax.legend(loc = "upper right", fontsize = 10, framealpha = 0.9)

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

        axes      = np.atleast_2d(axes).flatten()
        x         = np.arange(len(aggregated_data))

        for idx, (model_id, result) in enumerate(model_results.items()):
            ax            = axes[idx]

            ax.plot(x,
                    aggregated_data.values,
                    color     = "black",
                    linewidth = 2,
                    alpha     = 0.8,
                   )

            cps           = result.get("change_points", [])
            color         = result.get("color", "red")
            linestyle     = result.get("linestyle", "--")
            method        = result.get("method", "unknown")
            signal_length = result.get("signal_length", len(aggregated_data))

            # Normalize change points to integer indices
            cps_indices   = self._normalize_change_points(cps, signal_length, method)

            for cp in cps_indices:
                ax.axvline(cp, 
                           color     = color, 
                           linestyle = linestyle, 
                           alpha     = 0.9,
                           linewidth = 2,
                          )

            n_cps = len(cps_indices)
            
            # Show validation status
            validation = result.get('validation', {})
            is_valid   = validation.get('overall_significant', False)
            status     = "✓" if is_valid else "✗"

            ax.set_title(f"{model_id} {status} | CPs = {n_cps}", fontsize = 11, pad=10)

            ax.grid(True, alpha = 0.3, linestyle = ':')

        # Hide unused axes
        for i in range(n_models, len(axes)):
            axes[i].axis("off")

        fig.suptitle("Model Comparison Grid (✓ = statistically valid)", fontsize = 15, y = 0.995)
        plt.tight_layout()

        self._save_or_show(save_path = save_path)


    # BOCPD POSTERIOR DIAGNOSTICS
    def plot_bocpd_posterior(self, run_length_posterior: np.ndarray, cp_posterior: np.ndarray, posterior_threshold: float, save_path: Optional[Path] = None):
        """
        Plot BOCPD diagnostics:

        - Run-length posterior heatmap
        - Change-point posterior probability
        """
        fig = plt.figure(figsize=(16, 10))
        gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        # Heatmap
        sns.heatmap(run_length_posterior.T,
                    cmap   = "viridis",
                    ax     = ax1,
                    cbar_kws = {'label': 'P(run length | data)'},
                   )

        ax1.set_title("BOCPD Run-Length Posterior", fontsize = 14, pad = 15)
        ax1.set_ylabel("Run Length", fontsize = 12)
        ax1.set_xlabel("")

        # CP posterior
        ax2.plot(cp_posterior, color = "blue", linewidth = 2.5, label = 'P(change point)')
        
        ax2.axhline(posterior_threshold,
                    color     = "red",
                    linestyle = "--",
                    linewidth = 2,
                    label     = f"Threshold = {posterior_threshold:.2f}",
                   )

        # Shade detected regions
        detected = (cp_posterior >= posterior_threshold)
        if detected.any():
            ax2.fill_between(range(len(cp_posterior)),
                             0, 1,
                             where = detected,
                             alpha = 0.2,
                             color = 'red',
                             label = 'Detected CPs',
                            )

        ax2.set_title("BOCPD Change-Point Posterior", fontsize = 14, pad = 15)
        ax2.set_xlabel("Time Index", fontsize = 12)
        ax2.set_ylabel("Posterior Probability", fontsize = 12)
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, alpha = 0.3, linestyle = ':')
        ax2.legend(loc = 'upper right', fontsize = 10)

        self._save_or_show(save_path = save_path)


    # Diagnostic plots
    def plot_segment_diagnostics(self, aggregated_data: pd.Series, change_points: List, model_id: str, method: str = 'pelt', save_path: Optional[Path] = None):
        """
        Plot segment-level diagnostics

        Shows:
        ------
        - Segment means and variances
        - Residuals within segments
        - Segment lengths
        """
        # Normalize change points
        signal_length = len(aggregated_data)
        change_points = self._normalize_change_points(change_points = change_points, 
                                                      signal_length = signal_length, 
                                                      method        = method,
                                                     )

        fig          = plt.figure(figsize = (16, 12))
        gs           = GridSpec(3, 1, height_ratios = [2, 1, 1], hspace = 0.3)

        # Prepare segments
        boundaries   = [0] + list(change_points) + [len(aggregated_data)]
        segments     = []

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            seg_data = aggregated_data.iloc[start:end]
            segments.append({'start': start,
                             'end': end,
                             'mean': seg_data.mean(),
                             'std': seg_data.std(),
                             'length': end - start,
                           })

        # Signal with segment means
        ax1 = fig.add_subplot(gs[0])
        x   = np.arange(len(aggregated_data))

        ax1.plot(x, 
                 aggregated_data.values, 
                 'k-', 
                 alpha     = 0.6, 
                 linewidth = 1.5, 
                 label     = 'Signal',
                )

        for i, seg in enumerate(segments):
            seg_x = range(seg['start'], seg['end'])
            ax1.plot(seg_x, 
                     [seg['mean']] * len(seg_x), 
                     linewidth = 3, 
                     label     = f"Seg {i+1} (μ={seg['mean']:.3f})",
                    )

        for cp in change_points:
            ax1.axvline(cp, color = 'red', linestyle = '--', alpha = 0.7)

        ax1.set_title(f"Segment Means: {model_id}", fontsize = 14, pad = 15)
        ax1.set_ylabel("CV Value", fontsize = 12)
        ax1.grid(True, alpha = 0.3)
        ax1.legend(loc = 'best', fontsize = 9)

        # Residuals
        ax2 = fig.add_subplot(gs[1])
        
        for seg in segments:
            seg_data  = aggregated_data.iloc[seg['start']:seg['end']]
            residuals = seg_data - seg['mean']
            ax2.scatter(range(seg['start'], seg['end']), residuals, alpha = 0.6, s = 20)

        ax2.axhline(0, color = 'black', linestyle = '-', linewidth = 1)
        ax2.set_ylabel("Residuals", fontsize = 12)
        ax2.set_title("Within-Segment Residuals", fontsize = 13)
        ax2.grid(True, alpha = 0.3)

        # Segment statistics
        ax3         = fig.add_subplot(gs[2])
        
        seg_indices = range(len(segments))
        seg_lengths = [s['length'] for s in segments]
        seg_vars    = [s['std']**2 for s in segments]

        ax3_twin    = ax3.twinx()

        ax3.bar(seg_indices, 
                seg_lengths, 
                alpha = 0.6, 
                label = 'Length', 
                color = 'steelblue',
               )

        ax3_twin.plot(seg_indices, seg_vars, 'ro-', linewidth = 2, markersize = 8, label = 'Variance')

        ax3.set_xlabel("Segment Index", fontsize = 12)
        ax3.set_ylabel("Segment Length", fontsize = 12, color = 'steelblue')
        ax3_twin.set_ylabel("Segment Variance", fontsize = 12, color = 'red')
        ax3.set_title("Segment Statistics", fontsize = 13)
        ax3.grid(True, alpha = 0.3)

        self._save_or_show(save_path = save_path)


    # UTILITY FUNCTION
    def _save_or_show(self, save_path: Optional[Path]):
        if save_path:
            save_path.parent.mkdir(parents = True, exist_ok = True)

            plt.savefig(fname       = save_path,
                        dpi         = self.dpi,
                        bbox_inches = "tight",
                       )

            plt.close()

        else:
            plt.show()