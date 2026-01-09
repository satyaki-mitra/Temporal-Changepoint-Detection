# Dependencies
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List
from typing import Tuple
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt


class VisualizationGenerator:
    """
    Central class for generating all EDA visualizations
    """
    def __init__(self, figsize: Tuple[int, int] = (15, 10), dpi: int = 300, style: str = 'seaborn', color_palette: str = 'husl'):
        """
        Initialize visualization generator
        
        Arguments:
        ----------
            figsize       { tuple } : Default figure size

            dpi            { int }  : DPI for saved figures
            
            style          { str }  : Matplotlib style
            
            color_palette  { str }  : Seaborn color palette
        """
        self.figsize = figsize
        self.dpi     = dpi
        
        # Set style
        plt.style.use(style if (style != 'seaborn') else 'default')
        sns.set_palette(color_palette)

    
    def plot_scatter(self, data: pd.DataFrame, save_path: Optional[Path] = None):
        """
        Create scatter plot of all PHQ-9 scores
        
        Arguments:
        ----------
            data      { pd.DataFrame } : PHQ-9 DataFrame (Days x Patients)

            save_path     { Path }     : Path to save figure
        """
        fig, ax = plt.subplots(figsize = self.figsize)
        
        # Plot each day's scores
        for day_idx, (day_name, day_scores) in enumerate(data.iterrows()):
            scores = day_scores.dropna()
            
            if (len(scores) > 0):
                # Create color gradient based on observation order
                colors = np.linspace(0, 1, len(scores))
                
                ax.scatter([day_idx] * len(scores),
                           scores.values,
                           c     = colors,
                           cmap  = 'viridis',
                           s     = 30,
                           alpha = 0.6,
                          )
        
        # Formatting
        ax.set_xlabel('Day Index', fontsize = 12)
        ax.set_ylabel('PHQ-9 Score', fontsize = 12)
        ax.set_title('PHQ-9 Score Distribution Across All Days', fontsize = 16, pad = 20)
        ax.set_ylim(0, 27)
        ax.grid(True, alpha = 0.3)
        
        # Color bar
        sm   = plt.cm.ScalarMappable(cmap = 'viridis')
        cbar = plt.colorbar(sm, ax = ax)
        cbar.set_label('Score Order Within Day', fontsize = 10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(fname       = save_path, 
                        dpi         = self.dpi, 
                        bbox_inches = 'tight',
                       )

            plt.close()

        else:
            plt.show()
    

    def plot_daily_averages(self, data: pd.DataFrame, save_path: Optional[Path] = None):
        """
        Plot daily average PHQ-9 scores with trend line
        
        Arguments:
        ----------
            data      { pd.DataFrame } : PHQ-9 DataFrame

            save_path     { Path }     : Path to save figure
        """
        # Calculate daily averages
        daily_avg = data.mean(axis   = 1, 
                              skipna = True,
                             )

        days      = np.arange(len(daily_avg))
        
        fig, ax   = plt.subplots(figsize = self.figsize)
        
        # Plot daily averages
        ax.plot(days, 
                daily_avg, 
                'o-', 
                color      = 'steelblue', 
                linewidth  = 2, 
                markersize = 8,
                alpha      = 0.7,
                label      = 'Daily Average',
               )
        
        # Add trend line
        z = np.polyfit(days, daily_avg, 1)
        p = np.poly1d(z)

        ax.plot(days, 
                p(days), 
                '--', 
                color     = 'red', 
                linewidth = 2,
                alpha     = 0.8, 
                label     = f'Trend (slope: {z[0]:.3f})',
               )
        
        # Add severity level lines
        severity_levels = {'Minimal'  : 5,
                           'Mild'     : 10,
                           'Moderate' : 15,
                           'Severe'   : 20,
                          }

        colors          = ['green', 
                           'yellow', 
                           'orange', 
                           'red',
                          ]
        
        for (level, score), color in zip(severity_levels.items(), colors):
            ax.axhline(y         = score, 
                       color     = color, 
                       linestyle = ':', 
                       alpha     = 0.5, 
                       label     = f'{level} ({score})',
                      )
        
        # Formatting
        ax.set_xlabel('Day Index', fontsize = 12)
        ax.set_ylabel('Average PHQ-9 Score', fontsize = 12)
        ax.set_title('Daily Average PHQ-9 Scores Over Time', fontsize = 16, pad = 20)
        ax.set_ylim(0, 27)
        ax.grid(True, alpha = 0.3)
        ax.legend(bbox_to_anchor = (1.02, 1), loc = 'upper left', fontsize = 10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(fname       = save_path, 
                        dpi         = self.dpi, 
                        bbox_inches = 'tight',
                       )

            plt.close()

        else:
            plt.show()
    

    def plot_cluster_optimization(self, inertias: List[float], silhouettes: List[float], elbow_k: int, silhouette_k: int, save_path: Optional[Path] = None):
        """
        Plot elbow and silhouette analysis for optimal cluster selection
        
        Arguments:
        ----------
            inertias     { list } : List of inertia values
            
            silhouettes  { list } : List of silhouette scores
            
            elbow_k      { int }  : Optimal K from elbow method
            
            silhouette_k { int }  : Optimal K from silhouette
            
            save_path    { Path } : Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(nrows   = 1, 
                                       ncols   = 2, 
                                       figsize = (16, 6),
                                      )
        
        k_values        = np.arange(2, len(inertias) + 2)
        
        # Elbow plot
        ax1.plot(k_values, inertias, 'bo-', linewidth = 2, markersize = 8)

        ax1.axvline(x         = elbow_k, 
                    color     = 'red', 
                    linestyle = '--', 
                    linewidth = 2, 
                    alpha     = 0.7, 
                    label     = f'Elbow: K={elbow_k}',
                   )

        ax1.set_xlabel('Number of Clusters (K)', fontsize = 12)
        ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize = 12)
        ax1.set_title('Elbow Method for Optimal K', fontsize = 14, pad = 15)
        ax1.grid(True, alpha = 0.3)
        ax1.legend(fontsize = 11)
        
        # Silhouette plot
        ax2.plot(k_values, silhouettes, 'go-', linewidth = 2, markersize = 8)

        ax2.axvline(x         = silhouette_k, 
                    color     = 'red', 
                    linestyle = '--',
                    linewidth = 2, 
                    alpha     = 0.7, 
                    label     = f'Optimal: K={silhouette_k}',
                   )

        ax2.set_xlabel('Number of Clusters (K)', fontsize = 12)
        ax2.set_ylabel('Silhouette Score', fontsize = 12)
        ax2.set_title('Silhouette Analysis for Optimal K', fontsize = 14, pad = 15)
        ax2.grid(True, alpha = 0.3)
        ax2.legend(fontsize = 11)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(fname       = save_path, 
                        dpi         = self.dpi, 
                        bbox_inches = 'tight',
                       )

            plt.close()

        else:
            plt.show()
    

    def plot_clusters(self, data: pd.DataFrame, labels: np.ndarray, n_clusters: int, save_path: Optional[Path] = None):
        """
        Plot clustering results with cluster colors and boundaries
        
        Arguments:
        ----------
            data       { pd.DataFrame } : PHQ-9 DataFrame

            labels      { np.ndarray }  : Cluster labels for each day
            
            n_clusters      { int }     : Number of clusters
            
            save_path       { Path }    : Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(nrows   = 2, 
                                       ncols   = 1, 
                                       figsize = (self.figsize[0], self.figsize[1] * 1.2),
                                      )
        
        # Get colors for clusters
        colors          = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        
        # Plot 1: Scatter plot with cluster colors
        for day_idx, (day_name, day_scores) in enumerate(data.iterrows()):
            if (day_idx < len(labels)):
                cluster_id = labels[day_idx]
                scores     = day_scores.dropna()
                
                if (len(scores) > 0):
                    # Only add label for first occurrence of each cluster
                    label = f'Cluster {cluster_id}' if (day_idx == np.where(labels == cluster_id)[0][0]) else None
                    
                    ax1.scatter([day_idx] * len(scores),
                                scores.values,
                                c     = [colors[cluster_id]],
                                s     = 50,
                                alpha = 0.7,
                                label = label,
                               )
        
        ax1.set_xlabel('Day Index', fontsize = 12)
        ax1.set_ylabel('PHQ-9 Score', fontsize = 12)
        ax1.set_title(f'Clustering Results: {n_clusters} Clusters (Scatter View)', fontsize = 14, pad = 15)
        ax1.set_ylim(0, 27)
        ax1.grid(True, alpha = 0.3)
        ax1.legend(bbox_to_anchor = (1.02, 1), loc = 'upper left', fontsize = 10)
        
        # Plot 2: Daily averages with cluster boundaries
        daily_avg = data.mean(axis = 1, skipna = True)
        
        for cluster_id in range(n_clusters):
            cluster_mask = (labels == cluster_id)
            cluster_days = np.where(cluster_mask)[0]
            
            if (len(cluster_days) > 0):
                ax2.scatter(cluster_days,
                            daily_avg.iloc[cluster_days],
                            c     = [colors[cluster_id]],
                            s     = 100,
                            alpha = 0.8,
                            label = f'Cluster {cluster_id}',
                           )
        
        # Only draw boundaries at true segment transitions: Filter out isolated single-day flips by checking neighbors
        cluster_boundaries = list()

        for i in range(len(labels) - 1):
            if (labels[i] != labels[i + 1]):
                # Check if this is a true segment boundary or just noise
                is_valid_boundary = False
                
                # Edge cases: first and last transitions are always valid
                if (i == 0) or (i == len(labels) - 2):
                    is_valid_boundary = True
                
                # Middle transitions: valid if either side has consistent cluster
                else:
                    # Check if current cluster continues before this point
                    left_consistent   = (labels[i - 1] == labels[i])
                    
                    # Check if next cluster continues after this point
                    right_consistent  = (labels[i + 1] == labels[i + 2])
                    
                    # Valid boundary if at least one side is consistent (not isolated flip)
                    is_valid_boundary = (left_consistent or right_consistent)
                
                if is_valid_boundary:
                    cluster_boundaries.append(i + 0.5)
        
        # Draw filtered boundaries
        for boundary in cluster_boundaries:
            ax2.axvline(x         = boundary, 
                        linestyle = '--', 
                        color     = 'gray', 
                        alpha     = 0.6,
                        linewidth = 1.5,
                       )
        
        # Add trend line
        days = np.arange(len(daily_avg))
        ax2.plot(days, 
                 daily_avg, 
                 'k-', 
                 alpha     = 0.3, 
                 linewidth = 1, 
                 label     = 'Average Trend',
                )
        
        ax2.set_xlabel('Day Index', fontsize = 12)
        ax2.set_ylabel('Average PHQ-9 Score', fontsize = 12)
        ax2.set_title('Daily Averages with Cluster Boundaries', fontsize = 14, pad = 15)
        ax2.set_ylim(0, 27)
        ax2.grid(True, alpha = 0.3)
        ax2.legend(bbox_to_anchor = (1.02, 1), loc = 'upper left', fontsize = 10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(fname       = save_path, 
                        dpi         = self.dpi, 
                        bbox_inches = 'tight',
                       )

            plt.close()

        else:
            plt.show()



# Convenience functions
def plot_scatter(data: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Convenience function for scatter plot
    """
    viz = VisualizationGenerator()
    viz.plot_scatter(data, save_path)


def plot_daily_averages(data: pd.DataFrame, save_path: Optional[Path] = None):
    """
    Convenience function for daily averages plot
    """
    viz = VisualizationGenerator()
    viz.plot_daily_averages(data, save_path)


def plot_clusters(data: pd.DataFrame, labels: np.ndarray, n_clusters: int, save_path: Optional[Path] = None):
    """
    Convenience function for cluster plot
    """
    viz = VisualizationGenerator()
    viz.plot_clusters(data, labels, n_clusters, save_path)
