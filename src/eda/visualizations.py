# Dependencies
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List
from typing import Dict
from typing import Tuple
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
from config.eda_constants import eda_constants_instance


class VisualizationGenerator:
    """
    Central class for generating all EDA visualizations
    """
    def __init__(self, figsize: Tuple[int, int] = None, dpi: int = None, style: str = 'seaborn', color_palette: str = 'husl'):
        """
        Initialize visualization generator
        
        Arguments:
        ----------
            figsize       { tuple } : Default figure size

            dpi            { int }  : DPI for saved figures
            
            style          { str }  : Matplotlib style
            
            color_palette  { str }  : Seaborn color palette
        """
        self.figsize = figsize or (eda_constants_instance.DEFAULT_FIGURE_WIDTH, eda_constants_instance.DEFAULT_FIGURE_HEIGHT)
        self.dpi     = dpi or eda_constants_instance.DEFAULT_DPI
        
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
                           s     = eda_constants_instance.SCATTER_SIZE,
                           alpha = eda_constants_instance.SCATTER_ALPHA,
                          )
        
        # Formatting
        ax.set_xlabel('Day Index', fontsize = 12)
        ax.set_ylabel('PHQ-9 Score', fontsize = 12)
        ax.set_title('PHQ-9 Score Distribution Across All Days', fontsize = 16, pad = 20)
        ax.set_ylim(0, 27)
        ax.grid(True, alpha = eda_constants_instance.GRID_ALPHA)
        
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
                linewidth  = eda_constants_instance.LINE_WIDTH, 
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
                linewidth = eda_constants_instance.LINE_WIDTH,
                alpha     = 0.8, 
                label     = f'Trend (slope: {z[0]:.3f})',
               )
        
        # Add severity level lines
        severity_levels = {'Minimal'  : eda_constants_instance.MINIMAL_SEVERITY_THRESHOLD,
                           'Mild'     : eda_constants_instance.MILD_SEVERITY_THRESHOLD,
                           'Moderate' : eda_constants_instance.MODERATE_SEVERITY_THRESHOLD,
                           'Severe'   : eda_constants_instance.SEVERE_THRESHOLD,
                          }

        colors          = [eda_constants_instance.SEVERITY_COLOR_MINIMAL,
                           eda_constants_instance.SEVERITY_COLOR_MILD,
                           eda_constants_instance.SEVERITY_COLOR_MODERATE,
                           eda_constants_instance.SEVERITY_COLOR_SEVERE,
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
        ax.grid(True, alpha = eda_constants_instance.GRID_ALPHA)
        ax.legend(bbox_to_anchor = (1.02, 1), 
                  loc            = 'upper left', 
                  fontsize       = 10,
                 )
        
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
        
        k_values        = np.arange(eda_constants_instance.MIN_CLUSTERS, len(inertias) + eda_constants_instance.MIN_CLUSTERS)
        
        # Elbow plot
        ax1.plot(k_values, 
                 inertias, 
                 'bo-', 
                 linewidth  = eda_constants_instance.LINE_WIDTH, 
                 markersize = 8)

        ax1.axvline(x         = elbow_k, 
                    color     = 'red', 
                    linestyle = '--', 
                    linewidth = eda_constants_instance.LINE_WIDTH, 
                    alpha     = 0.7, 
                    label     = f'Elbow: K={elbow_k}',
                   )

        ax1.set_xlabel('Number of Clusters (K)', fontsize = 12)
        ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize = 12)
        ax1.set_title('Elbow Method for Optimal K', fontsize = 14, pad = 15)
        ax1.grid(True, alpha = eda_constants_instance.GRID_ALPHA)
        ax1.legend(fontsize = 11)
        
        # Silhouette plot
        ax2.plot(k_values, 
                 silhouettes, 
                 'go-', 
                 linewidth  = eda_constants_instance.LINE_WIDTH, 
                 markersize = 8)

        ax2.axvline(x         = silhouette_k, 
                    color     = 'red', 
                    linestyle = '--',
                    linewidth = eda_constants_instance.LINE_WIDTH, 
                    alpha     = 0.7, 
                    label     = f'Optimal: K={silhouette_k}',
                   )

        ax2.set_xlabel('Number of Clusters (K)', fontsize = 12)
        ax2.set_ylabel('Silhouette Score', fontsize = 12)
        ax2.set_title('Silhouette Analysis for Optimal K', fontsize = 14, pad = 15)
        ax2.grid(True, alpha = eda_constants_instance.GRID_ALPHA)
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
        
        # Scatter plot with cluster colors
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
        ax1.grid(True, alpha = eda_constants_instance.GRID_ALPHA)
        ax1.legend(bbox_to_anchor = (1.02, 1), loc = 'upper left', fontsize = 10)
        
        # Daily averages with cluster boundaries
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
        
        # Draw boundaries at cluster transitions
        cluster_boundaries = list()

        for i in range(len(labels) - 1):
            if (labels[i] != labels[i + 1]):
                # Check if this is a valid boundary (not isolated flip)
                is_valid_boundary = False
                
                # Edge cases: first and last transitions
                if ((i == 0) or (i == len(labels) - 2)):
                    is_valid_boundary = True
                
                # Middle transitions
                else:
                    left_consistent   = (labels[i - 1] == labels[i])
                    right_consistent  = (labels[i + 1] == labels[i + 2])
                    is_valid_boundary = (left_consistent or right_consistent)
                
                if is_valid_boundary:
                    cluster_boundaries.append(i + 0.5)
        
        # Draw filtered boundaries
        for boundary in cluster_boundaries:
            ax2.axvline(x         = boundary, 
                        linestyle = '--', 
                        color     = 'gray', 
                        alpha     = 0.6,
                        linewidth = eda_constants_instance.BOUNDARY_LINE_WIDTH,
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
        ax2.grid(True, alpha = eda_constants_instance.GRID_ALPHA)
        ax2.legend(bbox_to_anchor = (1.02, 1), 
                   loc            = 'upper left', 
                   fontsize       = 10,
                  )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(fname       = save_path, 
                        dpi         = self.dpi, 
                        bbox_inches = 'tight',
                       )

            plt.close()

        else:
            plt.show()
    

    def plot_response_patterns(self, response_analysis: pd.DataFrame, save_path: Optional[Path] = None):
        """
        Plot response pattern distribution
        
        Arguments:
        ----------
            response_analysis { pd.DataFrame } : Response pattern analysis results
            
            save_path             { Path }     : Path to save figure
        """
        fig, (ax1, ax2) = plt.subplots(nrows   = 1, 
                                       ncols   = 2, 
                                       figsize = (16, 6),
                                      )
        
        # Pattern distribution (bar chart)
        pattern_counts  = response_analysis['response_pattern'].value_counts()
        
        colors_map      = {'early_responder'   : 'green',
                           'gradual_responder' : 'blue',
                           'late_responder'    : 'orange',
                           'non_responder'     : 'red',
                          }
        
        colors          = [colors_map.get(p, 'gray') for p in pattern_counts.index]
        
        ax1.bar(range(len(pattern_counts)), 
                pattern_counts.values, 
                color = colors, 
                alpha = 0.7)
        
        ax1.set_xticks(range(len(pattern_counts)))
        ax1.set_xticklabels(pattern_counts.index, rotation = 45, ha = 'right')
        ax1.set_ylabel('Number of Patients', fontsize = 12)
        ax1.set_title('Response Pattern Distribution', fontsize = 14, pad = 15)
        ax1.grid(True, alpha = eda_constants_instance.GRID_ALPHA, axis = 'y')
        
        # Add count labels
        for i, count in enumerate(pattern_counts.values):
            ax1.text(i, count, str(count), ha = 'center', va = 'bottom')
        
        # Improvement distribution by pattern
        for pattern in pattern_counts.index:
            pattern_data = response_analysis[response_analysis['response_pattern'] == pattern]
            improvements = pattern_data['improvement_pct'] * 100
            
            ax2.hist(improvements, 
                     bins  = 20, 
                     alpha = 0.5, 
                     label = pattern,
                     color = colors_map.get(pattern, 'gray'))
        
        ax2.axvline(x         = eda_constants_instance.RESPONSE_IMPROVEMENT_THRESHOLD * 100, 
                    color     = 'black', 
                    linestyle = '--', 
                    linewidth = eda_constants_instance.LINE_WIDTH,
                    label     = f'Response Threshold ({eda_constants_instance.RESPONSE_IMPROVEMENT_THRESHOLD*100:.0f}%)')
        
        ax2.set_xlabel('Improvement (%)', fontsize = 12)
        ax2.set_ylabel('Number of Patients', fontsize = 12)
        ax2.set_title('Improvement Distribution by Response Pattern', fontsize = 14, pad = 15)
        ax2.legend(fontsize = 10)
        ax2.grid(True, alpha = eda_constants_instance.GRID_ALPHA, axis = 'y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(fname       = save_path, 
                        dpi         = self.dpi, 
                        bbox_inches = 'tight',
                       )

            plt.close()

        else:
            plt.show()
    

    def plot_patient_trajectories_by_pattern(self, data: pd.DataFrame, response_analysis: pd.DataFrame, save_path: Optional[Path] = None, max_patients_per_pattern: int = 5):
        """
        Plot sample patient trajectories grouped by response pattern
        
        Arguments:
        ----------
            data                     { pd.DataFrame } : PHQ-9 data
            
            response_analysis        { pd.DataFrame } : Response pattern analysis
            
            save_path                     { Path }    : Path to save figure
            
            max_patients_per_pattern      { int }     : Max patients to plot per pattern
        """
        patterns   = response_analysis['response_pattern'].unique()
        n_patterns = len(patterns)
        
        fig, axes  = plt.subplots(nrows   = 2, 
                                  ncols   = 2, 
                                  figsize = (16, 12),
                                 )
        
        axes       = axes.flatten()
        
        colors_map = {'early_responder'   : 'green',
                      'gradual_responder' : 'blue',
                      'late_responder'    : 'orange',
                      'non_responder'     : 'red',
                     }
        
        for idx, pattern in enumerate(patterns):
            if (idx >= len(axes)):
                break
            
            ax               = axes[idx]
            
            # Get patients with this pattern
            pattern_patients = response_analysis[response_analysis['response_pattern'] == pattern]['patient_id'].values
            
            # Sample patients
            sample_patients = pattern_patients[:max_patients_per_pattern]
            
            for patient_id in sample_patients:
                patient_col = f'Patient_{patient_id}'
                
                if patient_col in data.columns:
                    patient_scores = data[patient_col].dropna()
                    days           = [int(idx.split('_')[1]) for idx in patient_scores.index]
                    
                    ax.plot(days, 
                            patient_scores.values, 
                            'o-', 
                            alpha      = 0.6,
                            linewidth  = 1.5,
                            markersize = 4)
            
            ax.set_xlabel('Day', fontsize = 11)
            ax.set_ylabel('PHQ-9 Score', fontsize = 11)

            ax.set_title(f'{pattern.replace("_", " ").title()} (n={len(pattern_patients)})', 
                         fontsize = 13, 
                         color    = colors_map.get(pattern, 'black'),
                        )

            ax.set_ylim(0, 27)
            ax.grid(True, alpha = eda_constants_instance.GRID_ALPHA)
            
            # Add severity thresholds
            ax.axhline(y         = eda_constants_instance.MODERATE_SEVERITY_THRESHOLD, 
                       color     = 'gray', 
                       linestyle = ':', 
                       alpha     = 0.5,
                      )

            ax.axhline(y         = eda_constants_instance.SEVERE_THRESHOLD, 
                       color     = 'gray', 
                       linestyle = ':', 
                       alpha     = 0.5,
                      )
        
        # Hide unused subplots
        for idx in range(n_patterns, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Sample Patient Trajectories by Response Pattern', fontsize = 16, y = 0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(fname       = save_path, 
                        dpi         = self.dpi, 
                        bbox_inches = 'tight',
                       )

            plt.close()

        else:
            plt.show()
    

    def plot_relapse_events(self, relapse_results: Dict, save_path: Optional[Path] = None):
        """
        Plot relapse event distribution
        
        Arguments:
        ----------
            relapse_results { dict } : Relapse detection results
            
            save_path       { Path } : Path to save figure
        """
        relapse_details = relapse_results.get('relapse_details', [])
        
        if not relapse_details:
            return
        
        fig, (ax1, ax2) = plt.subplots(nrows   = 1, 
                                       ncols   = 2, 
                                       figsize = (16, 6),
                                      )
        
        # Relapse timing distribution
        relapse_days = [r['day'] for r in relapse_details]
        
        ax1.hist(relapse_days, 
                 bins      = 30, 
                 color     = 'red', 
                 alpha     = 0.7, 
                 edgecolor = 'black',
                )

        ax1.set_xlabel('Day', fontsize = 12)
        ax1.set_ylabel('Number of Relapses', fontsize = 12)
        ax1.set_title('Relapse Event Distribution Over Time', fontsize = 14, pad = 15)
        ax1.grid(True, alpha = eda_constants_instance.GRID_ALPHA, axis = 'y')
        
        # Relapse magnitude distribution
        relapse_magnitudes = [r['magnitude'] for r in relapse_details]
        
        ax2.hist(relapse_magnitudes, 
                 bins      = 20, 
                 color     = 'orange', 
                 alpha     = 0.7, 
                 edgecolor = 'black',
                )

        ax2.axvline(x         = eda_constants_instance.RELAPSE_SCORE_INCREASE_THRESHOLD, 
                    color     = 'black', 
                    linestyle = '--', 
                    linewidth = eda_constants_instance.LINE_WIDTH,
                    label     = f'Detection Threshold ({eda_constants_instance.RELAPSE_SCORE_INCREASE_THRESHOLD} points)',
                   )

        ax2.set_xlabel('Score Increase (points)', fontsize = 12)
        ax2.set_ylabel('Number of Relapses', fontsize = 12)
        ax2.set_title('Relapse Magnitude Distribution', fontsize = 14, pad = 15)
        ax2.legend()
        ax2.grid(True, alpha = eda_constants_instance.GRID_ALPHA, axis = 'y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(fname       = save_path, 
                        dpi         = self.dpi, 
                        bbox_inches = 'tight',
                       )

            plt.close()

        else:
            plt.show()
    

    def plot_distribution_comparison(self, comparison_df: pd.DataFrame, save_path: Optional[Path] = None):
        """
        Plot comparison of multiple datasets
        
        Arguments:
        ----------
            comparison_df { pd.DataFrame } : Comparison results
            
            save_path         { Path }     : Path to save figure
        """
        fig, axes = plt.subplots(nrows   = 2, 
                                 ncols   = 2, 
                                 figsize = (16, 12),
                                )
        
        axes      = axes.flatten()
        
        metrics   = ['temporal_stability', 
                     'clinical_realism', 
                     'statistical_quality', 
                     'metadata_consistency',
                    ]

        titles    = ['Temporal Stability', 
                     'Clinical Realism', 
                     'Statistical Quality', 
                     'Metadata Consistency',
                    ]
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax    = axes[idx]
            
            # Bar plot
            colors = ['green' if i == 0 else 'steelblue' for i in range(len(comparison_df))]
            
            bars   = ax.bar(range(len(comparison_df)), 
                            comparison_df[metric], 
                            color = colors, 
                            alpha = 0.7,
                           )
            
            ax.set_xticks(range(len(comparison_df)))
            ax.set_xticklabels(comparison_df['dataset_name'], rotation = 45, ha = 'right')
            ax.set_ylabel('Score', fontsize = 12)
            ax.set_title(title, fontsize = 14, pad = 15)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha = eda_constants_instance.GRID_ALPHA, axis = 'y')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{height:.1f}',
                        ha       = 'center', 
                        va       = 'bottom', 
                        fontsize = 10,
                       )
        
        plt.suptitle('Dataset Quality Comparison', fontsize = 16, y = 0.995)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(fname       = save_path, 
                        dpi         = self.dpi, 
                        bbox_inches = 'tight',
                       )

            plt.close()

        else:
            plt.show()
    

    def plot_composite_scores(self, comparison_df: pd.DataFrame, save_path: Optional[Path] = None):
        """
        Plot composite scores and ranking
        
        Arguments:
        ----------
            comparison_df { pd.DataFrame } : Comparison results
            
            save_path         { Path }     : Path to save figure
        """
        fig, ax       = plt.subplots(figsize = (12, 6))
        
        # Ensure sorted by composite score
        comparison_df = comparison_df.sort_values(by        = 'composite_score', 
                                                  ascending = False,
                                                 )
        
        # Color code: best = green, others = blue
        colors        = ['green'] + ['steelblue'] * (len(comparison_df) - 1)
        
        bars          = ax.barh(range(len(comparison_df)), 
                                comparison_df['composite_score'], 
                                color = colors, 
                                alpha = 0.7,
                               )
        
        ax.set_yticks(range(len(comparison_df)))
        ax.set_yticklabels([f"Rank {row['rank']}: {row['dataset_name']}" for _, row in comparison_df.iterrows()])
        ax.set_xlabel('Composite Score', fontsize = 12)
        ax.set_title('Dataset Ranking by Composite Score', fontsize = 14, pad = 15)
        ax.set_xlim(0, 100)
        ax.grid(True, alpha = eda_constants_instance.GRID_ALPHA, axis = 'x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, comparison_df['composite_score'])):
            width = bar.get_width()

            ax.text(width, bar.get_y() + bar.get_height() / 2.,
                    f'{score:.2f}',
                    ha         = 'left', 
                    va         = 'center', 
                    fontsize   = 10, 
                    fontweight = 'bold',
                   )
        
        # Add annotation
        ax.text(0.02, 0.98, 
                'Recommended dataset highlighted in green', 
                transform         = ax.transAxes,
                verticalalignment = 'top',
                fontsize          = 11,
                bbox              = dict(boxstyle = 'round', facecolor = 'wheat', alpha = 0.5),
               )
        
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