# Dependencies
import json
import numpy as np
import pandas as pd
from typing import Dict
from pathlib import Path
from typing import Tuple
from typing import Optional
from phq9_analysis.utils.logging_util import setup_logger
from phq9_analysis.eda.clustering import ClusteringEngine
from phq9_analysis.utils.logging_util import log_parameters
from phq9_analysis.eda.clustering import  TemporalClustering
from phq9_analysis.utils.logging_util import log_section_header
from phq9_analysis.utils.logging_util import log_dataframe_info
from phq9_analysis.eda.clustering import OptimalClusterSelector
from phq9_analysis.eda.visualizations import VisualizationGenerator


class PHQ9DataAnalyzer:
    """
    PHQ-9 exploratory data analysis, provides complete EDA pipeline with:
    - clustering
    - visualization
    - statistical analysis
    """
    def __init__(self, config):
        """
        Initialize analyzer with configuration
        
        Arguments:
        ----------
            config { EDAConfig } : EDAConfig instance
        """
        self.config            = config
        self.data              = None
        
        # Set up logging
        self.logger            = setup_logger(module_name = 'eda',
                                              log_level   = 'INFO',
                                              log_dir     = Path('logs'),
                                             )
        
        # Initialize components
        self.clustering_engine = ClusteringEngine(imputation_method = config.imputation_method,
                                                  standardize       = config.standardize_features,
                                                  random_seed       = config.clustering_random_seed,
                                                 )
        
        self.visualizer        = VisualizationGenerator(figsize       = config.figure_size,
                                                        dpi           = config.dpi,
                                                        style         = config.plot_style,
                                                        color_palette = config.color_palette,
                                                       )
        
        self.logger.info("Initialized PHQ9DataAnalyzer")

    
    def load_data(self, data_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load PHQ-9 data from CSV
        
        Arguments:
        ----------
            data_path { Path }  : Path to data file (uses config if None)
        
        Returns:
        --------
            { pd.DataFrame }    : Loaded DataFrame
        """
        if data_path is None:
            data_path = self.config.data_file_path
        
        log_section_header(self.logger, "LOADING DATA")
        
        try:
            self.data = pd.read_csv(data_path, index_col=0)
            log_dataframe_info(self.logger, self.data, "PHQ-9 Data")
            
            # Validate
            if not self._validate_data():
                raise ValueError("Data validation failed")
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    

    def _validate_data(self) -> bool:
        """
        Validate loaded data
        """
        if self.data is None:
            self.logger.error("No data loaded")
            return False
        
        # Check score range
        all_scores = self.data.values.flatten()
        all_scores = all_scores[~np.isnan(all_scores)]
        
        if (len(all_scores) == 0):
            self.logger.error("No valid scores in data")
            return False
        
        if (not np.all((all_scores >= 0) & (all_scores <= 27))):
            self.logger.error("Scores outside valid PHQ-9 range [0, 27]")
            return False
        
        self.logger.info("Data validation passed")
        
        return True
    

    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Calculate summary statistics
        
        Returns:
        --------
            { pd.DataFrame }    : DataFrame with statistics
        """
        log_section_header(self.logger, "CALCULATING SUMMARY STATISTICS")
        
        summary                  = self.data.T.describe().T
        
        # Add missing info
        summary['missing_count'] = self.data.isna().sum()
        summary['missing_pct']   = (self.data.isna().sum() / len(self.data)) * 100
        
        self.logger.info(f"Summary statistics calculated for {len(summary)} days")
        
        return summary
    

    def find_optimal_clusters(self, max_clusters: Optional[int] = None) -> Tuple[int, int, Dict]:
        """
        Find optimal number of clusters using multiple methods
        
        Arguments:
        ----------
            max_clusters { int } : Maximum clusters to test (uses config if None)
        
        Returns:
        --------
               { tuple }         : A python tuple containing:
                                   - elbow_k
                                   - silhouette_k
                                   - results_dict
        """
        if max_clusters is None:
            max_clusters = self.config.max_clusters_to_test
        
        log_section_header(self.logger, "FINDING OPTIMAL CLUSTERS")
        
        selector                  = OptimalClusterSelector(self.clustering_engine)
        
        # Elbow method
        self.logger.info("Running elbow method...")
        
        elbow_k, inertias         = selector.elbow_method(self.data, max_clusters)
        self.logger.info(f"  Elbow method suggests: {elbow_k} clusters")
        
        # Silhouette method
        self.logger.info("Running silhouette analysis...")
        
        silhouette_k, silhouettes = selector.silhouette_method(self.data, max_clusters)
        self.logger.info(f"  Silhouette method suggests: {silhouette_k} clusters")
        
        # Visualize
        output_path               = self.config.get_clustering_output_dir() / "cluster_optimization.png"
        output_path.parent.mkdir(parents = True, exist_ok = True)
        
        self.visualizer.plot_cluster_optimization(inertias     = inertias,
                                                  silhouettes  = silhouettes,
                                                  elbow_k      = elbow_k,
                                                  silhouette_k = silhouette_k,
                                                  save_path    = output_path,
                                                 )
        
        self.logger.info(f"Optimization plot saved: {output_path}")
        
        results                   = {'elbow_k'      : elbow_k,
                                     'silhouette_k' : silhouette_k,
                                     'inertias'     : inertias,
                                     'silhouettes'  : silhouettes,
                                    }
        
        return elbow_k, silhouette_k, results
    

    def fit_clustering(self, n_clusters: int, use_temporal: Optional[bool] = None) -> np.ndarray:
        """
        Fit clustering model
        
        Arguments:
        ----------
            n_clusters   { int }  : Number of clusters

            use_temporal { bool } : Use temporal clustering (uses config if None)
        
        Returns:
        --------
              { np.ndarray }      : Cluster labels
        """
        if use_temporal is None:
            use_temporal = self.config.use_temporal_clustering
        
        log_section_header(self.logger, f"FITTING {n_clusters} CLUSTERS")
        
        if use_temporal:
            self.logger.info("Using temporal-aware clustering...")

            temporal = TemporalClustering(temporal_weight = self.config.temporal_weight,
                                          random_seed     = self.config.clustering_random_seed,
                                         )

            labels   = temporal.fit(self.data,
                                    n_clusters,
                                    imputation_method = self.config.imputation_method,
                                   )
        else:
            self.logger.info("Using standard KMeans clustering...")

            labels, inertia, silhouette = self.clustering_engine.fit_kmeans(self.data,
                                                                            n_clusters
                                                                           )
            self.logger.info(f"  Inertia: {inertia:.2f}")
            self.logger.info(f"  Silhouette: {silhouette:.3f}")
        
        # Log cluster sizes
        unique, counts = np.unique(labels, return_counts = True)

        self.logger.info(f"Cluster sizes:")
        
        for cluster_id, count in zip(unique, counts):
            self.logger.info(f"  Cluster {cluster_id}: {count} days")
        
        return labels
    

    def analyze_clusters(self, labels: np.ndarray) -> pd.DataFrame:
        """
        Analyze cluster characteristics.

        Each cluster summarizes:
        - Average severity
        - Variability
        - Observation density
        - Clinical severity band
        """
        log_section_header(self.logger, "ANALYZING CLUSTER CHARACTERISTICS")

        daily_avg       = self.data.mean(axis = 1, skipna = True)
        daily_std       = self.data.std(axis = 1, skipna = True)

        cluster_stats   = list()
        unique_clusters = np.unique(labels)

        for cluster_id in unique_clusters:
            cluster_mask   = (labels == cluster_id)
            cluster_days   = np.where(cluster_mask)[0]
            cluster_scores = daily_avg.iloc[cluster_days]

            stats          = {'cluster_id'        : int(cluster_id),
                              'n_days'            : len(cluster_days),
                              'avg_score'         : float(cluster_scores.mean()),
                              'std_score_cluster' : float(cluster_scores.std()),
                              'std_score_daily'   : float(daily_std.iloc[cluster_days].mean()),
                              'n_obs_avg'         : float(self.data.notna().sum(axis=1).iloc[cluster_days].mean()),
                              'min_score'         : float(cluster_scores.min()),
                              'max_score'         : float(cluster_scores.max()),
                              'day_range'         : f"{cluster_days.min()}-{cluster_days.max()}",
                              'severity'          : self._classify_severity(cluster_scores.mean()),
                             }

            cluster_stats.append(stats)

            self.logger.info(f"Cluster {cluster_id}: {stats['n_days']} days | avg={stats['avg_score']:.2f} | {stats['severity']}")

        return pd.DataFrame(cluster_stats)
    

    def _classify_severity(self, avg_score: float) -> str:
        """
        Classify severity based on average score
        """
        if (avg_score >= 20):
            return "Severe"

        elif (avg_score >= 15):
            return "Moderately Severe"

        elif (avg_score >= 10):
            return "Moderate"

        elif (avg_score >= 5):
            return "Mild"

        else:
            return "Minimal"
    

    def generate_visualizations(self, labels: Optional[np.ndarray] = None, n_clusters: Optional[int] = None):
        """
        Generate all visualizations
        
        Arguments:
        ----------
            labels     { np.ndarray } : Cluster labels

            n_clusters    { int }     : Number of clusters
        """
        log_section_header(self.logger, "GENERATING VISUALIZATIONS")
        
        viz_dir = self.config.get_visualization_output_dir()
        viz_dir.mkdir(parents  = True, 
                      exist_ok = True,
                     )
        
        # Scatter plot
        self.logger.info("Creating scatter plot...")

        self.visualizer.plot_scatter(self.data,
                                     save_path = viz_dir / "scatter_plot.png",
                                    )
        
        # Daily averages
        self.logger.info("Creating daily averages plot...")

        self.visualizer.plot_daily_averages(self.data,
                                            save_path = viz_dir / "daily_averages.png",
                                           )
        
        # Cluster plot if labels provided
        if labels is not None and n_clusters is not None:
            self.logger.info("Creating cluster plot...")

            self.visualizer.plot_clusters(self.data,
                                          labels,
                                          n_clusters,
                                          save_path = viz_dir / "cluster_results.png",
                                         )
        
        self.logger.info(f"Visualizations saved to: {viz_dir}")
    

    def run_full_analysis(self) -> Dict:
        """
        Run complete EDA pipeline
        
        Returns:
        --------
            { dict }    : Dictionary with all results
        """
        log_section_header(self.logger, "RUNNING FULL EDA ANALYSIS")
        
        # Summary statistics
        summary_stats                      = self.get_summary_statistics()
        
        # Find optimal clusters
        elbow_k, silhouette_k, opt_results = self.find_optimal_clusters()
        
        # Use silhouette recommendation
        optimal_k                          = silhouette_k

        self.logger.info(f"\nUsing K={optimal_k} for final clustering")
        
        # Fit clustering
        labels                             = self.fit_clustering(optimal_k)
        
        # Analyze clusters
        cluster_analysis                   = self.analyze_clusters(labels)
        
        # Generate visualizations
        self.generate_visualizations(labels, optimal_k)
        
        # Compile results
        results                            = {'data_shape'         : self.data.shape,
                                              'total_observations' : int(self.data.notna().sum().sum()),
                                              'missing_pct'        : float(self.data.isna().sum().sum() / self.data.size * 100),
                                              'elbow_k'            : elbow_k,
                                              'silhouette_k'       : silhouette_k,
                                              'optimal_k'          : optimal_k,
                                              'summary_statistics' : summary_stats,
                                              'cluster_analysis'   : cluster_analysis,
                                              'labels'             : labels.tolist(),
                                             }
        
        # Save results
        self._save_results(results)
        
        log_section_header(self.logger, "EDA ANALYSIS COMPLETE")
        
        return results
    

    def _save_results(self, results: Dict):
        """
        Save analysis results
        """
        output_dir = self.config.results_base_directory
        output_dir.mkdir(parents  = True, 
                         exist_ok = True,
                        )
        
        # Save summary stats
        results['summary_statistics'].to_csv(path_or_buf = output_dir / "summary_statistics.csv")
        
        # Save cluster analysis
        results['cluster_analysis'].to_csv(path_or_buf = output_dir / "cluster_characteristics.csv", 
                                           index       = False,
                                          )
        
        # Save JSON summary
        json_results = {'data_shape'         : [int(results['data_shape'][0]), int(results['data_shape'][1])],
                        'total_observations' : int(results['total_observations']),
                        'missing_pct'        : float(results['missing_pct']),
                        'elbow_k'            : int(results['elbow_k']),
                        'silhouette_k'       : int(results['silhouette_k']),
                        'optimal_k'          : int(results['optimal_k']),
                        'labels'             : [int(x) for x in results['labels']],
                       }
        
        with open(output_dir / "analysis_summary.json", 'w') as f:
            json.dump(obj    = json_results, 
                      fp     = f, 
                      indent = 4,
                     )
        
        self.logger.info(f"Results saved to: {output_dir}")