# Dependencies
import json
import numpy as np
import pandas as pd
from typing import Dict
from pathlib import Path
from typing import Tuple
from typing import Optional
from src.eda.validators import EDADataValidator
from src.utils.logging_util import setup_logger
from src.eda.clustering import ClusteringEngine
from src.eda.metadata_loader import MetadataLoader
from src.eda.clustering import  TemporalClustering
from src.utils.logging_util import log_section_header
from src.utils.logging_util import log_dataframe_info
from src.eda.clustering import OptimalClusterSelector
from src.eda.response_patterns import RelapseDetector
from config.eda_constants import eda_constants_instance
from src.eda.visualizations import VisualizationGenerator
from config.eda_constants import validate_cluster_quality
from src.eda.response_patterns import ResponsePatternAnalyzer


class PHQ9DataAnalyzer:
    """
    PHQ-9 exploratory data analysis, provides complete EDA pipeline with:
    - clustering
    - visualization
    - statistical analysis
    - metadata integration
    - response pattern analysis
    - distribution comparison
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
        self.metadata          = None
        
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
        
        self.validator         = EDADataValidator()
        
        self.logger.info("Initialized PHQ9DataAnalyzer")

    
    def load_data(self, data_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load PHQ-9 data from CSV and optional metadata sidecar
        
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
            # Load CSV
            self.data = pd.read_csv(data_path, index_col = 0)
            log_dataframe_info(self.logger, self.data, "PHQ-9 Data")
            
            # Load metadata if enabled
            if self.config.load_metadata:
                metadata_path = data_path.with_suffix('.metadata.json')
                
                if metadata_path.exists():
                    metadata_loader = MetadataLoader(logger = self.logger)
                    self.metadata   = metadata_loader.load(metadata_path)
                else:
                    self.logger.warning(f"Metadata file not found: {metadata_path}")
            
            # Validate
            if not self._validate_data():
                raise ValueError("Data validation failed")
            
            return self.data
            
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
    

    def _validate_data(self) -> bool:
        """
        Validate loaded data using EDADataValidator
        """
        if self.data is None:
            self.logger.error("No data loaded")
            return False
        
        # Run comprehensive validation
        validation = self.validator.validate_all(self.data, self.metadata)
        
        # Log results
        if validation['overall_valid']:
            self.logger.info("✅ Data validation passed")
        else:
            self.logger.warning("⚠️ Data validation completed with issues")
        
        for warning in validation['warnings']:
            self.logger.warning(f"  {warning}")
        
        for error in validation['errors']:
            self.logger.error(f"  {error}")
        
        # Store validation results
        self.validation_results = validation
        
        return validation['overall_valid']
    

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
        summary['missing_count'] = self.data.isna().sum(axis = 1)
        summary['missing_pct']   = (self.data.isna().sum(axis = 1) / self.data.shape[1]) * 100
        
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
            
            # Interpret silhouette quality
            quality = validate_cluster_quality(silhouette)
            self.logger.info(f"  Clustering quality: {quality}")
        
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
        Classify severity based on average score (using constants)
        """
        if (avg_score >= eda_constants_instance.SEVERE_THRESHOLD):
            return "Severe"

        elif (avg_score >= eda_constants_instance.MODERATELY_SEVERE_THRESHOLD):
            return "Moderately Severe"

        elif (avg_score >= eda_constants_instance.MODERATE_SEVERITY_THRESHOLD):
            return "Moderate"

        elif (avg_score >= eda_constants_instance.MILD_SEVERITY_THRESHOLD):
            return "Mild"

        else:
            return "Minimal"
    

    def analyze_response_patterns(self) -> pd.DataFrame:
        """
        Analyze response patterns across all patients
        
        Returns:
        --------
            { pd.DataFrame } : Response pattern analysis results
        """
        if not self.config.analyze_response_patterns:
            self.logger.info("Response pattern analysis disabled in config")
            return None
        
        log_section_header(self.logger, "ANALYZING RESPONSE PATTERNS")
        
        analyzer           = ResponsePatternAnalyzer(logger = self.logger)
        response_analysis  = analyzer.analyze_all_patients(self.data)
        
        # Log distribution
        pattern_counts = response_analysis['response_pattern'].value_counts()
        self.logger.info("Response pattern distribution:")
        for pattern, count in pattern_counts.items():
            pct = count / len(response_analysis) * 100
            self.logger.info(f"  {pattern}: {count} ({pct:.1f}%)")
        
        # Compare to metadata if available
        if (self.metadata is not None) and self.config.validate_against_metadata:
            self.logger.info("\nComparing to metadata expectations...")
            comparison = analyzer.compare_to_metadata(response_analysis, self.metadata)
            
            if comparison.get('comparison_available', False):
                self.logger.info(f"Overall match score: {comparison['overall_match_score']:.1f}%")
                
                for pattern, stats in comparison['patterns'].items():
                    self.logger.info(f"  {pattern}: expected {stats['expected_pct']:.1f}%, observed {stats['observed_pct']:.1f}% (Δ={stats['difference_pct']:.1f}%)")
        
        return response_analysis
    

    def detect_relapses(self) -> Dict:
        """
        Detect relapse events across all patients
        
        Returns:
        --------
            { dict } : Relapse detection results
        """
        if not self.config.detect_relapses:
            self.logger.info("Relapse detection disabled in config")
            return None
        
        log_section_header(self.logger, "DETECTING RELAPSE EVENTS")
        
        detector        = RelapseDetector(threshold = eda_constants_instance.RELAPSE_SCORE_INCREASE_THRESHOLD)
        relapse_results = detector.detect_relapses(self.data)
        
        # Log summary
        self.logger.info(f"Total relapses detected: {relapse_results['total_relapses']}")
        self.logger.info(f"Patients with relapses: {relapse_results['patients_with_relapses']} ({relapse_results['relapse_rate']:.1%})")
        self.logger.info(f"Mean relapses per patient: {relapse_results['mean_relapses_per_patient']:.2f}")
        
        return relapse_results
    

    def generate_visualizations(self, labels: Optional[np.ndarray] = None, n_clusters: Optional[int] = None, 
                               response_analysis: Optional[pd.DataFrame] = None, relapse_results: Optional[Dict] = None):
        """
        Generate all visualizations
        
        Arguments:
        ----------
            labels            { np.ndarray }  : Cluster labels

            n_clusters        { int }         : Number of clusters
            
            response_analysis { pd.DataFrame }: Response pattern analysis
            
            relapse_results   { dict }        : Relapse detection results
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
        
        # Response pattern plots
        if response_analysis is not None:
            self.logger.info("Creating response pattern plots...")
            
            self.visualizer.plot_response_patterns(response_analysis,
                                                   save_path = viz_dir / "response_patterns.png",
                                                  )
            
            self.visualizer.plot_patient_trajectories_by_pattern(self.data,
                                                                  response_analysis,
                                                                  save_path = viz_dir / "patient_trajectories_by_pattern.png",
                                                                 )
        
        # Relapse plots
        if relapse_results is not None and relapse_results.get('total_relapses', 0) > 0:
            self.logger.info("Creating relapse event plots...")
            
            self.visualizer.plot_relapse_events(relapse_results,
                                                save_path = viz_dir / "relapse_events.png",
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
        
        # Response pattern analysis
        response_analysis                  = self.analyze_response_patterns()
        
        # Relapse detection
        relapse_results                    = self.detect_relapses()
        
        # Generate visualizations
        self.generate_visualizations(labels           = labels, 
                                     n_clusters       = optimal_k,
                                     response_analysis = response_analysis,
                                     relapse_results  = relapse_results,
                                    )
        
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
                                              'response_analysis'  : response_analysis,
                                              'relapse_results'    : relapse_results,
                                              'metadata'           : self.metadata,
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
        
        # Save cluster labels
        labels_df = pd.DataFrame({'day_index'  : range(len(results['labels'])),
                                  'cluster_id' : results['labels'],
                                })

        labels_df.to_csv(path_or_buf = output_dir / "cluster_labels.csv", 
                         index       = False,
                        )
        
        # Save response pattern analysis
        if results['response_analysis'] is not None:
            results['response_analysis'].to_csv(path_or_buf = output_dir / "response_pattern_analysis.csv",
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
        
        # Add response pattern summary
        if results['response_analysis'] is not None:
            pattern_counts = results['response_analysis']['response_pattern'].value_counts().to_dict()
            json_results['response_pattern_counts'] = {k: int(v) for k, v in pattern_counts.items()}
        
        # Add relapse summary
        if results['relapse_results'] is not None:
            json_results['relapse_summary'] = {'total_relapses'         : results['relapse_results']['total_relapses'],
                                               'patients_with_relapses' : results['relapse_results']['patients_with_relapses'],
                                               'relapse_rate'           : results['relapse_results']['relapse_rate'],
                                              }
        
        with open(output_dir / "analysis_summary.json", 'w') as f:
            json.dump(obj    = json_results, 
                      fp     = f, 
                      indent = 4,
                     )
        
        self.logger.info(f"Results saved to: {output_dir}")