# Dependencies
from dataclasses import field
from dataclasses import dataclass


@dataclass(frozen = True)
class EDAConstants:
    """
    Constants for PHQ-9 exploratory data analysis
    
    All magic numbers centralized here with clinical/statistical justification
    """
    # CLUSTERING PARAMETERS
    MIN_CLUSTERS                          : int   = 2       # Minimum clusters for optimization
    MAX_CLUSTERS_DEFAULT                  : int   = 20      # Default maximum clusters to test
    OPTIMAL_CLUSTER_RANGE_MIN             : int   = 3       # Reasonable minimum for PHQ-9 temporal data
    OPTIMAL_CLUSTER_RANGE_MAX             : int   = 8       # Reasonable maximum (avoids over-segmentation)
    
    # Clustering validation thresholds
    MIN_SILHOUETTE_SCORE_ACCEPTABLE       : float = 0.3     # Below this = poor clustering
    GOOD_SILHOUETTE_SCORE_THRESHOLD       : float = 0.5     # Above this = good clustering
    STRONG_SILHOUETTE_SCORE_THRESHOLD     : float = 0.7     # Above this = strong clustering
    
    # Cluster size constraints
    MIN_DAYS_PER_CLUSTER                  : int   = 5       # Minimum days for valid cluster
    MAX_CLUSTER_SIZE_IMBALANCE            : float = 0.8     # Max ratio of largest/smallest cluster
    
    # FEATURE EXTRACTION
    MINIMAL_SEVERITY_THRESHOLD            : float = 5.0     # 0-4: Minimal
    MILD_SEVERITY_THRESHOLD               : float = 10.0    # 5-9: Mild
    MODERATE_SEVERITY_THRESHOLD           : float = 15.0    # 10-14: Moderate
    MODERATELY_SEVERE_THRESHOLD           : float = 20.0    # 15-19: Moderately severe
    SEVERE_THRESHOLD                      : float = 20.0    # 20-27: Severe
    
    # Feature extraction parameters
    CV_CLIP_UPPER_BOUND                   : float = 5.0     # Prevent extreme CV values
    MIN_OBSERVATIONS_FOR_CV               : int   = 3       # Minimum observations to calculate CV
    
    # RESPONSE PATTERN CLASSIFICATION
    EARLY_RESPONSE_WEEKS                  : int   = 6       # Early responders plateau by week 6
    GRADUAL_RESPONSE_WEEKS                : int   = 12      # Gradual responders plateau by week 12
    LATE_RESPONSE_WEEKS                   : int   = 20      # Late responders plateau after week 12
    
    # Response detection thresholds
    RESPONSE_IMPROVEMENT_THRESHOLD        : float = 0.50    # ≥50% reduction = responder
    MINIMAL_IMPROVEMENT_THRESHOLD         : float = 0.20    # <20% = non-responder
    
    # Trajectory classification
    MIN_OBSERVATIONS_FOR_TRAJECTORY       : int   = 5       # Minimum observations to classify trajectory
    TRAJECTORY_SLOPE_FAST_THRESHOLD       : float = -0.08   # Steep improvement (early responder)
    TRAJECTORY_SLOPE_SLOW_THRESHOLD       : float = -0.04   # Slow improvement (late responder)
    TRAJECTORY_SLOPE_MINIMAL_THRESHOLD    : float = -0.02   # Minimal improvement (non-responder)
    
    # PLATEAU DETECTION
    PLATEAU_VARIANCE_THRESHOLD            : float = 2.0     # Max variance during plateau
    PLATEAU_WINDOW_WEEKS                  : int   = 4       # Rolling window for plateau detection
    PLATEAU_SLOPE_THRESHOLD               : float = 0.01    # Max absolute slope during plateau
    MIN_PLATEAU_DURATION_WEEKS            : int   = 3       # Minimum plateau duration
    
    # RELAPSE DETECTION
    RELAPSE_SCORE_INCREASE_THRESHOLD      : float = 5.0     # Minimum increase to flag relapse
    RELAPSE_DETECTION_MIN_GAP_DAYS        : int   = 7       # Minimum gap between observations
    RELAPSE_DETECTION_MAX_GAP_DAYS        : int   = 30      # Maximum gap (too far = unreliable)

    # TEMPORAL ANALYSIS
    # Boundary detection
    MIN_SEGMENT_LENGTH_DAYS               : int   = 7       # Minimum segment length
    BOUNDARY_SMOOTHING_WINDOW             : int   = 3       # Window for boundary smoothing
    
    # Trend analysis
    TREND_WINDOW_DAYS                     : int   = 30      # Window for local trend calculation
    SIGNIFICANT_TREND_SLOPE_THRESHOLD     : float = 0.05    # Minimum slope for significant trend
    
    # DISTRIBUTION COMPARISON: Scoring weights for dataset selection
    TEMPORAL_STABILITY_WEIGHT             : float = 0.35    # Weight for cluster stability
    CLINICAL_REALISM_WEIGHT               : float = 0.30    # Weight for severity distribution
    STATISTICAL_QUALITY_WEIGHT            : float = 0.20    # Weight for statistical properties
    METADATA_CONSISTENCY_WEIGHT           : float = 0.15    # Weight for metadata validation
    
    # Comparison thresholds
    CV_RANGE_EXPECTED_MIN                 : float = 0.10    # Minimum expected CV
    CV_RANGE_EXPECTED_MAX                 : float = 0.40    # Maximum expected CV
    OUTLIER_RATE_ACCEPTABLE               : float = 0.05    # Max 5% outliers acceptable
    
    # VISUALIZATION PARAMETERS
    DEFAULT_FIGURE_WIDTH                  : int   = 15      # Default figure width (inches)
    DEFAULT_FIGURE_HEIGHT                 : int   = 12      # Default figure height (inches)
    DEFAULT_DPI                           : int   = 300     # Default DPI for saved figures
    
    # Plot aesthetics
    SCATTER_ALPHA                         : float = 0.6     # Scatter plot transparency
    SCATTER_SIZE                          : int   = 30      # Scatter plot marker size
    LINE_WIDTH                            : int   = 2       # Line plot width
    BOUNDARY_LINE_WIDTH                   : float = 1.5     # Cluster boundary line width
    GRID_ALPHA                            : float = 0.3     # Grid transparency
    
    # Color mapping
    SEVERITY_COLOR_MINIMAL                : str   = 'green'
    SEVERITY_COLOR_MILD                   : str   = 'yellow'
    SEVERITY_COLOR_MODERATE               : str   = 'orange'
    SEVERITY_COLOR_SEVERE                 : str   = 'red'
    
    # DATA QUALITY THRESHOLDS
    MIN_DAYS_FOR_ANALYSIS                 : int   = 30      # Minimum days for meaningful analysis
    MIN_OBSERVATIONS_TOTAL                : int   = 100     # Minimum total observations
    MAX_MISSINGNESS_RATE                  : float = 0.98    # Maximum acceptable missingness
    
    # Outlier detection
    OUTLIER_Z_SCORE_THRESHOLD             : float = 3.0     # Z-score threshold for outliers
    OUTLIER_IQR_MULTIPLIER                : float = 1.5     # IQR multiplier for outlier detection
    
    # STATISTICAL TEST PARAMETERS
    DEFAULT_CONFIDENCE_LEVEL              : float = 0.95    # Default confidence level
    SIGNIFICANCE_ALPHA                    : float = 0.05    # Significance level for tests
    MIN_SAMPLE_SIZE_TTEST                 : int   = 30      # Minimum sample for t-test
    MIN_SAMPLE_SIZE_KS_TEST               : int   = 20      # Minimum sample for KS test


# Singleton instance
eda_constants_instance = EDAConstants()


# UTILITY FUNCTIONS
def get_severity_category_eda(score: float) -> str:
    """
    Classify PHQ-9 score into severity category (EDA-specific)
    
    Arguments:
    ----------
        score { float } : PHQ-9 score [0, 27]
    
    Returns:
    --------
            { str }     : Severity category name
    """
    if (score < eda_constants_instance.MINIMAL_SEVERITY_THRESHOLD):
        return 'minimal'
    
    elif (score < eda_constants_instance.MILD_SEVERITY_THRESHOLD):
        return 'mild'
    
    elif (score < eda_constants_instance.MODERATE_SEVERITY_THRESHOLD):
        return 'moderate'
    
    elif (score < eda_constants_instance.MODERATELY_SEVERE_THRESHOLD):
        return 'moderately_severe'
    
    else:
        return 'severe'


def get_severity_color(score: float) -> str:
    """
    Get color for severity level
    
    Arguments:
    ----------
        score { float } : PHQ-9 score
    
    Returns:
    --------
           { str }      : Color name
    """
    category  = get_severity_category_eda(score)
    
    color_map = {'minimal'           : eda_constants_instance.SEVERITY_COLOR_MINIMAL,
                 'mild'              : eda_constants_instance.SEVERITY_COLOR_MILD,
                 'moderate'          : eda_constants_instance.SEVERITY_COLOR_MODERATE,
                 'moderately_severe' : eda_constants_instance.SEVERITY_COLOR_MODERATE,
                 'severe'            : eda_constants_instance.SEVERITY_COLOR_SEVERE,
                }
    
    return color_map.get(category, 'gray')


def validate_cluster_quality(silhouette_score: float) -> str:
    """
    Interpret silhouette score quality
    
    Arguments:
    ----------
        silhouette_score { float } : Silhouette score [-1, 1]
    
    Returns:
    --------
                { str }            : Quality interpretation
    """
    if (silhouette_score >= eda_constants_instance.STRONG_SILHOUETTE_SCORE_THRESHOLD):
        return 'strong'
    
    elif (silhouette_score >= eda_constants_instance.GOOD_SILHOUETTE_SCORE_THRESHOLD):
        return 'good'
    
    elif (silhouette_score >= eda_constants_instance.MIN_SILHOUETTE_SCORE_ACCEPTABLE):
        return 'acceptable'
    
    else:
        return 'poor'


def get_response_pattern_from_slope(slope: float) -> str:
    """
    Classify response pattern based on trajectory slope
    
    Arguments:
    ----------
        slope { float } : Trajectory slope (points/day)
    
    Returns:
    --------
          { str }       : Response pattern classification
    """
    if (slope <= eda_constants_instance.TRAJECTORY_SLOPE_FAST_THRESHOLD):
        return 'early_responder'
    
    elif (slope <= eda_constants_instance.TRAJECTORY_SLOPE_SLOW_THRESHOLD):
        return 'gradual_responder'
    
    elif (slope <= eda_constants_instance.TRAJECTORY_SLOPE_MINIMAL_THRESHOLD):
        return 'late_responder'
    
    else:
        return 'non_responder'


# CONSTANTS SUMMARY
CONSTANTS_SUMMARY = {'Clustering'           : {'Min clusters'         : eda_constants_instance.MIN_CLUSTERS,
                                               'Max clusters default' : eda_constants_instance.MAX_CLUSTERS_DEFAULT,
                                               'Min silhouette'       : eda_constants_instance.MIN_SILHOUETTE_SCORE_ACCEPTABLE,
                                              },
                     'Response Patterns'    : {'Early weeks'    : eda_constants_instance.EARLY_RESPONSE_WEEKS,
                                               'Gradual weeks'  : eda_constants_instance.GRADUAL_RESPONSE_WEEKS,
                                               'Late weeks'     : eda_constants_instance.LATE_RESPONSE_WEEKS,
                                               'Response threshold' : f'{eda_constants_instance.RESPONSE_IMPROVEMENT_THRESHOLD:.0%}',
                                              },
                     'Distribution Scoring' : {'Temporal stability'    : f'{eda_constants_instance.TEMPORAL_STABILITY_WEIGHT:.0%}',
                                               'Clinical realism'      : f'{eda_constants_instance.CLINICAL_REALISM_WEIGHT:.0%}',
                                               'Statistical quality'   : f'{eda_constants_instance.STATISTICAL_QUALITY_WEIGHT:.0%}',
                                               'Metadata consistency'  : f'{eda_constants_instance.METADATA_CONSISTENCY_WEIGHT:.0%}',
                                              },
                     'Visualization'        : {'Default size'    : f'{eda_constants_instance.DEFAULT_FIGURE_WIDTH}x{eda_constants_instance.DEFAULT_FIGURE_HEIGHT}',
                                               'Default DPI'     : eda_constants_instance.DEFAULT_DPI,
                                               'Scatter alpha'   : eda_constants_instance.SCATTER_ALPHA,
                                              }
                    }