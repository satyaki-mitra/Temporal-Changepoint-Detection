# Dependencies
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict
from typing import List
from pathlib import Path
from config.eda_constants import eda_constants_instance


class DistributionComparator:
    """
    Compare EDA results across multiple datasets (3 relapse distributions): provides dataset ranking for selection in change-point detection pipeline
    """
    def __init__(self, logger = None):
        """
        Initialize comparator
        
        Arguments:
        ----------
            logger { Logger } : Optional logger instance
        """
        self.logger = logger
    

    def compare_datasets(self, eda_results_list: List[Dict]) -> pd.DataFrame:
        """
        Compare multiple datasets and rank them
        
        Arguments:
        ----------
            eda_results_list { list } : List of EDA result dictionaries
                                        Each dict should contain:
                                        - 'dataset_name'
                                        - 'data' (DataFrame)
                                        - 'labels' (cluster labels)
                                        - 'metadata'
                                        - other EDA results
        
        Returns:
        --------
              { pd.DataFrame }        : Comparison table with rankings
        """
        comparison_results = list()
        
        for eda_result in eda_results_list:
            dataset_name = eda_result['dataset_name']
            data         = eda_result['data']
            labels       = eda_result.get('labels')
            metadata     = eda_result.get('metadata')

            # Calculate scores
            scores                 = self._calculate_dataset_scores(data, labels, metadata)
            scores['dataset_name'] = dataset_name
        
            comparison_results.append(scores)
    
        # Create DataFrame
        comparison_df                    = pd.DataFrame(data = comparison_results)
        
        # Calculate composite score
        comparison_df['composite_score'] = self._calculate_composite_score(comparison_df = comparison_df)
        
        # Rank datasets
        comparison_df                    = comparison_df.sort_values(by        = 'composite_score', 
                                                                    ascending = False,
                                                                    )
                                                                    
        comparison_df['rank']            = range(1, len(comparison_df) + 1)
        
        return comparison_df


    def _calculate_dataset_scores(self, data: pd.DataFrame, labels: np.ndarray, metadata: Dict) -> Dict:
        """
        Calculate quality scores for a single dataset
        
        Arguments:
        ----------
            data     { pd.DataFrame } : PHQ-9 data
            
            labels   { np.ndarray }   : Cluster labels
            
            metadata { dict }         : Generation metadata
        
        Returns:
        --------
                   { dict }           : Score dictionary
        """
        scores                         = dict()
        
        # Temporal stability score
        scores['temporal_stability']   = self._calculate_temporal_stability(labels = labels)
        
        # Clinical realism score
        scores['clinical_realism']     = self._calculate_clinical_realism(data = data)
        
        # Statistical quality score
        scores['statistical_quality']  = self._calculate_statistical_quality(data = data)
        
        # Metadata consistency score
        scores['metadata_consistency'] = self._calculate_metadata_consistency(data     = data, 
                                                                              metadata = metadata,
                                                                             )
        
        return scores


    def _calculate_temporal_stability(self, labels: np.ndarray) -> float:
        """
        Calculate temporal stability score (fewer transitions = better)
        
        Arguments:
        ----------
            labels { np.ndarray } : Cluster labels
        
        Returns:
        --------
                 { float }        : Stability score [0, 100]
        """
        if((labels is None) or (labels.ndim != 1) or (len(labels) < 2)):
            # Neutral score
            return 50.0  
        
        # Count cluster transitions
        transitions              = np.sum(np.diff(labels) != 0)
        
        # Normalize: fewer transitions = higher score
        max_possible_transitions = len(labels) - 1
        stability_ratio          = 1.0 - (transitions / max_possible_transitions)
        
        return stability_ratio * 100.0


    def _calculate_clinical_realism(self, data: pd.DataFrame) -> float:
        """
        Calculate clinical realism score based on severity distribution
        
        Arguments:
        ----------
            data { pd.DataFrame } : PHQ-9 data
        
        Returns:
        --------
                 { float }        : Realism score [0, 100]
        """
        scores                   = data.values.flatten()
        scores                   = scores[~np.isnan(scores)]
        
        if (len(scores) == 0):
            return 0.0
        
        # Calculate Day-level CV (coefficient of variation) for clinical meaningfulness
        daily_mean               = data.mean(axis   = 1, 
                                             skipna = True,
                                            )

        daily_std                = data.std(axis   = 1, 
                                            skipna = True,
                                           )

        cv_values                = daily_std / (daily_mean + 1e-6)
        cv_values                = cv_values.replace([np.inf, -np.inf], np.nan)

        cv_mean                  = np.nanmean(cv_values)
        cv_in_range              = (eda_constants_instance.CV_RANGE_EXPECTED_MIN <= cv_mean <= eda_constants_instance.CV_RANGE_EXPECTED_MAX)

        # Calculate severity distribution
        minimal                  = np.sum(scores < eda_constants_instance.MINIMAL_SEVERITY_THRESHOLD) / len(scores)
        mild                     = np.sum((scores >= eda_constants_instance.MINIMAL_SEVERITY_THRESHOLD) & (scores < eda_constants_instance.MILD_SEVERITY_THRESHOLD)) / len(scores)
        moderate                 = np.sum((scores >= eda_constants_instance.MILD_SEVERITY_THRESHOLD) & (scores < eda_constants_instance.MODERATE_SEVERITY_THRESHOLD)) / len(scores)
        severe                   = np.sum(scores >= eda_constants_instance.MODERATELY_SEVERE_THRESHOLD) / len(scores)
        
        # Expected distribution for moderate-severe depression studies: most scores should be in moderate-severe range
        expected_moderate_severe = 0.70  # 70% in moderate-severe range
        observed_moderate_severe = moderate + severe
        
        distribution_match       = 100.0 - abs(observed_moderate_severe - expected_moderate_severe) * 100.0
        
        # Combine scores
        cv_score                 = 100.0 if cv_in_range else 50.0
        overall_score            = (cv_score * 0.5 + distribution_match * 0.5)
        
        return max(0.0, min(100.0, overall_score))


    def _calculate_statistical_quality(self, data: pd.DataFrame) -> float:
        """
        Calculate statistical quality score
        
        Arguments:
        ----------
            data { pd.DataFrame } : PHQ-9 data
        
        Returns:
        --------
                 { float }        : Quality score [0, 100]
        """
        scores         = data.values.flatten()
        scores         = scores[~np.isnan(scores)]
        
        if (len(scores) == 0):
            return 0.0
        
        # Outlier rate
        z_scores       = np.abs(stats.zscore(scores))
        outliers       = np.sum(z_scores > eda_constants_instance.OUTLIER_Z_SCORE_THRESHOLD)
        outlier_rate   = outliers / len(scores)
        
        outlier_score  = 100.0 if (outlier_rate <= eda_constants_instance.OUTLIER_RATE_ACCEPTABLE) else (100.0 * (1.0 - outlier_rate))
        
        # Distribution properties (skewness, kurtosis)
        skewness       = abs(stats.skew(scores))
        kurtosis       = abs(stats.kurtosis(scores))
        
        # Moderate skewness/kurtosis expected for PHQ-9
        skewness_score = 100.0 if (skewness < 1.0) else max(0.0, 100.0 - (skewness - 1.0) * 20.0)
        kurtosis_score = 100.0 if (kurtosis < 2.0) else max(0.0, 100.0 - (kurtosis - 2.0) * 20.0)
        
        # Combine
        overall_score  = (outlier_score * 0.5 + skewness_score * 0.25 + kurtosis_score * 0.25)
        
        return max(0.0, min(100.0, overall_score))


    def _calculate_metadata_consistency(self, data: pd.DataFrame, metadata: Dict) -> float:
        """
        Calculate metadata consistency score
        
        Arguments:
        ----------
            data     { pd.DataFrame } : PHQ-9 data
            
            metadata { dict }         : Generation metadata
        
        Returns:
        --------
            { float } : Consistency score [0, 100]
        """
        if metadata is None:
            # Neutral score if no metadata
            return 50.0  
        
        score                            = 100.0
        
        # Check shape consistency
        study_design                     = metadata.get('study_design', {})
        expected_days                    = study_design.get('total_days')
        expected_patients                = study_design.get('n_patients')
        
        observed_days, observed_patients = data.shape
        
        if ((expected_days is not None) and (observed_days != expected_days)):
            score -= 20.0
        
        if ((expected_patients is not None) and (observed_patients != expected_patients)):
            score -= 20.0
        
        # Check missingness consistency
        gen_stats            = metadata.get('generation_statistics', {})
        expected_missingness = gen_stats.get('missingness_rate')
        
        if expected_missingness is not None:
            observed_missingness = float(data.isna().sum().sum() / data.size)
            difference           = abs(observed_missingness - expected_missingness)
            
            if (difference > 0.03):  
                # >3% difference
                score -= 20.0
        
        return max(0.0, score)


    def _calculate_composite_score(self, comparison_df: pd.DataFrame) -> pd.Series:
        """
        Calculate weighted composite score
        
        Arguments:
        ----------
            comparison_df { pd.DataFrame } : Comparison DataFrame
        
        Returns:
        --------
                   { pd.Series }           : Composite scores
        """
        composite = (comparison_df['temporal_stability'] * eda_constants_instance.TEMPORAL_STABILITY_WEIGHT +
                     comparison_df['clinical_realism'] * eda_constants_instance.CLINICAL_REALISM_WEIGHT +
                     comparison_df['statistical_quality'] * eda_constants_instance.STATISTICAL_QUALITY_WEIGHT +
                     comparison_df['metadata_consistency'] * eda_constants_instance.METADATA_CONSISTENCY_WEIGHT
                    )
        
        return composite


    def generate_comparison_report(self, comparison_df: pd.DataFrame, output_path: Path):
        """
        Generate detailed comparison report
        
        Arguments:
        ----------
            comparison_df { pd.DataFrame } : Comparison results
            
            output_path   { Path }         : Output file path
        """
        report = {'comparison_timestamp' : pd.Timestamp.now().isoformat(),
                  'datasets_compared'    : len(comparison_df),
                  'scoring_weights'      : {'temporal_stability'   : eda_constants_instance.TEMPORAL_STABILITY_WEIGHT,
                                            'clinical_realism'     : eda_constants_instance.CLINICAL_REALISM_WEIGHT,
                                            'statistical_quality'  : eda_constants_instance.STATISTICAL_QUALITY_WEIGHT,
                                            'metadata_consistency' : eda_constants_instance.METADATA_CONSISTENCY_WEIGHT,
                                           },
                  'results'              : comparison_df.to_dict(orient = 'records'),
                  'recommendation'       : comparison_df.iloc[0]['dataset_name'] if len(comparison_df) > 0 else None,
                 }
        
        # Save report
        output_path.parent.mkdir(parents  = True, 
                                 exist_ok = True,
                                ) 
        
        with open(output_path, 'w') as f:
            json.dump(obj    = report, 
                      fp     = f, 
                      indent = 4,
                     )
        
        if self.logger:
            self.logger.info(f"Comparison report saved to {output_path}")
            self.logger.info(f"Recommended dataset: {report['recommendation']}")