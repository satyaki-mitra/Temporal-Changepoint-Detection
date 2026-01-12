# Dependencies
import json
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict
from typing import Tuple
from pathlib import Path
from src.validation.clinical_criteria import ClinicalBenchmark, StatisticalCriteria


class LiteratureValidator:
    """
    Validate PHQ-9 dataset against clinical literature benchmarks
    
    Validates:
    - Autocorrelation patterns (Kroenke et al., 2001)
    - Response rates (Rush et al., 2006 - STAR*D)
    - Baseline severity (typical RCT enrollment)
    - Improvement trajectories (Cuijpers et al., 2014)
    - Relapse patterns (Monroe & Harkness, 2011)
    - Statistical quality
    """
    def __init__(self):
        self.clinical    = ClinicalBenchmark()
        self.statistical = StatisticalCriteria()
    

    def validate_dataset(self, data_path: Path, eda_results_path: Path, generation_validation_path: Path) -> Dict:
        """
        Complete validation against literature
        
        Arguments:
        ----------
            data_path                  { Path } : Path to CSV data

            eda_results_path           { Path } : Path to EDA analysis_summary.json
            
            generation_validation_path { Path } : Path to validation_report.json
        
        Returns:
        --------
                        { dict }                : Validation report with scores and criteria matches
        """
        # Load data
        dataframe = pd.read_csv(filepath_or_buffer = data_path, 
                                index_col          = 0,
                               )
        
        
        with open(eda_results_path, 'r') as f:
            eda_results = json.load(f)

        with open(generation_validation_path, 'r') as f:
            gen_validation = json.load(f)
        
        validation = {'dataset'                     : data_path.stem,
                      'criteria_checks'             : {},
                      'scores'                      : {},
                      'warnings'                    : [],
                      'errors'                      : [],
                      'overall_valid'               : True,
                      'literature_compliance_score' : 0.0,
                     }
        
        # Run all checks
        self._check_autocorrelation(gen_validation, validation)
        self._check_baseline_severity(gen_validation, validation)
        self._check_response_rate(gen_validation, validation)
        self._check_improvement_trajectory(gen_validation, validation)
        self._check_dropout_rate(gen_validation, validation)
        self._check_statistical_quality(data, validation)
        self._check_temporal_pattern(data, validation)
        self._check_relapse_characteristics(gen_validation, validation)
        
        # Calculate overall compliance score
        validation['literature_compliance_score'] = self._calculate_compliance_score(validation = validation)
        
        # Determine overall validity
        validation['overall_valid']               = ((len(validation['errors']) == 0) and (validation['literature_compliance_score'] >= 0.70))
        
        return validation
    

    def _check_autocorrelation(self, gen_validation: Dict, validation: Dict):
        """
        Check autocorrelation against Kroenke et al. (2001)
        """
        autocorr                                         = gen_validation['checks']['autocorrelation']['mean_gap_aware']
        lo, hi                                           = self.clinical.AUTOCORRELATION_RANGE
        
        in_range                                         = (lo <= autocorr <= hi)
        validation['criteria_checks']['autocorrelation'] = {'value'          : autocorr,
                                                            'expected_range' : [lo, hi],
                                                            'passes'         : in_range,
                                                            'reference'      : 'Kroenke et al. (2001) - PHQ-9 test-retest r=0.84',
                                                           }
        
        if not in_range:validation['warnings'].append(f"Autocorrelation {autocorr:.3f} outside literature range [{lo}, {hi}]")
    

    def _check_baseline_severity(self, gen_validation: Dict, validation: Dict):
        """
        Check baseline severity against typical RCT enrollment
        """
        baseline                                           = gen_validation['checks']['baseline']['mean']
        lo, hi                                             = self.clinical.BASELINE_SEVERITY_RANGE
        
        in_range                                           = lo <= baseline <= hi
        validation['criteria_checks']['baseline_severity'] = {'value'          : baseline,
                                                              'expected_range' : [lo, hi],
                                                              'passes'         : in_range,
                                                              'reference'      : 'Typical RCT enrollment: PHQ-9 15-17 (moderate-severe)',
                                                             }
        
        if not in_range:
            validation['warnings'].append(f"Baseline severity {baseline:.1f} outside typical RCT range [{lo}, {hi}]")
    

    def _check_response_rate(self, gen_validation: Dict, validation: Dict):
        """
        Check response rate against STAR*D
        """
        response_rate                                  = gen_validation['checks']['response_rate']['rate']
        lo, hi                                         = self.clinical.RESPONSE_RATE_RANGE
        
        in_range                                       = lo <= response_rate <= hi
        validation['criteria_checks']['response_rate'] = {'value'          : response_rate,
                                                          'expected_range' : [lo, hi],
                                                          'passes'         : in_range,
                                                          'reference'      : 'Rush et al. (2006) - STAR*D Level-1: 47% response rate',
                                                         }
        
        if not in_range:
            validation['warnings'].append(f"Response rate {response_rate:.1%} outside STAR*D range [{lo*100:.0f}%, {hi*100:.0f}%]")
    

    def _check_improvement_trajectory(self, gen_validation: Dict, validation: Dict):
        """
        Check improvement magnitude against MCID
        """
        improvement                                  = gen_validation['checks']['improvement']['mean_improvement']
        
        meaningful                                   = improvement >= self.clinical.MIN_IMPROVEMENT
        validation['criteria_checks']['improvement'] = {'value'              : improvement,
                                                        'minimum_meaningful' : self.clinical.MIN_IMPROVEMENT,
                                                        'passes'             : meaningful,
                                                        'reference'          : 'Löwe et al. (2004) - MCID ≈ 5 points',
                                                       }
        
        if not meaningful:
            validation['warnings'].append(f"Mean improvement {improvement:.2f} below meaningful threshold {self.clinical.MIN_IMPROVEMENT}")

    
    def _check_dropout_rate(self, gen_validation: Dict, validation: Dict):
        """
        Check dropout against Fournier et al. (2010) meta-analysis
        
        Uses actual dropout data from generation process
        """
        # If dropout is properly tracked (needs to be added to validation report)
        if ('dropout_statistics' in gen_validation):
            dropout_data = gen_validation['dropout_statistics']
            
            dropout_rate = dropout_data.get('dropout_rate', 0.0)
            n_dropouts   = dropout_data.get('n_dropouts', 0)
            n_patients   = dropout_data.get('n_patients', 0)
            
        else:
            # Extract from generation config as fallback
            dropout_rate = gen_validation.get('config', {}).get('dropout_rate', 0.18)
        
        lo, hi                                   = self.clinical.DROPOUT_RATE_RANGE  # (0.10, 0.20)
        in_range                                 = lo <= dropout_rate <= hi
        
        validation['criteria_checks']['dropout'] = {'dropout_rate'   : dropout_rate,
                                                    'expected_range' : [lo, hi],
                                                    'passes'         : in_range,
                                                    'reference'      : 'Fournier et al. (2010) - Meta-analysis average ≈ 13%',
                                                   }
        
        if not in_range:
            validation['warnings'].append(f"Dropout rate {dropout_rate:.1%} outside meta-analysis range [{lo*100:.0f}%, {hi*100:.0f}%]")

    
    def _check_statistical_quality(self, data: pd.DataFrame, validation: Dict):
        """
        Check statistical properties
        """
        scores                                               = data.values.flatten()
        scores                                               = scores[~np.isnan(scores)]
        
        n_obs                                                = len(scores)
        variance                                             = np.var(scores)
        
        quality_checks                                       = {'n_observations'  : n_obs,
                                                                'min_required'    : self.statistical.MIN_OBSERVATIONS,
                                                                'sufficient_data' : n_obs >= self.statistical.MIN_OBSERVATIONS,
                                                                'variance'        : variance,
                                                                'non_degenerate'  : variance >= self.statistical.MIN_VARIANCE,
                                                               }
        
        validation['criteria_checks']['statistical_quality'] = quality_checks
        
        if not quality_checks['sufficient_data']:
            validation['errors'].append(f"Insufficient observations: {n_obs} < {self.statistical.MIN_OBSERVATIONS}")

    
    def _check_temporal_pattern(self, data: pd.DataFrame, validation: Dict):
        """
        Check for expected linear decline pattern
        """
        daily_avg                                         = data.mean(axis = 1, skipna = True).dropna()
        days                                              = np.arange(len(daily_avg))
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err       = stats.linregress(days, daily_avg)
        
        lo, hi                                            = self.clinical.TYPICAL_SLOPE_RANGE
        in_range                                          = lo <= slope <= hi
        
        validation['criteria_checks']['temporal_pattern'] = {'slope'          : slope,
                                                             'expected_range' : [lo, hi],
                                                             'r_squared'      : r_value**2,
                                                             'p_value'        : p_value,
                                                             'passes'         : in_range and (p_value < 0.05),
                                                             'reference'      : 'Expected linear decline in population average',
                                                            }
        
        if not in_range:
            validation['warnings'].append(f"Trajectory slope {slope:.4f} outside typical range [{lo}, {hi}]")

    
    def _check_relapse_characteristics(self, gen_validation: Dict, validation: Dict):
    """
    Check relapse patterns against Monroe & Harkness (2011)
    
    Extracts actual relapse data from generation process
    """
    # If relapse data is in validation report (needs to be added)
    if ('relapse_statistics' in gen_validation):
        relapse_data                             = gen_validation['relapse_statistics']
        
        relapse_rate                             = relapse_data.get('patient_relapse_rate', 0.0)  # % patients with ≥1 relapse
        avg_relapses_per_patient                 = relapse_data.get('mean_relapses_per_patient', 0.0)
        
        lo, hi                                   = self.clinical.RELAPSE_RATE_RANGE  # (0.20, 0.40)
        in_range                                 = lo <= relapse_rate <= hi
        
        validation['criteria_checks']['relapse'] = {'patient_relapse_rate'     : relapse_rate,
                                                    'avg_relapses_per_patient' : avg_relapses_per_patient,
                                                    'expected_range'           : [lo, hi],
                                                    'passes'                   : in_range,
                                                    'reference'                : 'Monroe & Harkness (2011) - 20-40% relapse within 1 year',
                                                   }
        
        if not in_range:
            validation['warnings'].append(f"Patient relapse rate {relapse_rate:.1%} outside literature range [{lo*100:.0f}%, {hi*100:.0f}%]")
    
    else:
        # Fallback: Mark as missing data
        validation['criteria_checks']['relapse'] = {'available' : False,
                                                    'reference' : 'Monroe & Harkness (2011) - 20-40% relapse within 1 year',
                                                    'note'      : 'Relapse statistics not tracked in generation validation report',
                                                   }

    
    def _calculate_compliance_score(self, validation: Dict) -> float:
        """
        Calculate overall literature compliance score (0-1)
        """
        checks  = validation['criteria_checks']
        
        weights = {'autocorrelation'     : 0.20,
                   'baseline_severity'   : 0.15,
                   'response_rate'       : 0.20,
                   'improvement'         : 0.15,
                   'statistical_quality' : 0.15,
                   'temporal_pattern'    : 0.15,
                 }
        
        score   = 0.0

        for criterion, weight in weights.items():
            if criterion in checks:
                if checks[criterion].get('passes', False):
                    score += weight
        
        return score