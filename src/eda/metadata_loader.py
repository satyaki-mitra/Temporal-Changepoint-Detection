# Dependencies
import json
import warnings
import numpy as np
from typing import Dict
from pathlib import Path
from typing import Optional
from datetime import datetime
from src.utils.logging_util import setup_logger


class MetadataLoader:
    """
    Load and parse generation metadata for EDA validation: handles metadata sidecars created by PHQ9DataGenerator
    """
    def __init__(self, logger = None):
        """
        Initialize metadata loader
        
        Arguments:
        ----------
            logger { Logger } : Optional logger instance
        """
        self.logger   = logger or setup_logger('metadata_loader')
        self.metadata = None
    

    def load(self, metadata_path: Path) -> Optional[Dict]:
        """
        Load metadata from JSON file
        
        Arguments:
        ----------
            metadata_path { Path } : Path to .metadata.json file
        
        Returns:
        --------
                 { dict }          : Metadata dictionary or None if not found
        """
        if not metadata_path.exists():
            self.logger.warning(f"Metadata file not found: {metadata_path}")
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            self.logger.info(f"Loaded metadata from {metadata_path}")
            self._log_metadata_summary()
            
            return self.metadata
        
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse metadata JSON: {e}")
            return None
        
        except Exception as e:
            self.logger.error(f"Error loading metadata: {e}")
            return None
    

    def _log_metadata_summary(self):
        """
        Log key metadata information
        """
        if self.metadata is None:
            return
        
        self.logger.info("Metadata Summary:")
        self.logger.info(f"  Generation timestamp : {self.metadata.get('generation_timestamp', 'N/A')}")
        self.logger.info(f"  Random seed          : {self.metadata.get('random_seed', 'N/A')}")
        self.logger.info(f"  Config hash          : {self.metadata.get('config_hash', 'N/A')}")
        
        # Study design
        study_design = self.metadata.get('study_design', {})
        self.logger.info(f"  Patients             : {study_design.get('n_patients', 'N/A')}")
        self.logger.info(f"  Study days           : {study_design.get('total_days', 'N/A')}")
        
        # Relapse configuration
        relapse_config = self.metadata.get('relapse_config', {})
        self.logger.info(f"  Relapse distribution : {relapse_config.get('distribution', 'N/A')}")
        
        # Features
        features = self.metadata.get('features', {})
        self.logger.info(f"  Response patterns    : {features.get('response_patterns_enabled', 'N/A')}")
        self.logger.info(f"  Plateau logic        : {features.get('plateau_logic_enabled', 'N/A')}")
        
        # Response pattern distribution
        if ('response_pattern_distribution' in self.metadata):
            self.logger.info("  Response pattern distribution:")
            for pattern, count in self.metadata['response_pattern_distribution'].items():
                self.logger.info(f"    {pattern}: {count}")
    

    def get_generation_timestamp(self) -> Optional[datetime]:
        """
        Get generation timestamp as datetime object
        
        Returns:
        --------
            { datetime } : Generation timestamp or None
        """
        if self.metadata is None:
            return None
        
        timestamp_str = self.metadata.get('generation_timestamp')
        
        if timestamp_str is None:
            return None
        
        try:
            return datetime.fromisoformat(timestamp_str)
        
        except ValueError:
            self.logger.warning(f"Invalid timestamp format: {timestamp_str}")
            return None
    

    def get_relapse_distribution(self) -> Optional[str]:
        """
        Get relapse distribution type
        
        Returns:
        --------
            { str } : 'exponential', 'gamma', or 'lognormal'
        """
        if self.metadata is None:
            return None
        
        return self.metadata.get('relapse_config', {}).get('distribution')
    

    def get_response_pattern_distribution(self) -> Optional[Dict[str, int]]:
        """
        Get response pattern distribution from metadata
        
        Returns:
        --------
            { dict } : Pattern name → count
        """
        if self.metadata is None:
            return None
        
        return self.metadata.get('response_pattern_distribution')
    

    def get_expected_response_rates(self) -> Optional[Dict[str, float]]:
        """
        Calculate expected response pattern rates from metadata
        
        Returns:
        --------
            { dict } : Pattern name → proportion [0, 1]
        """
        pattern_counts = self.get_response_pattern_distribution()
        
        if pattern_counts is None:
            return None
        
        total = sum(pattern_counts.values())
        
        if (total == 0):
            return None
        
        return {pattern: count / total for pattern, count in pattern_counts.items()}
    

    def get_plateau_info(self) -> Optional[Dict]:
        """
        Get plateau-related information
        
        Returns:
        --------
            { dict } : Plateau statistics
        """
        if self.metadata is None:
            return None
        
        gen_stats = self.metadata.get('generation_statistics', {})
        features  = self.metadata.get('features', {})
        
        return {'plateau_enabled'     : features.get('plateau_logic_enabled', False),
                'patients_in_plateau' : gen_stats.get('patients_in_plateau'),
               }
    

    def get_relapse_statistics(self) -> Optional[Dict]:
        """
        Get relapse statistics from metadata
        
        Returns:
        --------
            { dict } : Relapse statistics (if available)
        """
        if self.metadata is None:
            return None
        
        # Check if validation report included relapse stats
        return self.metadata.get('relapse_statistics')
    

    def get_dropout_statistics(self) -> Optional[Dict]:
        """
        Get dropout statistics from metadata
        
        Returns:
        --------
            { dict } : Dropout statistics (if available)
        """
        if self.metadata is None:
            return None
        
        return self.metadata.get('dropout_statistics')
    

    def get_model_parameters(self) -> Optional[Dict]:
        """
        Get model parameters used in generation
        
        Returns:
        --------
            { dict } : Model parameters
        """
        if self.metadata is None:
            return None
        
        return self.metadata.get('model_parameters')
    

    def get_missingness_config(self) -> Optional[Dict]:
        """
        Get missingness configuration
        
        Returns:
        --------
            { dict } : Dropout rate, MCAR rate
        """
        if self.metadata is None:
            return None
        
        return self.metadata.get('missingness_config')
    

    def is_feature_enabled(self, feature_name: str) -> bool:
        """
        Check if a specific feature was enabled during generation
        
        Arguments:
        ----------
            feature_name { str } : Feature name (e.g., 'response_patterns_enabled')
        
        Returns:
        --------
            { bool } : True if enabled, False otherwise
        """
        if self.metadata is None:
            return False
        
        features = self.metadata.get('features', {})
        return features.get(feature_name, False)
    

    def validate_data_consistency(self, observed_shape: tuple, observed_missingness: float) -> Dict:
        """
        Validate observed data against metadata expectations
        
        Arguments:
        ----------
            observed_shape        { tuple } : (n_days, n_patients)
            
            observed_missingness { float }  : Observed missingness rate [0, 1]
        
        Returns:
        --------
                      { dict }              : Validation results with warnings
        """
        validation = {'consistent'    : True,
                      'warnings'      : [],
                      'mismatches'    : [],
                     }
        
        if self.metadata is None:
            validation['warnings'].append("No metadata available for validation")
            return validation
        
        # Check study design
        study_design                     = self.metadata.get('study_design', {})
        expected_days                    = study_design.get('total_days')
        expected_patients                = study_design.get('n_patients')
        
        observed_days, observed_patients = observed_shape
        
        if ((expected_days is not None) and (observed_days != expected_days)):
            validation['consistent'] = False
            validation['mismatches'].append(f"Days mismatch: expected {expected_days}, observed {observed_days}")
        
        if ((expected_patients is not None) and (observed_patients != expected_patients)):
            validation['consistent'] = False
            validation['mismatches'].append(f"Patients mismatch: expected {expected_patients}, observed {observed_patients}")
        
        # Check missingness
        gen_stats               = self.metadata.get('generation_statistics', {})
        expected_missingness    = gen_stats.get('missingness_rate')
        
        if (expected_missingness is not None):
            missingness_diff = abs(observed_missingness - expected_missingness)
            
            if (missingness_diff > 0.01):  
                # >1% difference
                validation['warnings'].append(f"Missingness difference: expected {expected_missingness:.3f}, observed {observed_missingness:.3f} (Δ={missingness_diff:.3f})")
        
        return validation


class MetadataValidator:
    """
    Validate EDA results against generation metadata: compares observed statistics to expected values from metadata
    """
    def __init__(self, metadata: Dict, logger = None):
        """
        Initialize validator with metadata
        
        Arguments:
        ----------
            metadata { dict }   : Metadata dictionary from MetadataLoader
            
            logger   { Logger } : Optional logger instance
        """
        self.metadata = metadata
        self.logger   = logger or setup_logger('metadata_validator')
    

    def validate_response_patterns(self, observed_patterns: Dict[str, int]) -> Dict:
        """
        Compare observed response pattern distribution to expected
        
        Arguments:
        ----------
            observed_patterns { dict } : Pattern name → count from EDA
        
        Returns:
        --------
            { dict } : Validation results
        """
        expected_patterns = self.metadata.get('response_pattern_distribution')
        
        if expected_patterns is None:
            return {'valid'    : True,
                    'warnings' : ["No expected response patterns in metadata"],
                   }
        
        validation = {'valid'               : True,
                      'warnings'            : [],
                      'expected_patterns'   : expected_patterns,
                      'observed_patterns'   : observed_patterns,
                      'pattern_differences' : {},
                     }
        
        # Calculate proportions
        expected_total = sum(expected_patterns.values())
        observed_total = sum(observed_patterns.values())
        
        for pattern in expected_patterns.keys():
            expected_pct                               = (expected_patterns[pattern] / expected_total * 100) if (expected_total > 0) else 0
            observed_pct                               = (observed_patterns.get(pattern, 0) / observed_total * 100) if (observed_total > 0) else 0
            
            difference                                 = abs(expected_pct - observed_pct)
            validation['pattern_differences'][pattern] = {'expected_pct' : expected_pct,
                                                          'observed_pct' : observed_pct,
                                                          'difference'   : difference,
                                                         }
            
            # Flag if difference > 10%
            if (difference > 10.0):
                validation['valid'] = False
                validation['warnings'].append(f"{pattern}: expected {expected_pct:.1f}%, observed {observed_pct:.1f}% (Δ={difference:.1f}%)")
        
        return validation
    

    def validate_plateau_rates(self, observed_plateau_count: int, total_patients: int) -> Dict:
        """
        Validate plateau detection against metadata
        
        Arguments:
        ----------
            observed_plateau_count { int } : Number of patients detected in plateau
            
            total_patients         { int } : Total patients analyzed
        
        Returns:
        --------
                      { dict }             : Validation results
        """
        plateau_info   = self.metadata.get('generation_statistics', {})
        expected_count = plateau_info.get('patients_in_plateau')
        
        if expected_count is None:
            return {'valid'    : True,
                    'warnings' : ["No plateau information in metadata"],
                   }
        
        validation = {'valid'          : True,
                      'warnings'       : [],
                      'expected_count' : expected_count,
                      'observed_count' : observed_plateau_count,
                     }
        
        # Calculate rates
        expected_rate               = (expected_count / total_patients * 100) if (total_patients > 0) else 0
        observed_rate               = (observed_plateau_count / total_patients * 100) if (total_patients > 0) else 0
        difference                  = abs(expected_rate - observed_rate)
        validation['expected_rate'] = expected_rate
        validation['observed_rate'] = observed_rate
        validation['difference']    = difference
        
        # Flag if difference > 15%
        if (difference > 15.0):
            validation['valid'] = False
            validation['warnings'].append(f"Plateau rate: expected {expected_rate:.1f}%, observed {observed_rate:.1f}% (Δ={difference:.1f}%)")
        
        return validation
    

    def validate_severity_distribution(self, observed_baseline_mean: float, observed_baseline_std: float) -> Dict:
        """
        Validate baseline severity distribution
        
        Arguments:
        ----------
            observed_baseline_mean { float } : Observed mean baseline score
            
            observed_baseline_std  { float } : Observed std dev baseline score
        
        Returns:
        --------
                      { dict }               : Validation results
        """
        model_params  = self.metadata.get('model_parameters', {})
        expected_mean = model_params.get('baseline_mean')
        expected_std  = model_params.get('baseline_std')
        
        if (expected_mean is None) or (expected_std is None):
            return {'valid'    : True,
                    'warnings' : ["No baseline parameters in metadata"],
                   }
        
        validation = {'valid'         : True,
                      'warnings'      : [],
                      'expected_mean' : expected_mean,
                      'observed_mean' : observed_baseline_mean,
                      'expected_std'  : expected_std,
                      'observed_std'  : observed_baseline_std,
                     }
        
        # Check mean difference
        mean_diff  = abs(observed_baseline_mean - expected_mean)
        
        if (mean_diff > 1.0):  # >1 point difference
            validation['warnings'].append(f"Baseline mean: expected {expected_mean:.1f}, observed {observed_baseline_mean:.1f} (Δ={mean_diff:.1f})")
        
        # Check std difference
        std_diff = abs(observed_baseline_std - expected_std)
        
        if (std_diff > 0.5): 
            # >0.5 point difference
            validation['warnings'].append(f"Baseline std: expected {expected_std:.1f}, observed {observed_baseline_std:.1f} (Δ={std_diff:.1f})")
        
        return validation


# Convenience functions
def load_metadata(data_path: Path, logger = None) -> Optional[Dict]:
    """
    Convenience function to load metadata sidecar
    
    Arguments:
    ----------
        data_path { Path }   : Path to data CSV file
        
        logger    { Logger } : Optional logger
    
    Returns:
    --------
             { dict }        : Metadata dictionary or None
    """
    metadata_path = data_path.with_suffix('.metadata.json')
    loader        = MetadataLoader(logger = logger)
    
    return loader.load(metadata_path)