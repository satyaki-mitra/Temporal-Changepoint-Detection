# Dependencies
import json
import numpy as np
import pandas as pd
from typing import Dict
from typing import List
from typing import Tuple
from pathlib import Path
from typing import Optional
from phq9_analysis.utils.logging_util import setup_logger
from phq9_analysis.utils.logging_util import log_parameters
from phq9_analysis.generation.validators import DataValidator
from phq9_analysis.generation.trajectory_models import AR1Model
from phq9_analysis.utils.logging_util import log_section_header
from phq9_analysis.generation.validators import print_validation_report
from phq9_analysis.generation.trajectory_models import PatientTrajectory
from phq9_analysis.generation.trajectory_models import initialize_patient_trajectories


class PHQ9DataGenerator:
    """
    Generate clinically realistic synthetic PHQ-9 data
    
    Features:
    ---------
    - AR(1) temporal autocorrelation for symptom stability
    - Patient-specific heterogeneous trajectories
    - Realistic missing data (MCAR + informative dropout)
    - Symptom relapse modeling
    - Automatic validation against literature
    """
    def __init__(self, config):
        """
        Initialize generator with configuration
        
        Arguments:
        ----------
            config { DataGenerationConfig } : DataGenerationConfig instance
        """
        self.config               = config
        
        # Set up logging
        self.logger               = setup_logger(module_name = 'generation',
                                                 log_level   = 'INFO',
                                                 log_dir     = Path('logs')
                                                )
        
        # Initialize components
        self.ar_model             = AR1Model(random_seed = config.random_seed)
        self.patient_trajectories = dict()
        self.missingness_patterns = dict()
        self.observation_days     = list()
        
        # Set random seed
        np.random.seed(config.random_seed)
        
        self.logger.info(f"Initialized PHQ9DataGenerator | N={config.total_patients}, Days={config.total_days}, AR(1)={config.ar_coefficient:.2f}")
    

    def generate(self) -> pd.DataFrame:
        """
        Generate complete synthetic PHQ-9 dataset
        
        Returns:
        --------
            { pd.DataFram }    : DataFrame with Days as index, Patients as columns
        """
        log_section_header(self.logger, "STARTING DATA GENERATION")
        
        # Initialize patient trajectories
        self.logger.info("Step 1/4: Initializing patient trajectories...")
        self._initialize_trajectories()
        
        # Generate missingness patterns
        self.logger.info("Step 2/4: Generating missingness patterns...")
        self._generate_missingness_patterns()
        
        # Select observation days
        self.logger.info("Step 3/4: Selecting observation days...")
        self._select_observation_days()
        
        # Generate scores
        self.logger.info("Step 4/4: Generating PHQ-9 scores...")
        data = self._generate_scores()
        
        self.logger.info(f"Generation complete | Shape: {data.shape} | Sparsity: {data.isna().sum().sum() / data.size:.2%}")
        
        return data
    

    def _initialize_trajectories(self):
        """
        Initialize all patient trajectories
        """
        self.patient_trajectories = initialize_patient_trajectories(n_patients         = self.config.total_patients,
                                                                    baseline_mean      = self.config.baseline_mean_score,
                                                                    baseline_std       = self.config.baseline_std_score,
                                                                    recovery_rate_mean = self.config.recovery_rate_mean,
                                                                    recovery_rate_std  = self.config.recovery_rate_std,
                                                                    noise_std          = self.config.noise_std,
                                                                    ar_coefficient     = self.config.ar_coefficient,
                                                                    random_seed        = self.config.random_seed,
                                                                   )
        
        self.logger.info(f"Initialized {len(self.patient_trajectories)} patient trajectories")
    

    def _generate_missingness_patterns(self):
        """
        Generate realistic missing data patterns
        """
        for patient_id in range(1, self.config.total_patients + 1):
            # Determine dropout
            dropout_day = None

            if (np.random.random() < self.config.dropout_rate):
                # Exponential distribution for dropout timing
                dropout_day = int(np.random.exponential(scale = self.config.total_days * 0.3) + 100)
                dropout_day = min(dropout_day, self.config.total_days)
            
            # Random missed appointments (MCAR)
            n_missed                              = int(self.config.total_days * self.config.mcar_missingness_rate)
            missed_appointments                   = set(np.random.choice(range(1, self.config.total_days + 1), size = n_missed, replace = False))
            
            self.missingness_patterns[patient_id] = {'dropout_day'         : dropout_day,
                                                     'missed_appointments' : missed_appointments,
                                                    }
        
        n_dropouts = sum(1 for p in self.missingness_patterns.values() if p['dropout_day'])
        self.logger.info(f"Generated missingness patterns | Dropouts: {n_dropouts}/{self.config.total_patients}")

    
    def _select_observation_days(self):
        """
        Select days when surveys are administered
        """
        probabilities = list()
        
        for day in range(1, self.config.total_days + 1):
            if (day <= 30):
                prob = 0.85 + 0.10 * np.exp(-day / 10)

            elif (day <= 100):
                prob = 0.50 + 0.20 * np.exp(-(day - 30) / 20)

            else:
                prob = 0.15 + 0.10 * np.exp(-(day - 100) / 80)
            
            probabilities.append(prob)
        
        # Normalize
        probabilities         = np.array(probabilities)
        probabilities         = probabilities / probabilities.sum()
        
        # Sample
        self.observation_days = sorted(np.random.choice(range(1, self.config.total_days + 1),
                                                        size    = self.config.required_sample_count,
                                                        replace = False,
                                                        p       = probabilities,
                                                       )
                                      )
        
        self.logger.info(f"Selected {len(self.observation_days)} observation days (Days {self.observation_days[0]}-{self.observation_days[-1]})")

    
    def _generate_scores(self) -> pd.DataFrame:
        """
        Generate all PHQ-9 scores
        """
        data_dict = {f"Patient_{pid}": [np.nan] * len(self.observation_days) for pid in range(1, self.config.total_patients + 1)}
        
        for day_idx, day in enumerate(self.observation_days):
            for patient_id in range(1, self.config.total_patients + 1):
                # Check if missing
                if self._is_patient_missing(patient_id, day):
                    continue
                
                # Generate score
                trajectory                                  = self.patient_trajectories[patient_id]
                score                                       = self.ar_model.generate_score(trajectory             = trajectory,
                                                                                           day                    = day,
                                                                                           relapse_probability    = self.config.relapse_probability,
                                                                                           relapse_magnitude_mean = self.config.relapse_magnitude_mean,
                                                                                          )
                
                data_dict[f"Patient_{patient_id}"][day_idx] = score
        
        # Create DataFrame
        df            = pd.DataFrame(data  = data_dict,
                                     index = [f"Day_{day}" for day in self.observation_days],
                                    )
        df.index.name = 'Day'
        
        return df

    
    def _is_patient_missing(self, patient_id: int, day: int) -> bool:
        """
        Check if patient data is missing on given day
        """
        pattern     = self.missingness_patterns.get(patient_id, {})
        
        # Check dropout
        dropout_day = pattern.get('dropout_day')

        if (dropout_day and day >= dropout_day):
            return True
        
        # Check MCAR
        if (day in pattern.get('missed_appointments', set())):
            return True
        
        return False
    

    def validate(self, data: pd.DataFrame) -> Dict:
        """
        Validate generated data against literature
        
        Arguments:
        ----------
            data { pd.DataFrame} : Generated PHQ-9 DataFrame
        
        Returns:
        --------
                 { dict }        : Validation results dictionary
        """
        log_section_header(self.logger, "VALIDATING GENERATED DATA")
        
        validator  = DataValidator()
        validation = validator.validate_all(data)
        
        # Log results
        if validation['overall_valid']:
            self.logger.info("Validation PASSED")

        else:
            self.logger.warning("Validation has warnings/errors")
        
        # Log warnings
        for warning in validation['warnings']:
            self.logger.warning(warning)
        
        # Log errors
        for error in validation['errors']:
            self.logger.error(error)
        
        return validation
    
    
    def to_json_safe(self, obj):
        """
        utility that sanitize before json.dump 
        """
        if isinstance(obj, dict):
            return {k: self.to_json_safe(v) for k, v in obj.items()}
        
        elif isinstance(obj, list):
            return [self.to_json_safe(v) for v in obj]
        
        elif isinstance(obj, tuple):
            return tuple(self.to_json_safe(v) for v in obj)
        
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        elif isinstance(obj, np.integer):
            return int(obj)
        
        elif isinstance(obj, np.floating):
            return float(obj)
        
        else:
            return obj
            

    def save(self, data: pd.DataFrame, validation: Optional[Dict] = None):
        """
        Save data and validation results
        
        Arguments:
        ----------
            data       { pd.DataFrame } : Generated DataFrame

            validation     { dict }     : Validation results
        """
        # Save data
        data_path = self.config.output_data_path

        data_path.parent.mkdir(parents  = True, 
                               exist_ok = True,
                              )

        data.to_csv(path_or_buf = data_path, 
                    index       = True,
                   )

        self.logger.info(f"Data saved to: {data_path}")
        
        # Save validation
        if validation:
            val_path = self.config.validation_report_path
            val_path.parent.mkdir(parents  = True, 
                                  exist_ok = True,
                                 )
            
            with open(val_path, 'w') as f:
                json.dump(obj    = self.to_json_safe(obj = validation), 
                          fp     = f, 
                          indent = 4,
                         )
            
            self.logger.info(f"Validation report saved to: {val_path}")
    

    def generate_and_validate(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete pipeline: generate, validate, and save
        
        Returns:
        --------
            { tuple }    : A python tuple of (data, validation_results)
        """
        # Generate
        data       = self.generate()
        
        # Validate
        validation = self.validate(data)
        
        # Save
        self.save(data, validation)
        
        return data, validation