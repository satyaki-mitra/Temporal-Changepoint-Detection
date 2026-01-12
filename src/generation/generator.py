# Dependencies
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict
from typing import Tuple
from pathlib import Path
from typing import Optional
from datetime import datetime
from src.utils.logging_util import setup_logger
from src.generation.validators import DataValidator
from src.utils.logging_util import log_section_header
from src.generation.trajectory_models import AR1Model
from src.generation.validators import print_validation_report
from src.generation.trajectory_models import PatientTrajectory
from config.clinical_constants import clinical_constants_instance
from src.generation.trajectory_models import initialize_patient_trajectories


class PHQ9DataGenerator:
    """
    Generate clinically realistic synthetic PHQ-9 data

    Design Goals:
    -------------
    - Sparse, irregular survey sampling
    - Patient-specific longitudinal trajectories
    - Clinically grounded missingness (MCAR + dropout)
    - Population-level aggregation readiness (CV, change points)
    
    FEATURES:
    -------------
    - Response pattern heterogeneity (early/gradual/late/non-responders)
    - Plateau logic for symptom stabilization
    - Metadata tracking for dataset provenance
    - Enhanced validation reporting
    """
    def __init__(self, config):
        """
        Initialize generator with validated configuration

        Arguments:
        ----------
            config { DataGenerationConfig } : Data generation configuration
        """
        # Configuration variables 
        self.config               = config

        # Logger
        self.logger               = setup_logger(module_name = 'generation',
                                                 log_level   = 'INFO',
                                                 log_dir     = Path('logs')
                                                )

        # Models & state
        self.ar_model             = AR1Model(random_seed = config.random_seed)

        # Storages
        self.patient_trajectories = dict()
        self.missingness_patterns = dict()
        self.patient_schedules    = dict()
        self.observation_days     = list()

        # Metadata for provenance
        self.generation_metadata  = dict()

        np.random.seed(config.random_seed)

        self.logger.info(f"Initialized PHQ9DataGenerator | Patients={config.total_patients}, Study Days={config.total_days}, AR(1)={config.ar_coefficient:.2f}, "
                         f"Relapse Dist={config.relapse_distribution}, Response Patterns={'Enabled' if config.enable_response_patterns else 'Disabled'}"
                        )

    
    def generate(self) -> pd.DataFrame:
        """
        Generate complete synthetic PHQ-9 dataset

        Returns:
        --------
            { pd.DataFrame } : Rows    = observation days 
                               Columns = patients
        """
        log_section_header(self.logger, "STARTING DATA GENERATION")

        self.logger.info("Step 1/5: Initializing patient trajectories")
        self._initialize_trajectories()

        self.logger.info("Step 2/5: Generating missingness patterns")
        self._generate_missingness_patterns()

        self.logger.info("Step 3/5: Generating patient survey schedules")
        self._generate_patient_schedules()

        self.logger.info("Step 4/5: Consolidating observation days")
        self._consolidate_observation_days()

        self.logger.info("Step 5/5: Generating PHQ-9 scores")
        data     = self._generate_scores()

        sparsity = data.isna().sum().sum() / data.size
        self.logger.info(f"Generation complete | Shape={data.shape} | Missingness={sparsity:.2%}")

        # Capture generation metadata
        self._capture_metadata(data)

        return data


    def _initialize_trajectories(self):
        """
        Initialize patient-level trajectory parameters with response patterns
        """
        self.patient_trajectories = initialize_patient_trajectories(n_patients               = self.config.total_patients,
                                                                    baseline_mean            = self.config.baseline_mean_score,
                                                                    baseline_std             = self.config.baseline_std_score,
                                                                    recovery_rate_mean       = self.config.recovery_rate_mean,
                                                                    recovery_rate_std        = self.config.recovery_rate_std,
                                                                    noise_std                = self.config.noise_std,
                                                                    ar_coefficient           = self.config.ar_coefficient,
                                                                    random_seed              = self.config.random_seed,
                                                                    enable_response_patterns = self.config.enable_response_patterns,
                                                                   )

        # Log response pattern distribution
        if self.config.enable_response_patterns:
            pattern_counts = dict()
            for traj in self.patient_trajectories.values():
                pattern_counts[traj.response_pattern] = pattern_counts.get(traj.response_pattern, 0) + 1
            
            self.logger.info("Response pattern distribution:")
            for pattern, count in sorted(pattern_counts.items()):
                pct = count / len(self.patient_trajectories) * 100
                self.logger.info(f"  {pattern}: {count} ({pct:.1f}%)")

        self.logger.info(f"Initialized trajectories for {len(self.patient_trajectories)} patients")


    def _generate_missingness_patterns(self):
        """
        Generate dropout and MCAR missingness patterns per patient
        """
        for pid in range(1, self.config.total_patients + 1):
            dropout_day = None

            if (np.random.rand() < self.config.dropout_rate):
                # Exponential dropout: most dropouts occur mid-to-late study
                dropout_day = int(np.random.exponential(scale = self.config.total_days * clinical_constants_instance.DROPOUT_EXPONENTIAL_SCALE_FACTOR) + clinical_constants_instance.DROPOUT_MINIMUM_OFFSET_DAYS)
                dropout_day = min(dropout_day, self.config.total_days)

            self.missingness_patterns[pid] = {'dropout_day': dropout_day}

        n_dropouts = sum(1 for p in self.missingness_patterns.values() if p['dropout_day'])

        self.logger.info(f"Generated dropout patterns | Dropouts={n_dropouts}/{self.config.total_patients} ({n_dropouts/self.config.total_patients:.1%})")

    
    def _generate_patient_schedules(self):
        """
        Generate patient-specific survey days

        Guarantees:
        -----------
        - min_surveys_attempted ≤ surveys ≤ max_surveys_attempted
        - Surveys occur before dropout (if applicable)
        - Irregular spacing (clinically realistic)
        - Handles cases where available days < min_surveys
        """
        for pid in range(1, self.config.total_patients + 1):
            n_surveys   = np.random.randint(self.config.min_surveys_attempted,
                                            self.config.maximum_surveys_attempted + 1,
                                           )

            # Determine last valid day for this patient
            dropout_day = self.missingness_patterns[pid]['dropout_day']
            
            if dropout_day:
                max_day = dropout_day - 1

            else:
                max_day = self.config.total_days

            available_days = max_day
            
            if (n_surveys > available_days):
                n_surveys = available_days
                self.logger.debug(f"Patient {pid}: Reduced surveys to {n_surveys} (only {available_days} days available before dropout)")

            if (available_days < 1):
                self.logger.warning(f"Patient {pid}: No valid days for surveys (dropout_day={dropout_day}, total_days={self.config.total_days})")
                self.patient_schedules[pid] = list()
                continue

            # Generate random survey days (irregular spacing)
            survey_days = np.sort(np.random.choice(np.arange(1, max_day + 1),
                                                   size    = min(n_surveys, available_days),
                                                   replace = False,
                                                  )
                                 )

            self.patient_schedules[pid] = survey_days.tolist()


    def _consolidate_observation_days(self):
        """
        Create global observation day index from all patient schedules
        """
        all_days              = set()

        for days in self.patient_schedules.values():
            all_days.update(days)

        self.observation_days = sorted(all_days)

        self.logger.info(f"Total unique observation days: {len(self.observation_days)} (Range: {self.observation_days[0]}–{self.observation_days[-1]})")

   
    def _generate_scores(self) -> pd.DataFrame:
        """
        Generate PHQ-9 scores for all patients and observation days: incorporates plateau logic and response patterns
        """
        data      = {f"Patient_{pid}": [np.nan] * len(self.observation_days) for pid in range(1, self.config.total_patients + 1)}

        day_index = {day: idx for idx, day in enumerate(self.observation_days)}

        for pid, days in self.patient_schedules.items():
            trajectory = self.patient_trajectories[pid]

            for day in days:
                idx   = day_index[day]
                
                score                       = self.ar_model.generate_score(trajectory             = trajectory,
                                                                           day                    = day,
                                                                           relapse_probability    = self.config.relapse_probability,
                                                                           relapse_magnitude_mean = self.config.relapse_magnitude_mean,
                                                                           relapse_distribution   = self.config.relapse_distribution,
                                                                           enable_plateau         = self.config.enable_plateau_logic,
                                                                          )

                data[f"Patient_{pid}"][idx] = score

        dataframe            = pd.DataFrame(data  = data,
                                            index = [f"Day_{d}" for d in self.observation_days],
                                           )

        dataframe.index.name = "Day"

        return dataframe

    
    def _capture_metadata(self, data: pd.DataFrame):
        """
        Capture generation metadata for dataset provenance: tracks all configuration and generation statistics
        """
        # Response pattern statistics
        pattern_dist  = dict()
        plateau_count = 0
        
        for traj in self.patient_trajectories.values():
            pattern_dist[traj.response_pattern] = pattern_dist.get(traj.response_pattern, 0) + 1
            if traj.in_plateau_phase:
                plateau_count += 1
        
        self.generation_metadata = {'generation_timestamp'          : datetime.now().isoformat(),
                                    'random_seed'                   : self.config.random_seed,
                                    'config_hash'                   : self._compute_config_hash(),
                                    'study_design'                  : {'n_patients'              : self.config.total_patients,
                                                                       'total_days'              : self.config.total_days,
                                                                       'min_surveys_per_patient' : self.config.min_surveys_attempted,
                                                                       'max_surveys_per_patient' : self.config.maximum_surveys_attempted,
                                                                      },
                                    'model_parameters'              : {'ar_coefficient'     : self.config.ar_coefficient,
                                                                       'baseline_mean'      : self.config.baseline_mean_score,
                                                                       'baseline_std'       : self.config.baseline_std_score,
                                                                       'recovery_rate_mean' : self.config.recovery_rate_mean,
                                                                       'recovery_rate_std'  : self.config.recovery_rate_std,
                                                                       'noise_std'          : self.config.noise_std,
                                                                      },
                                    'relapse_config'                : {'distribution'   : self.config.relapse_distribution,
                                                                       'probability'    : self.config.relapse_probability,
                                                                       'magnitude_mean' : self.config.relapse_magnitude_mean,
                                                                      },
                                    'missingness_config'            : {'dropout_rate' : self.config.dropout_rate,
                                                                       'mcar_rate'    : self.config.mcar_missingness_rate,
                                                                      },
                                    'features'                      : {'response_patterns_enabled' : self.config.enable_response_patterns,
                                                                       'plateau_logic_enabled'     : self.config.enable_plateau_logic,
                                                                      },
                                
                                    'response_pattern_distribution' : pattern_dist,
                                    'generation_statistics'         : {'total_observations'  : int(data.notna().sum().sum()),
                                                                       'missingness_rate'    : float(data.isna().sum().sum() / data.size),
                                                                       'n_observation_days'  : len(self.observation_days),
                                                                       'patients_in_plateau' : plateau_count if self.config.enable_plateau_logic else None,
                                                                      }
                                   }


    def _compute_config_hash(self) -> str:
        """
        Compute deterministic hash of configuration for reproducibility tracking
        
        Returns:
        --------
            { str } : SHA256 hash of configuration
        """
        try:
            config_dict = self.config.model_dump()

        except AttributeError:
            config_dict = self.config.dict()
        
        # Sort for determinism
        config_str = json.dumps(config_dict, sort_keys = True, default = str)
        
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


    def validate(self, data: pd.DataFrame) -> Dict:
        """
        Validate generated data against clinical and statistical expectations
        """
        log_section_header(self.logger, "VALIDATING GENERATED DATA")

        validator                         = DataValidator(max_autocorr_gap_days    = self.config.max_autocorr_gap_days,
                                                          autocorr_weight_halflife = self.config.autocorr_weight_halflife,
                                                          max_autocorr_window_days = self.config.max_autocorr_window_days,
                                                         )

        validation                        = validator.validate_all(data)

        validation['relapse_statistics']  = self._compute_relapse_statistics()
        validation['dropout_statistics']  = self._compute_dropout_statistics()
        validation['generation_metadata'] = self.generation_metadata  

        if validation['overall_valid']:
            self.logger.info("✅ Validation PASSED")
        
        else:
            self.logger.warning("⚠️  Validation completed with warnings/errors")

        for w in validation['warnings']:
            self.logger.warning(w)

        for e in validation['errors']:
            self.logger.error(e)

        return validation


    def _compute_relapse_statistics(self) -> Dict:
        """
        Compute relapse statistics from patient trajectories
        """
        total_relapses         = 0
        patients_with_relapses = 0
        relapse_counts         = list()
        
        for pid, trajectory in self.patient_trajectories.items():
            n_relapses      = len(trajectory.relapse_history)
            total_relapses += n_relapses

            relapse_counts.append(n_relapses)
            
            if (n_relapses > 0):
                patients_with_relapses += 1
        
        n_patients = len(self.patient_trajectories)
        
        return {'n_patients'                  : n_patients,
                'total_relapses'              : total_relapses,
                'patients_with_relapses'      : patients_with_relapses,
                'patient_relapse_rate'        : patients_with_relapses / n_patients if n_patients > 0 else 0.0,
                'mean_relapses_per_patient'   : total_relapses / n_patients if n_patients > 0 else 0.0,
                'max_relapses_single_patient' : max(relapse_counts) if relapse_counts else 0
               }


    def _compute_dropout_statistics(self) -> Dict:
        """
        Compute dropout statistics from missingness patterns
        """
        n_dropouts   = sum(1 for p in self.missingness_patterns.values() if p['dropout_day'] is not None)
        n_patients   = len(self.missingness_patterns)
        
        dropout_days = [p['dropout_day'] for p in self.missingness_patterns.values() if p['dropout_day'] is not None]
        
        return {'n_patients'         : n_patients,
                'n_dropouts'         : n_dropouts,
                'dropout_rate'       : n_dropouts / n_patients if n_patients > 0 else 0.0,
                'mean_dropout_day'   : float(np.mean(dropout_days)) if dropout_days else None,
                'median_dropout_day' : float(np.median(dropout_days)) if dropout_days else None,
               }
               

    def _to_json_safe(self, obj):
        """
        Convert NumPy / non-serializable objects to JSON-safe equivalents
        """
        if isinstance(obj, dict):
            return {k: self._to_json_safe(v) for k, v in obj.items()}

        if isinstance(obj, list):
            return [self._to_json_safe(v) for v in obj]
        
        if isinstance(obj, tuple):
            return [self._to_json_safe(v) for v in obj]
        
        if isinstance(obj, np.generic):
            return obj.item()
        
        if isinstance(obj, Path):
            return str(obj)
        
        return obj


    def save(self, data: pd.DataFrame, validation: Optional[Dict] = None):
        """
        Persist generated data and validation report with metadata
        """
        # Save data
        self.config.output_data_path.parent.mkdir(parents = True, exist_ok = True)
        data.to_csv(path_or_buf = self.config.output_data_path)
        
        self.logger.info(f"\nData saved to {self.config.output_data_path}")

        # Save metadata sidecar (NEW)
        metadata_path = self.config.output_data_path.with_suffix('.metadata.json')
        
        with open(metadata_path, 'w') as f:
            json.dump(obj    = self._to_json_safe(self.generation_metadata), 
                      fp     = f, 
                      indent = 4,
                     )
         
        self.logger.info(f"\nMetadata saved to {metadata_path}")

        # Save validation report with timestamp
        if validation:
            timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir       = self.config.validation_report_path.parent
            report_name      = self.config.validation_report_path.stem
            report_ext       = self.config.validation_report_path.suffix
            
            timestamped_path = report_dir / f"{report_name}_{timestamp}{report_ext}"
            latest_path      = self.config.validation_report_path
            
            report_dir.mkdir(parents = True, exist_ok = True)
            
            # Save timestamped version
            with open(timestamped_path, "w") as f:
                json.dump(obj    = self._to_json_safe(validation), 
                          fp     = f, 
                          indent = 4,
                         )
            
            # Save/overwrite latest
            with open(latest_path, "w") as f:
                json.dump(obj    = self._to_json_safe(validation), 
                          fp     = f, 
                          indent = 4,
                         )

            self.logger.info(f"\nValidation report saved to {timestamped_path}")
            self.logger.info(f"\nLatest report: {latest_path}")


    def generate_and_validate(self) -> Tuple[pd.DataFrame, Dict]:
        """
        End-to-end pipeline: generate → validate → save.
        """
        data       = self.generate()
        validation = self.validate(data)

        self.save(data, validation)
        
        return data, validation