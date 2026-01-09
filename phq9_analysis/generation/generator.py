# Dependencies
import json
import numpy as np
import pandas as pd
from typing import Dict
from typing import Tuple
from pathlib import Path
from typing import Optional
from phq9_analysis.utils.logging_util import setup_logger
from phq9_analysis.generation.validators import DataValidator
from phq9_analysis.utils.logging_util import log_section_header
from phq9_analysis.generation.trajectory_models import AR1Model
from phq9_analysis.generation.validators import print_validation_report
from phq9_analysis.generation.trajectory_models import PatientTrajectory
from phq9_analysis.generation.trajectory_models import initialize_patient_trajectories


class PHQ9DataGenerator:
    """
    Generate clinically realistic synthetic PHQ-9 data

    Design Goals:
    -------------
    - Sparse, irregular survey sampling
    - Patient-specific longitudinal trajectories
    - Clinically grounded missingness (MCAR + dropout)
    - Population-level aggregation readiness (CV, change points)
    """
    def __init__(self, config):
        """
        Initialize generator with validated configuration

        Arguments:
        ----------
            config { DataGenerationConfig } : Data generation configuration
        """
        self.config               = config

        # CRITICAL: Validate survey count constraints
        if (self.config.min_surveys_attempted > self.config.maximum_surveys_attempted):
            raise ValueError(
                f"min_surveys_attempted ({self.config.min_surveys_attempted}) cannot exceed "
                f"maximum_surveys_attempted ({self.config.maximum_surveys_attempted})"
            )

        # Logger
        self.logger               = setup_logger(module_name = 'generation',
                                                 log_level   = 'INFO',
                                                 log_dir     = Path('logs')
                                                )

        # Models & state
        self.ar_model             = AR1Model(random_seed = config.random_seed)
        self.patient_trajectories = dict()
        self.missingness_patterns = dict()
        self.patient_schedules    = dict()
        self.observation_days     = list()

        np.random.seed(config.random_seed)

        self.logger.info(f"Initialized PHQ9DataGenerator | Patients={config.total_patients}, Study Days={config.total_days}, AR(1)={config.ar_coefficient:.2f}")

    
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

        return data


    def _initialize_trajectories(self):
        """
        Initialize patient-level trajectory parameters
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

        self.logger.info(f"Initialized trajectories for {len(self.patient_trajectories)} patients")


    def _generate_missingness_patterns(self):
        """
        Generate dropout and MCAR missingness patterns per patient
        """
        for pid in range(1, self.config.total_patients + 1):
            dropout_day = None

            if (np.random.rand() < self.config.dropout_rate):
                # Exponential dropout: most dropouts occur mid-to-late study
                # Scale factor   : controls typical dropout timing
                # Minimum offset : ensures some follow-up before dropout
                dropout_day = int(np.random.exponential(scale = self.config.total_days * 0.3) + 60)
                dropout_day = min(dropout_day, self.config.total_days)

            self.missingness_patterns[pid] = {'dropout_day' : dropout_day}

        n_dropouts = sum(1 for p in self.missingness_patterns.values() if p['dropout_day'])

        self.logger.info(f"Generated dropout patterns | Dropouts={n_dropouts}/{self.config.total_patients}")

    
    def _generate_patient_schedules(self):
        """
        Generate patient-specific survey days

        Guarantees:
        -----------
        - min_surveys_attempted ≤ surveys ≤ max_surveys_attempted
        - Surveys occur before dropout (if applicable)
        - Irregular spacing (clinically realistic)
        - Handles edge cases where max_day < n_surveys
        """
        for pid in range(1, self.config.total_patients + 1):
            n_surveys                   = np.random.randint(self.config.min_surveys_attempted,
                                                            self.config.maximum_surveys_attempted + 1,
                                                           )

            last_day                    = self.missingness_patterns[pid]['dropout_day']
            max_day                     = last_day - 1 if last_day else self.config.total_days

            # CRITICAL FIX: Handle case where n_surveys > available days
            if (n_surveys > max_day):
                n_surveys = max_day
                self.logger.debug(f"Patient {pid}: Reduced surveys to {n_surveys} (max_day={max_day})")

            # Edge case: If max_day < 1, skip this patient
            if (max_day < 1):
                self.logger.warning(f"Patient {pid}: Insufficient days for any surveys (max_day={max_day})")
                self.patient_schedules[pid] = []
                continue

            survey_days                 = np.sort(np.random.choice(np.arange(1, max_day + 1),
                                                                   size    = n_surveys,
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
        Generate PHQ-9 scores for all patients and observation days
        
        CRITICAL FIX:
        -------------
        - Removed redundant trajectory.last_day assignment (already handled in generate_score)
        - State updates are now exclusively managed by AR1Model.generate_score()
        """
        data      = {f"Patient_{pid}": [np.nan] * len(self.observation_days) for pid in range(1, self.config.total_patients + 1)}

        day_index = {day: idx for idx, day in enumerate(self.observation_days)}

        for pid, days in self.patient_schedules.items():
            trajectory = self.patient_trajectories[pid]

            for day in days:
                idx                         = day_index[day]
                
                # Generate score using AR(1) model
                # NOTE: generate_score() internally calls trajectory.update_last_observation()
                # which sets both trajectory.last_score and trajectory.last_day
                score                       = self.ar_model.generate_score(trajectory             = trajectory,
                                                                           day                    = day,
                                                                           relapse_probability    = self.config.relapse_probability,
                                                                           relapse_magnitude_mean = self.config.relapse_magnitude_mean,
                                                                          )

                data[f"Patient_{pid}"][idx] = score

        df            = pd.DataFrame(data  = data,
                                     index = [f"Day_{d}" for d in self.observation_days],
                                    )

        df.index.name = "Day"

        return df

    
    def validate(self, data: pd.DataFrame) -> Dict:
        """
        Validate generated data against clinical and statistical expectations
        """
        log_section_header(self.logger, "VALIDATING GENERATED DATA")

        validator  = DataValidator()
        validation = validator.validate_all(data)

        if validation['overall_valid']:
            self.logger.info("Validation PASSED")
        
        else:
            self.logger.warning("Validation completed with warnings/errors")

        for w in validation['warnings']:
            self.logger.warning(w)

        for e in validation['errors']:
            self.logger.error(e)

        return validation


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
        
        return obj


    def save(self, data: pd.DataFrame, validation: Optional[Dict] = None):
        """
        Persist generated data and validation report
        """
        self.config.output_data_path.parent.mkdir(parents  = True, 
                                                  exist_ok = True,
                                                 )

        data.to_csv(path_or_buf = self.config.output_data_path)

        self.logger.info(f"Data saved to {self.config.output_data_path}")

        if validation:
            self.config.validation_report_path.parent.mkdir(parents  = True, 
                                                            exist_ok = True,
                                                           )

            with open(self.config.validation_report_path, "w") as f:
                json.dump(obj    = self._to_json_safe(validation), 
                          fp     = f, 
                          indent = 4,
                         )

            self.logger.info(f"Validation report saved to {self.config.validation_report_path}")


    def generate_and_validate(self) -> Tuple[pd.DataFrame, Dict]:
        """
        End-to-end pipeline: generate → validate → save.
        """
        data       = self.generate()
        validation = self.validate(data)

        self.save(data, validation)
        
        return data, validation