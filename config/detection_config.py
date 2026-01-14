# Dependencies
import warnings
from typing import List
from pathlib import Path
from typing import Tuple
from typing import Literal
from pydantic import Field
from pydantic import validator
from pydantic import BaseModel
from pydantic import model_validator


class ChangePointDetectionConfig(BaseModel):
    """
    Unified configuration for change point detection on aggregated PHQ-9 statistics

    Supports:
    - Offline detection via PELT (Killick et al., 2012)
    - Online Bayesian detection via BOCPD (Adams & MacKay, 2007)

    Execution is fully config-driven:
    - Single detector
    - Side-by-side comparison (default)
    - Future extensibility
    """
    # EXECUTION CONTROL
    execution_mode               : Literal['single', 'compare', 'ensemble'] = Field(default     = 'compare',
                                                                                    description = "Execution mode: single detector or side-by-side comparison",
                                                                                   )

    detectors                    : List[Literal['pelt', 'bocpd']]           = Field(default     = ['pelt', 'bocpd'],
                                                                                    description = "List of detectors to run (order preserved)",
                                                                                   )

    
    # MODEL SELECTION (OPTIONAL POST-DETECTION)
    selection_enabled            : bool                                     = Field(default     = False,
                                                                                    description = "Whether to run model selection after detection",
                                                                                   )

    model_selection_config_path  : Path | None                              = Field(default     = Path('config/model_selection_config.json'),
                                                                                    description = "Path to model_selection_config.py (required if selection_enabled=True)",
                                                                                   )

    # INPUT / OUTPUT PATHS
    data_path                    : Path                                     = Field(default     = Path('data/raw/synthetic_phq9_data.csv'),
                                                                                    description = "Path to PHQ-9 data CSV file",
                                                                                   )
    
    results_base_directory       : Path                                     = Field(default     = Path('results/detection'),
                                                                                    description = "Base directory for all detection results",
                                                                                   )

    # PELT PARAMETERS (OFFLINE)
    penalty                      : float                                    = Field(default     = 0.5,
                                                                                    ge          = 0.1,
                                                                                    le          = 10.0,
                                                                                    description = "PELT penalty parameter",
                                                                                   )
    
    penalty_range                : Tuple[float, float]                      = Field(default     = (0.1, 10.0),
                                                                                    description = "Penalty tuning range for BIC",
                                                                                   )
    
    auto_tune_penalty            : bool                                     = Field(default     = True,
                                                                                    description = "Automatically tune PELT penalty via BIC",
                                                                                   )
    
    pelt_cost_models             : List[Literal['l1', 'l2', 'rbf', 'ar']]   = Field(default     = ['l1', 'l2', 'rbf', 'ar'],
                                                                                    description = "PELT cost function",
                                                                                   )
    
    jump                         : int                                      = Field(default     = 1,
                                                                                    ge          = 1,
                                                                                    le          = 5,
                                                                                    description = "PELT subsampling jump parameter",
                                                                                   )
    
    minimum_segment_size         : int                                      = Field(default     = 5,
                                                                                    ge          = 3,
                                                                                    le          = 15,
                                                                                    description = "Minimum segment size for PELT",
                                                                                   )

    # BOCPD PARAMETERS (ONLINE BAYESIAN)
    likelihood_model             : Literal['gaussian']                      = Field(default     = 'gaussian',
                                                                                    description = "BOCPD likelihood model(Gaussian only, Student-t not yet implemented)",
                                                                                   )

    max_run_length               : int                                      = Field(default     = 500,
                                                                                    ge          = 50,
                                                                                    le          = 2000,
                                                                                    description = "Maximum run length tracked",
                                                                                   )

    hazard_lambda                : float                                    = Field(default     = 30.0,
                                                                                    ge          = 2.0,
                                                                                    le          = 500.0,
                                                                                    description = "Expected run length (λ) for BOCPD. Auto-tuned if auto_tune_hazard=True"
                                                                                   )

    cp_posterior_threshold       : float                                    = Field(default     = 0.6,
                                                                                    ge          = 0.1,
                                                                                    le          = 0.99,
                                                                                    description = "Posterior probability threshold for CP declaration",
                                                                                   )

    reset_on_cp                  : bool                                     = Field(default     = True,
                                                                                    description = "Reset posterior after change point",
                                                                                   )

    posterior_smoothing          : int                                      = Field(default     = 3,
                                                                                    ge          = 1,
                                                                                    le          = 15,
                                                                                    description = "Smoothing window for posterior probabilities",
                                                                                   )

    bocpd_persistence            : int                                      = Field(default     = 3,
                                                                                    ge          = 1,
                                                                                    le          = 10,
                                                                                    description = "Minimum consecutive time steps CP posterior must exceed threshold",
                                                                                   )

    # BOCPD tuning
    auto_tune_hazard             : bool                                     = Field(default     = True,
                                                                                    description = "Automatically tune BOCPD hazard parameter",
                                                                                   )
    
    hazard_range                 : Tuple[float, float]                      = Field(default     = (10.0, 300.0),
                                                                                    description = "Hazard parameter search range (expected run lengths)",
                                                                                   )
    
    hazard_tuning_method         : Literal['predictive_ll', 'heuristic']    = Field(default     = 'heuristic',
                                                                                    description = "Method for BOCPD hazard tuning",
                                                                                   )

    # STATISTICAL TESTING
    alpha                        : float                                    = Field(default     = 0.05,
                                                                                    gt          = 0.0,
                                                                                    lt          = 0.5,
                                                                                    description = "Significance level",
                                                                                   )
    
    multiple_testing_correction  : Literal['bonferroni', 'fdr_bh', 'none']  = Field(default     = 'fdr_bh',
                                                                                    description = "Multiple testing correction method",
                                                                                   )
    
    effect_size_threshold        : float                                    = Field(default     = 0.3,
                                                                                    ge          = 0.0,
                                                                                    le          = 2.0,
                                                                                    description = "Minimum Cohen's d for meaningful change",
                                                                                   )
    
    statistical_test             : Literal['mannwhitney', 't-test', 'auto'] = Field(default     = 'auto',
                                                                                    description = "Statistical test selection. 'auto' selects Mann-Whitney U for most cases, t-test for large samples",
                                                                                   )

    # VISUALIZATION
    smoothing_window_size        : int                                      = Field(default     = 10,
                                                                                    ge          = 3,
                                                                                    le          = 30,
                                                                                    description = "Smoothing window for plots",
                                                                                   )
    
    figure_size                  : Tuple[int, int]                          = Field(default     = (15, 10),
                                                                                    description = "Figure size",
                                                                                   )
    
    dpi                          : int                                      = Field(default     = 300,
                                                                                    ge          = 100,
                                                                                    le          = 600,
                                                                                    description = "Figure DPI",
                                                                                   )


    # VALIDATORS
    @validator('minimum_segment_size')
    def validate_minimum_segment_size(cls, v):
        if (v < 5):
            raise ValueError("minimum_segment_size must be ≥5 for reliable statistical testing")
        
        return v


    @model_validator(mode='after')
    def validate_execution(self):
        if ((self.execution_mode == 'single') and (len(self.detectors) != 1)):
            raise ValueError("execution_mode='single' requires exactly one detector")

        if ((self.execution_mode == 'compare') and (len(self.detectors) < 2)):
            raise ValueError("execution_mode='compare' requires at least two detectors")

        if ((self.execution_mode == 'ensemble') and not self.selection_enabled):
            raise ValueError("execution_mode='ensemble' requires selection_enabled=True to define how models are combined or selected")

        return self


    # Cross-parameter validators
    @model_validator(mode='after')
    def validate_cross_parameters(self):
        """
        Validate parameter interactions
        """
        # Check: hazard_lambda vs max_run_length
        if (self.hazard_lambda > self.max_run_length):
            warnings.warn(f"hazard_lambda ({self.hazard_lambda}) > max_run_length ({self.max_run_length}) - expected run length unreachable. Consider increasing max_run_length.")

        # Check: posterior_smoothing vs bocpd_persistence
        if (self.posterior_smoothing > self.bocpd_persistence):
            warnings.warn(f"posterior_smoothing ({self.posterior_smoothing}) > bocpd_persistence ({self.bocpd_persistence}) - smoothing may obscure persistence detection.")

        # Check: minimum_segment_size vs penalty_range
        if (self.minimum_segment_size >= 10) and (self.penalty_range[1] < 1.0):
            warnings.warn(f"Large minimum_segment_size ({self.minimum_segment_size}) with small penalty_range ({self.penalty_range}) - "
                         f"may not detect any change points. Consider increasing penalty upper bound.")

        return self


    @validator('effect_size_threshold')
    def validate_effect_size(cls, v):
        if (v < 0.2):
            warnings.warn("Effect size < 0.2 may not be clinically meaningful (Cohen's d interpretation: 0.2=small, 0.5=medium, 0.8=large)")

        return v
    

    @validator('hazard_range')
    def validate_hazard_range(cls, v):
        low, high = v
        
        if (low < 2):
            raise ValueError("hazard_range lower bound must be ≥ 2 (run lengths < 2 are unrealistic)")

        if (high <= low):
            raise ValueError("hazard_range upper bound must be > lower bound")

        return v


    class Config:
        validate_assignment = True
        extra               = 'forbid'


    # HELPERS
    def create_output_directories(self):
        dirs = [self.results_base_directory,
                self.results_base_directory / "per_model",
                self.results_base_directory / "best_model",
                self.results_base_directory / "change_points",
                self.results_base_directory / "statistical_tests",
                self.results_base_directory / "plots",
                self.results_base_directory / "diagnostics",
               ]

        for d in dirs:
            d.mkdir(parents = True, exist_ok = True)


    def get_summary(self) -> dict:
        """
        Human-readable configuration summary: suitable for logs, reports, and JSON metadata
        """
        summary = {'Execution'            : {'Mode'      : self.execution_mode,
                                             'Detectors' : self.detectors,
                                            },
                    'Data'                : {'Input path' : str(self.data_path)},
                    'Statistical Testing' : {'Alpha'                       : self.alpha,
                                             'Test'                        : self.statistical_test,
                                             'Multiple testing correction' : self.multiple_testing_correction,
                                             'Effect size threshold'       : self.effect_size_threshold,
                                            },
                    'Visualization'       : {'Smoothing window' : self.smoothing_window_size,
                                             'Figure size'      : f"{self.figure_size[0]}x{self.figure_size[1]}",
                                             'DPI'              : self.dpi,
                                            },
                    'Output'              : {'Results directory' : str(self.results_base_directory)},
                   }

        if ('pelt' in self.detectors):
            summary['PELT'] = {'Cost model'        : self.pelt_cost_models,
                               'Penalty'           : self.penalty,
                               'Auto-tune penalty' : self.auto_tune_penalty,
                               'Penalty range'     : self.penalty_range,
                               'Min segment size'  : self.minimum_segment_size,
                               'Jump'              : self.jump,
                              }

        if ('bocpd' in self.detectors):
            summary['BOCPD'] = {'Hazard tuning'       : 'auto' if self.auto_tune_hazard else 'disabled',
                                'Hazard range'        : self.hazard_range,
                                'Hazard method'       : self.hazard_tuning_method,
                                'Likelihood model'    : self.likelihood_model,
                                'Prior type'          : 'empirical (data-adaptive)',
                                'Max run length'      : self.max_run_length,
                                'Posterior threshold' : self.cp_posterior_threshold,
                                'Posterior smoothing' : self.posterior_smoothing,
                                'bocpd_persistence'   : self.bocpd_persistence,
                                'Reset on CP'         : self.reset_on_cp,
                               }

        summary['Model Selection'] = {'Enabled' : self.selection_enabled,
                                      'Config'  : str(self.model_selection_config_path) if self.model_selection_config_path else None,
                                     }

        return summary