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

    model_selection_config_path  : Path | None                              = Field(default     = None,
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
    
    cost_model                   : Literal['l1', 'l2', 'rbf', 'ar']         = Field(default     = 'l1',
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

    # BOCPD tuning
    auto_tune_hazard             : bool                                     = Field(default     = True,
                                                                                    description = "Automatically tune BOCPD hazard parameter",
                                                                                   )
    
    hazard_range                 : Tuple[float, float]                      = Field(default     = (10.0, 300.0),
                                                                                    description = "Hazard parameter search range (expected run lengths)",
                                                                                   )
    
    hazard_tuning_method         : Literal['predictive_ll', 'heuristic']    = Field(default     = 'heuristic',
                                                                                    description = "Method for hazard tuning",
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
    
    statistical_test             : Literal['wilcoxon', 't-test', 'auto']    = Field(default     = 'auto',
                                                                                    description = "Statistical test selection",
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
            raise ValueError("minimum_segment_size must be ≥5")
        
        return v


    @model_validator(mode='after')
    def validate_execution(self):
        if ((self.execution_mode == 'single') and (len(self.detectors) != 1)):
            raise ValueError("execution_mode='single' requires exactly one detector")

        if ((self.execution_mode == 'compare') and (len(self.detectors) < 2)):
            raise ValueError("execution_mode='compare' requires at least two detectors")

        if ((self.execution_mode == 'ensemble') and (not self.selection_enabled)):
            warnings.warn("execution_mode='ensemble' typically requires selection_enabled=True")

        return self


    @validator('effect_size_threshold')
    def validate_effect_size(cls, v):
        if (v < 0.2):
            warnings.warn("Effect size < 0.2 may not be clinically meaningful")

        return v
    

    @validator('hazard_range')
    def validate_hazard_range(cls, v):
        lo, hi = v
        
        if (lo < 2):
            raise ValueError("hazard_range lower bound must be ≥ 2")

        if (hi <= lo):
            raise ValueError("hazard_range upper bound must be > lower bound")

        return v


    class Config:
        validate_assignment = True
        extra               = 'forbid'


    # HELPERS
    def create_output_directories(self):
        dirs = [self.results_base_directory,
                self.results_base_directory / "change_points",
                self.results_base_directory / "statistical_tests",
                self.results_base_directory / "plots",
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
                    'Output'              : {'Results directory' : str(self.results_base_directory)}
                   }

        if 'pelt' in self.detectors:
            summary['PELT'] = {'Cost model'        : self.cost_model,
                               'Penalty'           : self.penalty,
                               'Auto-tune penalty' : self.auto_tune_penalty,
                               'Penalty range'     : self.penalty_range,
                               'Min segment size'  : self.minimum_segment_size,
                               'Jump'              : self.jump,
                              }

        if 'bocpd' in self.detectors:
            summary['BOCPD'] = {'Hazard tuning'       : 'auto' if self.auto_tune_hazard else 'disabled',
                                'Hazard range'        : self.hazard_range,
                                'Hazard method'       : self.hazard_tuning_method,
                                'Likelihood model'    : self.likelihood_model,
                                'Prior type'          : 'empirical (data-adaptive)',
                                'Max run length'      : self.max_run_length,
                                'Posterior threshold' : self.cp_posterior_threshold,
                                'Posterior smoothing' : self.posterior_smoothing,
                                'Reset on CP'         : self.reset_on_cp,
                               }

        summary['Model Selection'] = {'Enabled' : self.selection_enabled,
                                      'Config'  : str(self.model_selection_config_path) if self.model_selection_config_path else None,
                                     }

        return summary