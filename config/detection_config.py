# Dependencies
import warnings
from pathlib import Path
from typing import Tuple
from typing import Literal
from pydantic import Field
from pydantic import validator
from pydantic import BaseModel
from pydantic import root_validator


class ChangePointDetectionConfig(BaseModel):
    """
    Configuration for PELT-based change point detection
    
    Parameters validated against:
    - Killick et al. (2012): Optimal detection of changepoints
    - Truong et al. (2020): Selective review of offline methods
    """
    # INPUT/OUTPUT PATHS
    data_path                    : Path                                    = Field(default     = Path('data/raw/synthetic_phq9_data.csv'),
                                                                                   description = "Path to PHQ-9 data CSV file",
                                                                                  )
    
    results_base_directory       : Path                                    = Field(default     = Path('results/detection'),
                                                                                   description = "Base directory for all detection results",
                                                                                  )
    
    # PELT ALGORITHM PARAMETERS
    penalty                      : float                                   = Field(default     = 0.5,
                                                                                   ge          = 0.1,
                                                                                   le          = 10.0,
                                                                                   description = "PELT penalty parameter (controls number of change points)",
                                                                                  )
    
    penalty_range                : Tuple[float, float]                     = Field(default     = (0.1, 10.0),
                                                                                   description = "Range for penalty parameter tuning via BIC",
                                                                                  )
    
    auto_tune_penalty            : bool                                    = Field(default     = True,
                                                                                   description = "Automatically tune penalty using BIC criterion",
                                                                                  )
    
    cost_model                   : Literal['l1', 'l2', 'rbf', 'ar']        = Field(default     = 'l1',
                                                                                   description = "Cost function: 'l1' (robust), 'l2' (Gaussian), 'rbf', 'ar'",
                                                                                  )
    
    jump                         : int                                     = Field(default     = 1,
                                                                                   ge          = 1,
                                                                                   le          = 5,
                                                                                   description = "Subsampling jump parameter (1=no subsampling)",
                                                                                  )
    
    minimum_segment_size         : int                                     = Field(default     = 5,
                                                                                   ge          = 3,
                                                                                   le          = 15,
                                                                                   description = "Minimum segment size (need ≥5 for reliable statistics)",
                                                                                  )
    
    # STATISTICAL TESTING PARAMETERS
    alpha                        : float                                   = Field(default     = 0.05,
                                                                                   gt          = 0.0,
                                                                                   lt          = 0.5,
                                                                                   description = "Significance level for hypothesis testing",
                                                                                  )
    
    multiple_testing_correction  : Literal['bonferroni', 'fdr_bh', 'none'] = Field(default     = 'fdr_bh',
                                                                                   description = "Multiple testing correction method (Benjamini-Hochberg recommended)",
                                                                                  )
    
    effect_size_threshold        : float                                   = Field(default     = 0.3,
                                                                                   ge          = 0.0,
                                                                                   le          = 2.0,
                                                                                   description = "Minimum Cohen's d for meaningful change (0.3=small, 0.5=medium, 0.8=large)",
                                                                                  )
    
    statistical_test             : Literal['wilcoxon', 't-test', 'auto']   = Field(default     = 'auto',
                                                                                   description = "Statistical test for change point validation ('auto' chooses based on data)",
                                                                                  )
    
    # VISUALIZATION PARAMETERS
    smoothing_window_size        : int                                     = Field(default     = 10,
                                                                                   ge          = 3,
                                                                                   le          = 30,
                                                                                   description = "Moving average window for trend visualization",
                                                                                  )
    
    figure_size                  : Tuple[int, int]                         = Field(default     = (15, 10),
                                                                                   description = "Default figure size (width, height)",
                                                                                  )
    
    dpi                          : int                                     = Field(default     = 300,
                                                                                   ge          = 100,
                                                                                   le          = 600,
                                                                                   description = "DPI for saved figures",
                                                                                  )
    
    # OUTPUT FILE PATHS
    aggregated_cv_data_path      : Path                                    = Field(default     = Path('results/detection/aggregated_cv_data.csv'),
                                                                                   description = "Path to save daily CV values",
                                                                                  )
    
    change_points_json_path      : Path                                    = Field(default     = Path('results/detection/change_points/analysis_results.json'),
                                                                                   description = "Path to save change point metadata JSON",
                                                                                  )
    
    statistical_tests_csv_path   : Path                                    = Field(default     = Path('results/detection/statistical_tests/test_results.csv'),
                                                                                   description = "Path to save statistical test results",
                                                                                  )
    
    cluster_boundaries_csv_path  : Path                                    = Field(default     = Path('results/detection/change_points/cluster_boundaries.csv'),
                                                                                   description = "Path to save cluster/segment boundaries",
                                                                                  )
    
    # Plot paths
    aggregated_plot_path         : Path                                    = Field(default     = Path('results/detection/plots/aggregated_cv_plot.png'),
                                                                                   description = "Path to save aggregated CV plot",
                                                                                  )
    
    change_points_plot_path      : Path                                    = Field(default     = Path('results/detection/plots/change_points_detected.png'),
                                                                                   description = "Path to save change points scatter plot",
                                                                                  )
    
    validation_plot_path         : Path                                    = Field(default     = Path('results/detection/plots/model_validation.png'),
                                                                                   description = "Path to save validation diagnostics plot",
                                                                                  )
    
    

    @validator('minimum_segment_size')
    def validate_minimum_segment_size(cls, v):
        """
        Ensure minimum points sufficient for statistics
        """
        if (v < 5):
            raise ValueError(f"minimum_segment_size={v} is too small. Need ≥5 for:\n"
                              "- Reliable mean/std estimation\n"
                              "- Valid hypothesis testing (Wilcoxon requires n≥5)\n"
                              "- Stable CV calculation"
                            )

        return v

    
    @root_validator
    def validate_penalty_range(cls, values):
        """
        Ensure penalty is within penalty_range
        """
        penalty       = values.get('penalty')
        penalty_range = values.get('penalty_range')
        
        if penalty_range and penalty:
            if (not (penalty_range[0] <= penalty <= penalty_range[1])):
                raise ValueError(f"penalty={penalty} outside penalty_range={penalty_range}")
        
        return values
    

    @validator('effect_size_threshold')
    def validate_effect_size(cls, v):
        """
        Validate effect size threshold
        """
        if (v < 0.2):
            warnings.warn(f"effect_size_threshold={v} is very small. Cohen's d < 0.2 may not represent meaningful clinical change.")

        return v

    
    class Config:
        validate_assignment = True
        extra               = 'forbid'
    

    def create_output_directories(self):
        """
        Create all output directories
        """
        dirs = [self.results_base_directory,
                self.results_base_directory / "change_points",
                self.results_base_directory / "statistical_tests",
                self.results_base_directory / "plots",
               ]
        
        for directory in dirs:
            directory.mkdir(parents = True, exist_ok = True)
    

    def get_summary(self) -> dict:
        """
        Get human-readable configuration summary
        """
        return {'Algorithm'           : {'Cost model'        : self.cost_model,
                                         'Penalty'           : self.penalty,
                                         'Auto-tune penalty' : self.auto_tune_penalty,
                                         'Min segment size'  : self.minimum_segment_size,
                                        },
                'Statistical Testing' : {'Alpha'                       : self.alpha,
                                         'Test'                        : self.statistical_test,
                                         'Multiple testing correction' : self.multiple_testing_correction,
                                         'Effect size threshold'       : self.effect_size_threshold,
                                        },
                'Visualization'       : {'Smoothing window' : self.smoothing_window_size,
                                         'Figure size'      : f"{self.figure_size[0]}x{self.figure_size[1]}",
                                        },
                'Output'              : {'Results directory' : str(self.results_base_directory)}
               }