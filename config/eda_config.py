# Dependencies
import warnings
from pathlib import Path
from typing import Tuple
from typing import Literal
from pydantic import Field
from pydantic import validator
from pydantic import BaseModel


class EDAConfig(BaseModel):
    """
    Configuration for exploratory data analysis of PHQ-9 data
    """
    # INPUT/OUTPUT PATHS
    data_file_path               : Path                                                                     = Field(default     = Path("data/raw/synthetic_phq9_data.csv"),
                                                                                                                    description = "Path to input PHQ-9 data CSV file",
                                                                                                                   )
                                                                                                                
    results_base_directory       : Path                                                                     = Field(default     = Path("results/eda"),
                                                                                                                    description = "Base directory for EDA results",
                                                                                                                   )
    
    # VISUALIZATION PARAMETERS
    figure_size                  : Tuple[int, int]                                                          = Field(default     = (15, 12),
                                                                                                                    description = "Default figure size (width, height) in inches",
                                                                                                                   )
    
    dpi                          : int                                                                      = Field(default     = 300,
                                                                                                                    ge          = 100,
                                                                                                                    le          = 600,
                                                                                                                    description = "DPI for saved figures",
                                                                                                                   )
    
    plot_style                   : Literal['default', 'seaborn', 'ggplot']                                  = Field(default     = 'seaborn',
                                                                                                                    description = "Matplotlib style for plots",
                                                                                                                   )
    
    color_palette                : str                                                                      = Field(default     = 'husl',
                                                                                                                    description = "Seaborn color palette",
                                                                                                                   )
    
    # CLUSTERING PARAMETERS
    max_clusters_to_test         : int                                                                      = Field(default     = 20,
                                                                                                                    ge          = 5,
                                                                                                                    le          = 30,
                                                                                                                    description = "Maximum number of clusters for elbow/silhouette analysis",
                                                                                                                   )
    
    clustering_random_seed       : int                                                                      = Field(default     = 1234,
                                                                                                                    description = "Random seed for clustering reproducibility",
                                                                                                                   )
    
    clustering_algorithm         : Literal['kmeans', 'agglomerative', 'both']                               = Field(default     = 'kmeans',
                                                                                                                    description = "Clustering algorithm to use",
                                                                                                                   )
    
    # MISSING DATA HANDLING
    imputation_method            : Literal['mean', 'median', 'forward_fill', 'iterative', 'complete_cases'] = Field(default     = 'mean',
                                                                                                                    description = "Method for handling missing values in clustering",
                                                                                                                   )
    
    min_observations_per_patient : int                                                                      = Field(default     = 3,
                                                                                                                    ge          = 2,
                                                                                                                    le          = 10,
                                                                                                                    description = "Minimum observations required to include patient in analysis",
                                                                                                                   )
    
    
    # ADVANCED CLUSTERING OPTIONS
    use_temporal_clustering      : bool                                                                     = Field(default     = False,
                                                                                                                    description = "Use temporal constraints in clustering (penalize distant days)",
                                                                                                                   )
    
    temporal_weight              : float                                                                    = Field(default     = 0.3,
                                                                                                                    ge          = 0.0,
                                                                                                                    le          = 1.0,
                                                                                                                    description = "Weight for temporal proximity in clustering (0=ignore time, 1=only time)",
                                                                                                                   )
    
    standardize_features         : bool                                                                     = Field(default     = False,
                                                                                                                    description = "Standardize features before clustering (usually not needed for PHQ-9)",
                                                                                                                   )
    
    # STATISTICAL ANALYSIS PARAMETERS
    confidence_level             : float                                                                    = Field(default     = 0.95,
                                                                                                                    ge          = 0.90,
                                                                                                                    le          = 0.99,
                                                                                                                    description = "Confidence level for statistical tests",
                                                                                                                   )
    
    outlier_detection            : bool                                                                     = Field(default     = True,
                                                                                                                    description = "Flag potential outliers in the data",
                                                                                                                   )
    
    outlier_threshold            : float                                                                    = Field(default     = 3.0,
                                                                                                                    ge          = 2.0,
                                                                                                                    le          = 5.0,
                                                                                                                    description = "Z-score threshold for outlier detection",
                                                                                                                   )
    
    # OUTPUT CONTROL
    save_intermediate_results    : bool                                                                     = Field(default     = True,
                                                                                                                    description = "Save intermediate analysis results",
                                                                                                                   )
    
    generate_comprehensive_report: bool                                                                     = Field(default     = True,
                                                                                                                    description = "Generate comprehensive HTML/PDF report",
                                                                                                                   )
    
    # VALIDATORS
    @validator('figure_size')
    def validate_figure_size(cls, v):
        """
        Ensure reasonable figure dimensions
        """
        width, height = v

        if ((width < 6) or (height < 4)):
            raise ValueError(f"Figure size {v} too small. Minimum (6, 4)")

        if ((width > 30) or (height > 30)):
            raise ValueError(f"Figure size {v} too large. Maximum (30, 30)")

        return v

    
    @validator('temporal_weight')
    def validate_temporal_weight(cls, v, values):
        """
        Ensure temporal weight makes sense
        """
        use_temporal = values.get('use_temporal_clustering', False)

        if (use_temporal and (v == 0.0)):
            raise ValueError("temporal_weight cannot be 0.0 when use_temporal_clustering=True")

        if (not use_temporal and (v > 0.0)):
            
            warnings.warn(f"temporal_weight={v} will be ignored since use_temporal_clustering=False")

        return v
    

    @validator('max_clusters_to_test')
    def validate_max_clusters(cls, v):
        """
        Warn if testing too many clusters
        """
        if (v > 15):
            warnings.warn(f"Testing {v} clusters may be computationally expensive. Consider reducing to 10-15 for faster analysis.")

        return v
    

    class Config:
        validate_assignment = True
        extra               = 'forbid'
    

    def get_clustering_output_dir(self) -> Path:
        """
        Get clustering output directory
        """
        return self.results_base_directory / "clustering"
    

    def get_visualization_output_dir(self) -> Path:
        """
        Get visualization output directory
        """
        return self.results_base_directory / "visualizations"
    

    def create_output_directories(self):
        """
        Create all output directories
        """
        self.results_base_directory.mkdir(parents = True, exist_ok = True)
        self.get_clustering_output_dir().mkdir(parents = True, exist_ok = True)
        self.get_visualization_output_dir().mkdir(parents = True, exist_ok = True)
    

    def get_summary(self) -> dict:
        """
        Get human-readable configuration summary
        """
        return {
                'Data'          : {'Input file'               : str(self.data_file_path),
                                   'Min observations/patient' : self.min_observations_per_patient,
                                  },
                'Clustering'    : {'Algorithm'           : self.clustering_algorithm,
                                   'Max clusters tested' : self.max_clusters_to_test,
                                   'Imputation method'   : self.imputation_method,
                                   'Temporal clustering' : self.use_temporal_clustering,
                                  },
                'Visualization' : {'Figure size' : f"{self.figure_size[0]}x{self.figure_size[1]}",
                                   'DPI'         : self.dpi,
                                   'Style'       : self.plot_style,
                                  },
                'Output'        : {'Results directory'  : str(self.results_base_directory),
                                   'Save intermediates' : self.save_intermediate_results,
                                  }
               }
