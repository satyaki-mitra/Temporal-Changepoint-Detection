# Dependencies
import json
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic import BaseModel
from config.eda_config import EDAConfig
from config.generation_config import DataGenerationConfig
from config.detection_config import ChangePointDetectionConfig


class MasterConfig(BaseModel):
    """
    Master configuration combining all modules
    
    This class provides:
    - Unified configuration management
    - Cross-module validation
    - Save/load functionality
    - Directory creation
    """
    # MODULE CONFIGURATIONS
    generation       : DataGenerationConfig                         = Field(default_factory = DataGenerationConfig,
                                                                            description     = "Data generation configuration",
                                                                           )
    
    eda              : EDAConfig                                    = Field(default_factory = EDAConfig,
                                                                            description     = "Exploratory data analysis configuration",
                                                                           )
    
    detection        : ChangePointDetectionConfig                   = Field(default_factory = ChangePointDetectionConfig,
                                                                            description     = "Change point detection configuration",
                                                                           )
    
    # GLOBAL SETTINGS
    project_name     : str                                          = Field(default     = "PHQ-9 Temporal Change-point Detection",
                                                                            description = "Project name"
                                                                           )
    
    log_level        : Literal['DEBUG', 'INFO', 'WARNING', 'ERROR'] = Field(default     = 'INFO',
                                                                            description = "Global logging level",
                                                                           )
    
    log_directory    : Path                                         = Field(default     = Path('logs'),
                                                                            description = "Base directory for all logs",
                                                                           )
    
    enable_profiling : bool                                         = Field(default     = False,
                                                                            description = "Enable execution time profiling",
                                                                           )
    
    random_seed      : int                                          = Field(default     = 2023,
                                                                            description = "Global random seed (overrides module seeds if set)",
                                                                           )
     
    
    def create_all_directories(self):
        """
        Create all necessary directories for the project
        """
        # Base directories
        self.log_directory.mkdir(parents = True, exist_ok = True)
        Path('data/raw').mkdir(parents = True, exist_ok = True)
        Path('data/processed').mkdir(parents = True, exist_ok = True)
        
        # Module log directories
        (self.log_directory / 'generation').mkdir(parents = True, exist_ok = True)
        (self.log_directory / 'eda').mkdir(parents = True, exist_ok = True)
        (self.log_directory / 'detection').mkdir(parents = True, exist_ok = True)
        
        # Generation directories
        self.generation.output_data_path.parent.mkdir(parents = True, exist_ok = True)
        self.generation.validation_report_path.parent.mkdir(parents = True, exist_ok = True)
        
        # EDA directories
        self.eda.create_output_directories()
        
        # Detection directories
        self.detection.create_output_directories()
    

    def save(self, path: Path):
        """
        Save configuration to JSON file
        
        Arguments:
        ----------
            path { Path } : Path to save configuration JSON
        """
        path.parent.mkdir(parents = True, exist_ok = True)
        
        with open(path, 'w') as f:
            json.dump(obj     = self.dict(),
                      fp      = f,
                      indent  = 4,
                      default = str,  # Convert Path objects to strings
                     )
    

    @classmethod
    def load(cls, path: Path) -> 'MasterConfig':
        """
        Load configuration from JSON file
        
        Arguments:
        ----------
            path { Path } : Path to configuration JSON
        
        Returns:
        --------
            MasterConfig instance
        """
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    

    def get_full_summary(self) -> dict:
        """
        Get comprehensive summary of all configurations
        """
        return {'Project'    : {'Name'        : self.project_name,
                                'Log Level'   : self.log_level,
                                'Random Seed' : self.random_seed,
                               },
                'Generation' : self.generation.get_summary(),
                'EDA'        : self.eda.get_summary(),
                'Detection'  : self.detection.get_summary(),
               }

    
    def print_summary(self):
        """
        Print formatted configuration summary
        """
        summary = self.get_full_summary()
        
        print("=" * 80)
        print(f"{self.project_name} - CONFIGURATION SUMMARY".center(80))
        print("=" * 80)
        
        for module, params in summary.items():
            print(f"\n{'─' * 80}")
            print(f"{module.upper()}")
            print('─' * 80)
            
            if isinstance(params, dict):
                self._print_dict_recursive(params, indent = 2)
            
            else:
                print(f"  {params}")
        
        print("\n" + "=" * 80)
    

    def _print_dict_recursive(self, d: dict, indent: int = 0):
        """
        Recursively print nested dictionary
        """
        for key, value in d.items():
            if isinstance(value, dict):
                print(f"{'  ' * indent}{key}:")
                self._print_dict_recursive(value, indent + 1)
            
            else:
                print(f"{'  ' * indent}{key}: {value}")
    

    class Config:
        validate_assignment = True



# Convenience functions
def load_config(path: Path = Path('config/master_config.json')) -> MasterConfig:
    """
    Load configuration from JSON file
    
    Arguments:
    ----------
        path { Path }   : Path to configuration file
    
    Returns:
    --------
       { MasterConfig } : MasterConfig instance
    """
    return MasterConfig.load(path)


def save_config(config: MasterConfig, path: Path = Path('config/master_config.json')):
    """
    Save configuration to JSON file
    
    Arguments:
    ----------
        config { MasterConfig } : MasterConfig instance

        path       { Path }     : Path to save configuration
    """
    config.save(path)


def get_default_config() -> MasterConfig:
    """
    Get default configuration with all defaults
    
    Returns:
    --------
        { MasterConfig } : MasterConfig with default values
    """
    config = MasterConfig()
    config.create_all_directories()

    return config