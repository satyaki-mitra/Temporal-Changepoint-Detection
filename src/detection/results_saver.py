# Dependencies
import json
import pandas as pd
from typing import Dict
from pathlib import Path
from datetime import datetime


class DetectionResultsSaver:
    """
    Saves detection results to structured directory hierarchy

    Directory Structure:
    --------------------
    results/detection_{dataset}/
    ├── all_model_results.json
    ├── per_model/
    │   ├── pelt_l1.json
    │   ├── pelt_l2.json
    │   └── bocpd_gaussian_heuristic.json
    ├── change_points/
    │   ├── pelt_l1_changepoints.csv
    │   ├── pelt_l2_changepoints.csv
    │   └── all_changepoints_comparison.csv
    ├── statistical_tests/
    │   ├── pelt_l1_tests.json
    │   ├── bocpd_gaussian_heuristic_validation.json
    │   └── statistical_summary.csv
    ├── diagnostics/
    │   ├── pelt_l1_segments.png
    │   └── detection_summary.json
    ├── plots/
    │   ├── aggregated_cv_all_models.png
    │   └── model_comparison_grid.png
    └── best_model/
        ├── metadata.json
        └── model_result.json
    """
    def __init__(self, base_directory: Path):
        """
        Initialize results saver

        Arguments:
        ----------
            base_directory { Path } : Base results directory (e.g., results/detection_exponential)
        """
        self.base_dir = Path(base_directory)
        self._create_directories()


    def _create_directories(self):
        """
        Create all required subdirectories
        """
        subdirs = [self.base_dir,
                   self.base_dir / "per_model",
                   self.base_dir / "change_points",
                   self.base_dir / "statistical_tests",
                   self.base_dir / "diagnostics",
                   self.base_dir / "plots",
                   self.base_dir / "best_model",
                  ]

        for d in subdirs:
            d.mkdir(parents = True, exist_ok = True)


    def save_all_results(self, model_results: Dict[str, Dict], config: Dict = None):
        """
        Save all detection results to appropriate locations

        Arguments:
        ----------
            model_results { dict } : All model results from orchestrator
            
            config        { dict } : Optional configuration metadata
        """
        # Save complete results JSON
        self._save_all_model_results(model_results = model_results)

        # Save individual model results
        for model_id, result in model_results.items():
            self._save_per_model_result(model_id = model_id, 
                                        result   = result,
                                       )

        # Save change points
        self._save_change_points(model_results = model_results)

        # Save statistical tests
        self._save_statistical_tests(model_results = model_results)

        # Save diagnostics summary
        self._save_diagnostics_summary(model_results = model_results, 
                                       config        = config,
                                      )


    def _save_all_model_results(self, model_results: Dict[str, Dict]):
        """
        Save complete results to all_model_results.json
        """
        output_path          = self.base_dir / "all_model_results.json"

        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_json_serializable(obj = model_results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=4)

        print(f"\n- All model results saved at: {output_path}")


    def _save_per_model_result(self, model_id: str, result: Dict):
        """
        Save individual model result to per_model/{model_id}.json
        """
        output_path   = self.base_dir / "per_model" / f"{model_id}.json"

        # Extract key information
        model_summary = {'model_id'       : model_id,
                         'method'         : result.get('method'),
                         'algorithm'      : result.get('algorithm'),
                         'variant'        : result.get('variant'),
                         'n_changepoints' : result.get('n_changepoints'),
                         'signal_length'  : result.get('signal_length'),
                         'change_points'  : result.get('change_points'),
                         'validation'     : result.get('validation'),
                         'timestamp'      : datetime.now().isoformat(),
                        }

        # Add method-specific metadata
        if (result.get('method') == 'pelt'):
            model_summary.update({'penalty_used'          : result.get('penalty_used'),
                                  'penalty_source'        : result.get('penalty_source'),
                                  'min_segment_size_used' : result.get('min_segment_size_used'),
                                  'tuning_results'        : result.get('tuning_results'),
                                })

        elif (result.get('method') == 'bocpd'):
            model_summary.update({'hazard_lambda'        : result.get('hazard_lambda'),
                                  'hazard_rate'          : result.get('hazard_rate'),
                                  'posterior_threshold'  : result.get('validation', {}).get('posterior_threshold'),
                                  'max_run_length_used'  : result.get('max_run_length_used'),
                                  'hazard_tuning'        : result.get('hazard_tuning'),
                                })

        # Make JSON serializable
        model_summary = self._make_json_serializable(model_summary)

        with open(output_path, 'w') as f:
            json.dump(obj    = model_summary, 
                      fp     = f, 
                      indent = 4,
                     )


    def _save_change_points(self, model_results: Dict[str, Dict]):
        """
        Save change points to CSV files
        """
        # Individual model change points
        for model_id, result in model_results.items():
            cps = result.get('change_points', [])

            if not cps:
                continue

            # Create DataFrame
            df            = pd.DataFrame({'change_point_index' : cps,
                                          'model_id'           : model_id,
                                        })

            # Add normalized positions for comparison
            signal_length = result.get('signal_length', 365)
            
            if (result.get('method') == 'pelt'):
                # Already indices
                df['normalized_position'] = df['change_point_index'] / signal_length
            
            else:  
                # bocpd - already normalized
                df['normalized_position'] = df['change_point_index']
                df['change_point_index']  = (df['normalized_position'] * signal_length).astype(int)

            output_path = self.base_dir / "change_points" / f"{model_id}_changepoints.csv"

            df.to_csv(output_path, index = False)

        # Comparison table (all models)
        comparison_data = list()

        for model_id, result in model_results.items():
            cps           = result.get('change_points', [])
            method        = result.get('method')
            signal_length = result.get('signal_length', 365)

            if (method == 'pelt'):
                indices    = cps
                normalized = [cp / signal_length for cp in cps]
            
            else:  
                # bocpd
                normalized = cps
                indices    = [int(cp * signal_length) for cp in cps]

            for idx, norm in zip(indices, normalized):
                comparison_data.append({'model_id'            : model_id,
                                        'method'              : method,
                                        'change_point_index'  : idx,
                                        'normalized_position' : norm,
                                        'day_label'           : f"Day_{idx+1}",
                                      })

        if comparison_data:
            df          = pd.DataFrame(data = comparison_data)
            df          = df.sort_values(['normalized_position', 'model_id'])

            output_path = self.base_dir / "change_points" / "all_changepoints_comparison.csv"
            df.to_csv(output_path, index = False)

            print(f"\n- All Changepoints Comparison Saved at: {output_path}")


    def _save_statistical_tests(self, model_results: Dict[str, Dict]):
        """
        Save statistical test results
        """
        for model_id, result in model_results.items():
            validation = result.get('validation', {})

            if not validation:
                continue

            # Save detailed tests
            output_path  = self.base_dir / "statistical_tests" / f"{model_id}_tests.json"

            test_summary = {'model_id'           : model_id,
                            'method'             : result.get('method'),
                            'test_family'        : validation.get('test_family'),
                            'n_changepoints'     : validation.get('n_changepoints'),
                            'n_significant'      : validation.get('n_significant'),
                            'overall_significant': validation.get('overall_significant'),
                            'alpha'              : validation.get('alpha'),
                            'tests'              : validation.get('tests', {}),
                            'summary'            : validation.get('summary', {}),
                            'rejection_reason'   : validation.get('rejection_reason'),
                            'timestamp'          : datetime.now().isoformat(),
                           }

            test_summary = self._make_json_serializable(test_summary)

            with open(output_path, 'w') as f:
                json.dump(obj    = test_summary, 
                          fp     = f, 
                          indent = 4,
                         )

        # Create summary table
        summary_data = list()

        for model_id, result in model_results.items():
            validation = result.get('validation', {})

            summary_data.append({'model_id'            : model_id,
                                 'method'              : result.get('method'),
                                 'n_changepoints'      : result.get('n_changepoints'),
                                 'n_significant'       : validation.get('n_significant', 0),
                                 'overall_significant' : validation.get('overall_significant', False),
                                 'test_family'         : validation.get('test_family'),
                                 'mean_effect_size'    : validation.get('summary', {}).get('mean_effect_size', 0.0),
                               })

        if summary_data:
            df          = pd.DataFrame(data = summary_data)

            output_path = self.base_dir / "statistical_tests" / "statistical_summary.csv"
            df.to_csv(output_path, index = False)

            print(f"\n- Statistical Tests Results Saved at: {output_path}")


    def _save_diagnostics_summary(self, model_results: Dict[str, Dict], config: Dict = None):
        """
        Save diagnostics summary JSON
        """
        summary = {'timestamp'     : datetime.now().isoformat(),
                   'n_models'      : len(model_results),
                   'models'        : list(model_results.keys()),
                   'configuration' : config,
                   'model_summary' : {},
                  }

        for model_id, result in model_results.items():
            summary['model_summary'][model_id] = {'method'         : result.get('method'),
                                                  'n_changepoints' : result.get('n_changepoints'),
                                                  'is_valid'       : result.get('validation', {}).get('overall_significant', False),
                                                 }

        output_path = self.base_dir / "diagnostics" / "detection_summary.json"

        summary     = self._make_json_serializable(obj = summary)

        with open(output_path, 'w') as f:
            json.dump(obj    = summary, 
                      fp     = f, 
                      indent = 4,
                     )

        print(f"\n- Model Diagnostics saved at: {output_path}")


    def save_best_model(self, best_model_id: str, model_results: Dict[str, Dict], selection_metadata: Dict = None):
        """
        Save best model selection results

        Arguments:
        ----------
            best_model_id       { str }  : ID of selected best model
            
            model_results       { dict } : All model results
            
            selection_metadata  { dict } : Model selection metadata (scores, explanation, etc.)
        """
        if best_model_id not in model_results:
            print(f"⚠ Warning: Best model '{best_model_id}' not found in results")
            return

        # Save best model result
        best_result   = model_results[best_model_id]
        result_path   = self.base_dir / "best_model" / "model_result.json"

        best_result   = self._make_json_serializable(best_result)

        with open(result_path, 'w') as f:
            json.dump(best_result, f, indent=4)

        # Save selection metadata
        metadata      = {'best_model_id'      : best_model_id,
                         'selection_method'   : selection_metadata.get('selection_strategy') if selection_metadata else 'manual',
                         'timestamp'          : datetime.now().isoformat(),
                         'n_candidates'       : selection_metadata.get('n_candidates') if selection_metadata else len(model_results),
                         'n_valid'            : selection_metadata.get('n_valid') if selection_metadata else None,
                         'scores'             : selection_metadata.get('scores') if selection_metadata else None,
                         'ranking'            : selection_metadata.get('ranking') if selection_metadata else None,
                         'explanation'        : selection_metadata.get('explanation') if selection_metadata else None,
                        }

        metadata_path = self.base_dir / "best_model" / "metadata.json"

        metadata      = self._make_json_serializable(obj = metadata)

        with open(metadata_path, 'w') as f:
            json.dump(obj    = metadata, 
                      fp     = f, 
                      indent = 4,
                     )

        print(f"\n - Saved best model: {best_model_id}")


    @staticmethod
    def _make_json_serializable(obj):
        """
        Convert numpy arrays and other non-serializable objects to JSON-compatible types
        """
        import numpy as np

        if isinstance(obj, dict):
            return {key: DetectionResultsSaver._make_json_serializable(value) for key, value in obj.items()}
        
        elif isinstance(obj, list):
            return [DetectionResultsSaver._make_json_serializable(item) for item in obj]
        
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        
        else:
            return obj