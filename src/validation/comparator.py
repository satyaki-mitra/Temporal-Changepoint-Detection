# Dependencies
import json
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List
from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt


class DatasetComparator:
    """
    Compare multiple datasets and select best based on literature compliance
    """
    def __init__(self, validation_reports: List[Dict]):
        self.reports = validation_reports
    

    def rank_datasets(self) -> pd.DataFrame:
        """
        Rank datasets by literature compliance score
        
        Returns:
        --------
            { pd.DataFrame } : DataFrame with rankings
        """
        rankings = list()
        
        for report in self.reports:
            rankings.append({'dataset'          : report['dataset'],
                             'compliance_score' : report['literature_compliance_score'],
                             'n_warnings'       : len(report['warnings']),
                             'n_errors'         : len(report['errors']),
                             'overall_valid'    : report['overall_valid']
                           })
        
        df         = pd.DataFrame(rankings)
        df         = df.sort_values('compliance_score', ascending = False).reset_index(drop = True)
        df['rank'] = df.index + 1
        
        return df
    

    def generate_comparison_report(self, save_path: Path):
        """
        Generate detailed comparison report
        """
        ranking = self.rank_datasets()
        
        report  = {'comparison_timestamp' : pd.Timestamp.now().isoformat(),
                   'n_datasets_compared'  : len(self.reports),
                   'ranking'              : ranking.to_dict(orient='records'),
                   'best_dataset'         : ranking.iloc[0]['dataset'],
                   'detailed_criteria'    : self._compare_criteria(),
                  }
        
        with open(save_path, 'w') as f:
            json.dump(obj    = report, 
                      fp     = f, 
                      indent = 2,
                     )
        
        return report
    

    def _compare_criteria(self) -> Dict:
        """
        Compare all criteria across datasets
        """
        comparison = dict()
        
        for report in self.reports:
            dataset             = report['dataset']
            comparison[dataset] = dict()
            
            for criterion, data in report['criteria_checks'].items():
                if ('passes' in data):
                    comparison[dataset][criterion] = {'value'  : data.get('value'),
                                                      'passes' : data['passes'],
                                                     }
        
        return comparison
    

    def plot_comparison(self, save_path: Path):
        """
        Visualize dataset comparison
        """
        ranking   = self.rank_datasets()
        
        fig, axes = plt.subplots(nrows   = 1, 
                                 ncols   = 2, 
                                 figsize = (16, 6),
                                )
        
        # Plot 1: Compliance scores
        ax        = axes[0]
        colors    = ['#2ecc71' if v else '#e74c3c' for v in ranking['overall_valid']]

        ax.barh(ranking['dataset'], 
                ranking['compliance_score'], 
                color = colors,
               )

        ax.axvline(0.70, 
                   color     = 'red', 
                   linestyle = '--', 
                   linewidth = 2, 
                   label     = 'Minimum threshold (0.70)',
                  )

        ax.set_xlabel('Literature Compliance Score', fontsize = 12)
        ax.set_title('Dataset Quality Comparison', fontsize = 14, fontweight = 'bold')
        ax.legend()
        ax.grid(True, alpha = 0.3, axis = 'x')
        
        # Plot 2: Criteria heatmap
        ax              = axes[1]
        criteria_matrix = list()
        criteria_names  = list()
        
        for report in self.reports:
            row = list()
            for criterion, data in report['criteria_checks'].items():
                if ('passes' in data):
                    row.append(1 if data['passes'] else 0)
                    if (len(criteria_names) < len(report['criteria_checks'])):
                        criteria_names.append(criterion)

            criteria_matrix.append(row)
        
        sns.heatmap(criteria_matrix, 
                    annot       = True, 
                    fmt         = 'd', 
                    cmap        = 'RdYlGn',
                    xticklabels = criteria_names, 
                    yticklabels = ranking['dataset'],
                    cbar_kws    = {'label': 'Pass (1) / Fail (0)'}, 
                    ax          = ax,
                   )

        ax.set_title('Criteria Compliance Matrix', fontsize = 14, fontweight = 'bold')
        plt.setp(ax.get_xticklabels(), rotation = 45, ha = 'right')
        
        plt.tight_layout()
        plt.savefig(fname       = save_path, 
                    dpi         = 300, 
                    bbox_inches = 'tight',
                   )

        plt.close()
    

    def select_best_dataset(self) -> str:
        """
        Select best dataset based on compliance score
        """
        ranking = self.rank_datasets()
        best    = ranking.iloc[0]
        
        if not best['overall_valid']:
            raise ValueError(f"Best dataset '{best['dataset']}' failed validation checks!")
        
        return best['dataset']