# Dependencies
from typing import List
from typing import Dict
from typing import Literal
from pydantic import Field
from typing import Optional
from pydantic import validator
from pydantic import BaseModel


class ModelSelectorConfig(BaseModel):
    """
    Configuration for selecting the best change-point model from a pool of completed PELT and BOCPD variants

    The selector:
    - Treats each model run independently
    - Scores models using comparable metrics
    - Rewards cross-model agreement
    - Produces a ranked, explainable outcome
    """
    # MODEL ELIGIBILITY
    allowed_families         : List[Literal['pelt', 'bocpd']]                                                                                = Field(default     = ['pelt', 'bocpd'],
                                                                                                                                                     description = "Detector families eligible for selection",
                                                                                                                                                    )

    # Optional hard filter (exact model_ids)
    allowed_model_ids        : Optional[List[str]]                                                                                           = Field(default     = None,
                                                                                                                                                     description = "Explicit allow-list of model_ids (None = all)",
                                                                                                                                                    )

    # METRICS USED FOR SCORING (GLOBAL)
    scoring_metrics          : List[Literal['n_significant_cps', 'mean_effect_size', 'posterior_mass', 'cp_persistence', 'stability_score']] = Field(default     = ['n_significant_cps', 'mean_effect_size', 'stability_score'],
                                                                                                                                                     description = "Metrics used to score each model",
                                                                                                                                                    )

    # METRIC NORMALIZATION
    metric_normalization     : Dict[str, Literal['minmax', 'zscore', 'rank']]                                                                = Field(default     = {'n_significant_cps' : 'minmax',
                                                                                                                                                                    'mean_effect_size'  : 'minmax',
                                                                                                                                                                    'posterior_mass'    : 'minmax',
                                                                                                                                                                    'cp_persistence'    : 'minmax',
                                                                                                                                                                    'stability_score'   : 'minmax',
                                                                                                                                                                   },
                                                                                                                                                     description = "How each metric is normalized across models",
                                                                                                                                                    )

    # METRIC WEIGHTS (SUM ≤ 1.0)
    metric_weights           : Dict[str, float]                                                                                              = Field(default      = {'n_significant_cps' : 0.30,
                                                                                                                                                                     'mean_effect_size'  : 0.30, 
                                                                                                                                                                     'stability_score'   : 0.20, 
                                                                                                                                                                     'posterior_mass'    : 0.20,
                                                                                                                                                                    },
                                                                                                                                                     description = "Relative importance of metrics",
                                                                                                                                                    )

    # CROSS-MODEL AGREEMENT
    agreement_metrics        : List[Literal['temporal_consensus', 'boundary_density', 'directional_consistency']]                            = Field(default     = ['temporal_consensus', 'boundary_density'],
                                                                                                                                                     description = "Agreement metrics computed against all other models",
                                                                                                                                                    )

    agreement_weight         : float                                                                                                         = Field(default     = 0.25,
                                                                                                                                                     ge          = 0.0,
                                                                                                                                                     le          = 0.5,
                                                                                                                                                     description = "Bonus weight for agreement with global consensus",
                                                                                                                                                    )

    # SELECTION STRATEGY
    selection_strategy       : Literal['weighted_score', 'agreement_first', 'conservative']                                                  = Field(default     = 'agreement_first',
                                                                                                                                                     description = """
                                                                                                                                                                       weighted_score   : maximize total score
                                                                                                                                                                       agreement_first  : prioritize agreement, then score
                                                                                                                                                                       conservative     : fewer CPs + strong effects
                                                                                                                                                                   """,
                                                                                                                                                    )

    # TIE BREAKING (APPLIED IN ORDER)
    tie_breaking_rules       : List[Literal['higher_agreement', 'higher_effect_size', 'fewer_change_points', 'simpler_model']]               = Field(default     = ['higher_agreement', 'higher_effect_size', 'fewer_change_points'],
                                                                                                                                                     description = "Tie-breaking rules applied sequentially",
                                                                                                                                                    )

    # OUTPUT CONTROL
    save_full_ranking        : bool                                                                                                          = Field(default     = True,
                                                                                                                                                     description = "Persist ranked list of all models",
                                                                                                                                                    )

    save_only_best_model     : bool                                                                                                          = Field(default     = False,
                                                                                                                                                     description = "If True, only best model is promoted to final output",
                                                                                                                                                    )

    generate_explanations    : bool                                                                                                           = Field(default     = True,
                                                                                                                                                      description = "Generate human-readable explanation for selection",
                                                                                                                                                     )

    explanation_detail_level : Literal['brief', 'standard', 'full']                                                                           = Field(default     = 'full',
                                                                                                                                                      description = "Verbosity of selection explanation",
                                                                                                                                                     )


    
    # VALIDATION
    @validator('metric_weights')
    def validate_weights(cls, v):
        total = sum(v.values())
        
        if (total > 1.0):
            raise ValueError(f"Metric weights sum to {total:.2f} (> 1.0)")

        return v


    @validator('agreement_weight')
    def validate_agreement_weight(cls, v):
        if (v > 0.5):
            raise ValueError("agreement_weight > 0.5 risks overpowering evidence")

        return v