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
    # Model Eligibility
    allowed_families         : List[Literal['pelt', 'bocpd']]                                                                                = Field(default     = ['pelt', 'bocpd'],
                                                                                                                                                     description = "Detector families eligible for selection",
                                                                                                                                                    )

    # Optional hard filter (exact model_ids)
    allowed_model_ids        : Optional[List[str]]                                                                                           = Field(default     = None,
                                                                                                                                                     description = "Explicit allow-list of model_ids (None = all)",
                                                                                                                                                    )

    # Metrics used for Scoring (Global)
    scoring_metrics          : List[Literal['n_significant_cps', 'mean_effect_size', 'stability_score', 'posterior_mass']]                   = Field(default     = ['n_significant_cps', 'mean_effect_size', 'stability_score', 'posterior_mass'],
                                                                                                                                                     description = "Metrics used to score each model",
                                                                                                                                                    )

    # Metric Normalization
    metric_normalization     : Dict[str, Literal['minmax', 'zscore', 'rank']]                                                                = Field(default     = {'n_significant_cps' : 'minmax',
                                                                                                                                                                    'mean_effect_size'  : 'minmax',
                                                                                                                                                                    'posterior_mass'    : 'minmax',
                                                                                                                                                                    'stability_score'   : 'minmax',
                                                                                                                                                                   },
                                                                                                                                                     description = "How each metric is normalized across models",
                                                                                                                                                    )

    # Metric Weights (sum ≤ 1.0)
    metric_weights           : Dict[str, float]                                                                                              = Field(default      = {'n_significant_cps' : 0.30,
                                                                                                                                                                     'mean_effect_size'  : 0.30, 
                                                                                                                                                                     'posterior_mass'    : 0.20,
                                                                                                                                                                     'stability_score'   : 0.20, 
                                                                                                                                                                    },
                                                                                                                                                     description = "Relative importance of metrics",
                                                                                                                                                    )

    # Cross-Model Agreement
    agreement_metrics        : List[Literal['temporal_consensus', 'boundary_density']]                                                       = Field(default     = ['temporal_consensus', 'boundary_density'],
                                                                                                                                                     description = "Agreement metrics computed against all other models",
                                                                                                                                                    )

    agreement_weight         : float                                                                                                         = Field(default     = 0.25,
                                                                                                                                                     ge          = 0.0,
                                                                                                                                                     le          = 0.5,
                                                                                                                                                     description = "Bonus weight for agreement with global consensus",
                                                                                                                                                    )

    # Model Selection Strategy
    selection_strategy       : Literal['weighted_score', 'agreement_first', 'conservative']                                                  = Field(default     = 'agreement_first',
                                                                                                                                                     description = (
                                                                                                                                                                    "weighted_score   : maximize total score\n"
                                                                                                                                                                    "agreement_first  : prioritize agreement, then score\n"
                                                                                                                                                                    "conservative     : fewer CPs + strong effects"
                                                                                                                                                                   ),
                                                                                                                                                    )

    # Tie Breaking (Apllied in exact order)
    tie_breaking_rules       : List[Literal['higher_agreement', 'higher_effect_size', 'fewer_change_points', 'simpler_model']]               = Field(default     = ['higher_agreement', 'higher_effect_size', 'fewer_change_points'],
                                                                                                                                                     description = "Tie-breaking rules applied sequentially",
                                                                                                                                                    )

    # Output Control
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
    @validator('allowed_model_ids')
    def validate_model_ids(cls, v):
        if v is not None and not v:
            raise ValueError("allowed_model_ids cannot be empty if provided")

        return v


    @validator('metric_weights')
    def validate_weights(cls, v, values):
        scoring = values.get('scoring_metrics', [])
        extra   = set(v.keys()) - set(scoring)

        if extra:
            raise ValueError(f"Metric weights defined for unused metrics: {extra}")

        total = sum(v.values())

        if (total > 1.0):
            raise ValueError(f"Metric weights sum to {total:.2f} (> 1.0)")

        return v

    
    @validator('metric_normalization')
    def validate_normalization(cls, v, values):
        scoring = values.get('scoring_metrics', [])
        extra   = set(v.keys()) - set(scoring)

        if extra:
            raise ValueError(f"Normalization defined for unused metrics: {extra}")

        return v


    class Config:
        validate_assignment = True
        extra               = 'forbid'