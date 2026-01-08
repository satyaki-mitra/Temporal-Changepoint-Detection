# Dependencies
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from dataclasses import dataclass
from collections import defaultdict
from config.model_selection_config import ModelSelectorConfig


# Result Schema
@dataclass
class ModelResult:
    """
    Canonical representation of a completed change-point model run

    All detector outputs (PELT / BOCPD variants) are mapped into this structure to allow fair comparison and explainable ranking
    """
    model_id              : str
    family                : str                       # 'pelt' | 'bocpd'
    change_points         : List[int]

    # Frequentist metrics (PELT)
    n_significant_cps     : int            = 0
    mean_effect_size      : float          = 0.0

    # Bayesian metrics (BOCPD)
    posterior_mass        : float          = 0.0
    cp_persistence        : float          = 0.0

    # Cross-model agreement
    stability_score       : float          = 0.0

    # Raw detector output (for traceability)
    raw_result            : Dict[str, Any] = None


# Result Adapters (Detector → ModelResult)
class ModelResultAdapter:
    """
    Convert detector-specific outputs into ModelResult
    """
    @staticmethod
    def from_pelt(model_id: str, result: Dict) -> ModelResult:
        """
        Adapt PELT detector output
        """
        validation = result.get('validation', {})
        summary    = validation.get('summary', {})

        cps        = result.get('change_points', [])
        
        if (cps and cps[-1] >= len(cps)):
            cps = cps[:-1]

        return ModelResult(model_id          = model_id,
                           family            = 'pelt',
                           change_points     = cps,
                           n_significant_cps = validation.get('n_significant', 0),
                           mean_effect_size  = summary.get('mean_effect_size', 0.0),
                           raw_result        = result,
                          )


    @staticmethod
    def from_bocpd(model_id: str, result: Dict) -> ModelResult:
        """
        Adapt BOCPD detector output
        """
        validation = result.get('validation', {})
        summary    = validation.get('summary', {})

        return ModelResult(model_id       = model_id,
                           family         = 'bocpd',
                           change_points  = validation.get('indices', []),
                           posterior_mass = summary.get('mean_posterior_at_cp', 0.0),
                           cp_persistence = summary.get('coverage_ratio', 0.0),
                           raw_result     = result,
                          )


# Metric Normalization
class MetricNormalizer:
    """
    Normalize metrics across all models using a specified strategy
    """
    @staticmethod
    def normalize(values: Dict[str, float], method: str) -> Dict[str, float]:
        keys = list(values.keys())
        vals = np.array(list(values.values()), dtype=float)

        if (len(vals) == 0):
            return values

        if (method == 'minmax'):
            min_v, max_v = np.min(vals), np.max(vals)

            if (max_v - min_v < 1e-9):
                return {k: 0.0 for k in keys}

            norm = (vals - min_v) / (max_v - min_v)

        elif (method == 'zscore'):
            std = np.std(vals)

            if (std < 1e-9):
                return {k: 0.0 for k in keys}

            norm = (vals - np.mean(vals)) / std

        elif (method == 'rank'):
            order       = vals.argsort()
            norm        = np.empty_like(order, dtype=float)
            norm[order] = np.linspace(0, 1, len(vals))

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return dict(zip(keys, norm))


# Agreement Computation
class AgreementComputer:
    """
    Compute cross-model agreement metrics
    """
    @staticmethod
    def temporal_consensus(model_cps : List[int], all_cps   : List[List[int]], tolerance : int = 2) -> float:
        """
        Fraction of model CPs that align with at least one other model.
        """
        if not model_cps:
            return 0.0

        matches = 0

        for cp in model_cps:
            for other in all_cps:
                if any(abs(cp - ocp) <= tolerance for ocp in other):
                    matches += 1
                    break

        return matches / len(model_cps)


    @staticmethod
    def boundary_density(model_cps : List[int], all_cps   : List[List[int]]) -> float:
        """
        Density of model CPs within the global CP pool
        """
        if not model_cps:
            return 0.0

        flat = [cp for cps in all_cps for cp in cps]

        if not flat:
            return 0.0

        return sum(cp in flat for cp in model_cps) / len(model_cps)


# Model Selector
class ModelSelector:
    """
    Select the best change-point model from a pool of completed PELT and BOCPD variants
    """
    def __init__(self, config: ModelSelectorConfig):
        """
        Initialize model selector
        """
        self.config = config


    def select(self, raw_results: Dict[str, Dict]) -> Dict:
        """
        Select the best model

        Arguments:
        ----------
            raw_results { dict } : Dictionary of completed detector results

        Returns:
        --------
                 { dict }        : Ranking, scores, and explanation
        """
        canonical_models  = self._canonicalize(raw_results = raw_results)

        self._compute_agreement(models = canonical_models)

        scores            = self._score_models(canonical_models)

        ranked_models     = sorted(canonical_models.values(),
                                   key     = lambda m: scores.get(m.model_id, 0.0),
                                   reverse = True,
                                  )

        explanation       = self._generate_explanation(ranked_models, scores)

        return {'best_model'  : ranked_models[0].model_id if ranked_models else None,
                'ranking'     : [m.model_id for m in ranked_models],
                'scores'      : scores,
                'explanation' : explanation,
               }


    # Internal Helpers
    def _canonicalize(self, raw_results: Dict[str, Dict]) -> Dict[str, ModelResult]:
        """
        Convert raw detector outputs into canonical representations
        """
        models = dict()

        for model_id, result in raw_results.items():
            method = result.get('method')

            if (method == 'pelt'):
                models[model_id] = ModelResultAdapter.from_pelt(model_id, result)

            elif (method == 'bocpd'):
                models[model_id] = ModelResultAdapter.from_bocpd(model_id, result)

        return models


    def _compute_agreement(self, models: Dict[str, ModelResult]):
        """
        Compute agreement-based stability scores
        """
        all_cps = [m.change_points for m in models.values()]

        for m in models.values():
            tc                = AgreementComputer.temporal_consensus(m.change_points, all_cps)
            bd                = AgreementComputer.boundary_density(m.change_points, all_cps)

            m.stability_score = 0.5 * tc + 0.5 * bd


    def _score_models(self, models: Dict[str, ModelResult]) -> Dict[str, float]:
        """
        Compute weighted scores for each model
        """
        scores = defaultdict(float)

        for metric, weight in self.config.metric_weights.items():
            raw_values = {m.model_id: getattr(m, metric, 0.0) for m in models.values()}

            normalized = MetricNormalizer.normalize(raw_values,
                                                    self.config.metric_normalization.get(metric, 'minmax'),
                                                   )

            for model_id, value in normalized.items():
                scores[model_id] += weight * value

        # Agreement bonus
        for m in models.values():
            scores[m.model_id] += self.config.agreement_weight * m.stability_score

        return dict(scores)


    def _generate_explanation(self, ranked_models: List[ModelResult], scores : Dict[str, float]) -> str:
        """
        Generate human-readable explanation for model selection
        """
        if not ranked_models:
            return "No valid models were available for selection."

        best = ranked_models[0]

        return (f"Selected model '{best.model_id}' ({best.family.upper()}) because it:\n"
                f"- achieved the highest overall score ({scores[best.model_id]:.3f})\n"
                f"- demonstrated strong cross-model agreement (stability={best.stability_score:.2f})\n"
                f"- balanced statistical strength with conservative change-point detection\n"
               )