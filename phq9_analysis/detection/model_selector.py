# Dependencies
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from dataclasses import dataclass
from collections import defaultdict
from config.model_selection_config import ModelSelectorConfig


# Canonical Model Result
@dataclass
class ModelResult:
    """
    Canonical representation of a completed change-point model run

    - All detector outputs (PELT / BOCPD variants) are mapped into this structure to enable fair comparison, scoring, and explainable selection
    """
    model_id          : str
    family            : str                    # 'pelt' | 'bocpd'
    change_points     : List[int]

    # Frequentist (PELT)
    n_significant_cps : int   = 0
    mean_effect_size  : float = 0.0

    # Bayesian (BOCPD)
    posterior_mass    : float = 0.0
    cp_persistence    : float = 0.0

    # Cross-model agreement
    stability_score   : float = 0.0

    # Traceability
    raw_result        : Dict[str, Any] = None


# Detector → Canonical Adapters
class ModelResultAdapter:
    """
    Convert detector-specific outputs into canonical ModelResult objects.
    """
    @staticmethod
    def from_pelt(model_id: str, result: Dict) -> ModelResult:
        validation = result.get('validation', {})
        summary    = validation.get('summary', {})
        cps        = list(result.get('change_points', []))

        return ModelResult(model_id          = model_id,
                           family            = 'pelt',
                           change_points     = cps,
                           n_significant_cps = validation.get('n_significant', 0),
                           mean_effect_size  = summary.get('mean_effect_size', 0.0),
                           raw_result        = result,
                          )


    @staticmethod
    def from_bocpd(model_id: str, result: Dict) -> ModelResult:
        validation = result.get('validation', {})
        summary    = validation.get('summary', {})

        return ModelResult(model_id        = model_id,
                           family          = 'bocpd',
                           change_points   = validation.get('indices', []),
                           posterior_mass  = summary.get('mean_posterior_at_cp', 0.0),
                           cp_persistence  = summary.get('coverage_ratio', 0.0),
                           raw_result      = result,
                          )


# Metric Normalization
class MetricNormalizer:
    """
    Normalize metrics across models using a chosen strategy
    """
    @staticmethod
    def normalize(values: Dict[str, float], method: str) -> Dict[str, float]:
        keys = list(values.keys())
        vals = np.array(list(values.values()), dtype=float)

        if (len(vals) == 0):
            return values

        if (method == 'minmax'):
            lo, hi = np.min(vals), np.max(vals)
            
            if ((hi - lo) < 1e-9):
                return {k: 0.0 for k in keys}

            norm = (vals - lo) / (hi - lo)

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


# Cross-Model Agreement
class AgreementComputer:
    """
    Compute cross-model agreement metrics
    """
    @staticmethod
    def temporal_consensus(model_cps: List[int], other_cps: List[List[int]], tolerance: int = 2) -> float:
        if not model_cps or not other_cps:
            return 0.0

        matches = 0
        for cp in model_cps:
            if any((abs(cp - ocp) <= tolerance) for cps in other_cps for ocp in cps):
                matches += 1

        return matches / len(model_cps)


    @staticmethod
    def boundary_density(model_cps: List[int], other_cps: List[List[int]]) -> float:
        if not model_cps:
            return 0.0

        flat = [cp for cps in other_cps for cp in cps]
        
        if not flat:
            return 0.0

        return sum(cp in flat for cp in model_cps) / len(model_cps)


# Model Selector
class ModelSelector:
    """
    Select the best change-point model using agreement-first strategy
    """
    def __init__(self, config: ModelSelectorConfig):
        self.config = config

        if self.config.selection_strategy != 'agreement_first':
            raise NotImplementedError("Only 'agreement_first' strategy is currently implemented. Other strategies are intentionally blocked to prevent silent misuse.")


    def select(self, raw_results: Dict[str, Dict]) -> Dict:
        models = self._canonicalize(raw_results)

        if not models:
            return {'best_model'  : None,
                    'ranking'     : [],
                    'scores'      : {},
                    'explanation' : "No eligible models after filtering.",
                   }

        self._compute_agreement(models = models)

        scores      = self._score_models(models = models)

        ranked      = sorted(models.values(),
                             key     = lambda m: scores.get(m.model_id, 0.0),
                             reverse = True,
                            )

        explanation = self._generate_explanation(ranked = ranked, 
                                                 scores = scores,
                                                )

        return {'best_model'  : ranked[0].model_id,
                'ranking'     : [m.model_id for m in ranked],
                'scores'      : scores,
                'explanation' : explanation,
               }


    def _canonicalize(self, raw_results: Dict[str, Dict]) -> Dict[str, ModelResult]:
        models = dict()

        for model_id, result in raw_results.items():
            family = result.get('method')

            if family not in self.config.allowed_families:
                continue

            if self.config.allowed_model_ids and model_id not in self.config.allowed_model_ids:
                continue

            if (family == 'pelt'):
                models[model_id] = ModelResultAdapter.from_pelt(model_id, result)

            elif (family == 'bocpd'):
                models[model_id] = ModelResultAdapter.from_bocpd(model_id, result)

        return models


    def _compute_agreement(self, models: Dict[str, ModelResult]):
        for m in models.values():
            others            = [o.change_points for o in models.values() if o.model_id != m.model_id]
            tc                = AgreementComputer.temporal_consensus(m.change_points, others)
            bd                = AgreementComputer.boundary_density(m.change_points, others)
            m.stability_score = 0.5 * tc + 0.5 * bd


    def _score_models(self, models: Dict[str, ModelResult]) -> Dict[str, float]:
        scores = defaultdict(float)

        for metric, weight in self.config.metric_weights.items():
            raw  = {m.model_id: getattr(m, metric, 0.0) for m in models.values()}
            norm = MetricNormalizer.normalize(raw, self.config.metric_normalization.get(metric, 'minmax'))

            for mid, val in norm.items():
                scores[mid] += weight * val

        for m in models.values():
            scores[m.model_id] += self.config.agreement_weight * m.stability_score

        return dict(scores)


    def _generate_explanation(self, ranked: List[ModelResult], scores: Dict[str, float]) -> str:
        best = ranked[0]

        if (best.family == 'pelt'):
            signal_line = f"meaningful effect size (d={best.mean_effect_size:.2f})"

        else:
            signal_line = f"strong posterior support (mass={best.posterior_mass:.2f})"

        return (f"Selected model '{best.model_id}' ({best.family.upper()}) because it:\n"
                f"- achieved the highest overall score ({scores[best.model_id]:.3f})\n"
                f"- showed strong cross-model agreement (stability={best.stability_score:.2f})\n"
                f"- demonstrated {signal_line}"
               )