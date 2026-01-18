# Dependencies
import numpy as np
from typing import Any
from typing import Dict
from typing import List
from dataclasses import field
from dataclasses import dataclass
from collections import defaultdict
from config.model_selection_config import ModelSelectorConfig


# Canonical Model Result
@dataclass
class ModelResult:
    """
    Canonical representation of a completed change-point model run

    All detector outputs (PELT / BOCPD variants) are mapped into this structure to enable fair comparison, scoring, and explainable selection
    """
    model_id               : str
    family                 : str                      # 'pelt' | 'bocpd'
    change_points          : List[float]              # normalized [0, 1]

    # Frequentist (PELT)
    n_significant_cps      : int            = 0
    mean_effect_size       : float          = 0.0

    # Bayesian (BOCPD)
    posterior_mass         : float          = 0.0
    posterior_coverage     : float          = 0.0

    # Cross-model agreement
    stability_score        : float          = 0.0

    # Statistical significance flag
    is_statistically_valid : bool           = False

    # Traceability
    raw_result             : Dict[str, Any] = None



# Detector → Canonical Adapters
class ModelResultAdapter:
    """
    Convert detector-specific outputs into canonical ModelResult objects
    """
    @staticmethod
    def from_pelt(model_id: str, result: Dict) -> ModelResult:
        validation     = result.get('validation', {})
        summary        = validation.get('summary', {})
        cps            = list(result.get('change_points', []))
        signal_length  = max(result.get('signal_length', 1), 1)

        # Normalize change points to [0, 1]
        norm_cps       = [cp / signal_length for cp in cps]

        # Check statistical validity
        n_sig          = validation.get('n_significant', 0)
        is_valid       = (n_sig > 0) and validation.get('overall_significant', False)

        return ModelResult(model_id               = model_id,
                           family                 = 'pelt',
                           change_points          = norm_cps,
                           n_significant_cps      = n_sig,
                           mean_effect_size       = summary.get('mean_effect_size', 0.0),
                           is_statistically_valid = is_valid,
                           raw_result             = result,
                          )


    @staticmethod
    def from_bocpd(model_id: str, result: Dict) -> ModelResult:
        validation = result.get('validation', {})
        summary    = validation.get('summary', {})

        # Check statistical validity - use detected_indices if available
        detected_indices = validation.get('detected_indices', [])
        norm_positions   = validation.get('normalized_positions', [])
        
        is_valid         = validation.get('overall_significant', False) and len(detected_indices) > 0

        return ModelResult(model_id               = model_id,
                           family                 = 'bocpd',
                           change_points          = norm_positions,
                           posterior_mass         = summary.get('mean_posterior_at_cp', 0.0),
                           posterior_coverage     = summary.get('coverage_ratio', 0.0),
                           is_statistically_valid = is_valid,
                           raw_result             = result,
                          )


# Metric Normalization
class MetricNormalizer:
    """
    Normalize metrics across models using a chosen strategy
    """
    @staticmethod
    def normalize(values: Dict[str, float], method: str, robust: bool = True) -> Dict[str, float]:
        keys = list(values.keys())
        vals = np.array(list(values.values()), dtype=float)

        if (len(vals) == 0):
            return values

        # Robust normalization option
        if robust and (method == 'minmax'):
            # Use percentile-based normalization to handle outliers
            low  = np.percentile(vals, 5)
            high = np.percentile(vals, 95)

            if ((high - low) < 1e-9):
                return {k: 0.5 for k in keys}

            norm = np.clip((vals - low) / (high - low), 0.0, 1.0)

        elif (method == 'minmax'):
            low, high = np.min(vals), np.max(vals)

            if ((high - low) < 1e-9):
                return {k: 0.0 for k in keys}

            norm = (vals - low) / (high - low)

        elif (method == 'zscore'):
            std = np.std(vals)

            if (std < 1e-9):
                return {k: 0.0 for k in keys}

            norm = (vals - np.mean(vals)) / std

        elif (method == 'rank'):
            order       = vals.argsort()
            norm        = np.empty_like(order, dtype = float)
            norm[order] = np.linspace(0, 1, len(vals))

        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return dict(zip(keys, norm))


# Cross-Model Agreement
class AgreementComputer:
    """
    Compute cross-model agreement metrics (normalized CPs in [0, 1])
    """
    @staticmethod
    def temporal_consensus(model_cps: List[float], other_cps: List[List[float]], tolerance: float = 0.02) -> float:
        """
        Uses distance-weighted scoring instead of binary matching
        """
        if (not model_cps or not other_cps):
            return 0.0

        consensus_scores = list()

        for cp in model_cps:
            # Find minimum distance to any CP in other models
            distances = list()

            for cps in other_cps:
                if cps:
                    min_dist = min(abs(cp - ocp) for ocp in cps)
                    distances.append(min_dist)

            if distances:
                # Exponential decay: closer = higher weight
                min_distance = min(distances)
                weight       = np.exp(-(min_distance / tolerance) ** 2)

                consensus_scores.append(weight)

        return float(np.mean(consensus_scores)) if consensus_scores else 0.0


    @staticmethod
    def boundary_density(model_cps: List[float], other_cps: List[List[float]], bandwidth: float = 0.02) -> float:
        """
        Kernel density-based agreement
        """
        if not model_cps:
            return 0.0

        flat = [cp for cps in other_cps for cp in cps]

        if not flat:
            return 0.0

        densities = list()

        for cp in model_cps:
            # Gaussian kernel density
            density = sum(np.exp(-((cp - fcp) / bandwidth) ** 2) for fcp in flat)
            
            densities.append(density)

        # Normalize by number of other CPs
        return float(np.mean(densities)) / max(len(flat), 1)



# Model Selector
class ModelSelector:
    """
    Select the best change-point model using agreement-first strategy
    """
    def __init__(self, config: ModelSelectorConfig):
        self.config = config

        if (self.config.selection_strategy != 'agreement_first'):
            raise NotImplementedError("Only 'agreement_first' strategy is currently implemented. Other strategies are intentionally blocked to prevent silent misuse.")


    def select(self, raw_results: Dict[str, Dict]) -> Dict:
        """
        Select best model via user-defined selection strategy
        """
        models = self._canonicalize(raw_results)

        if not models:
            return {'best_model'  : None,
                    'ranking'     : [],
                    'scores'      : {},
                    'explanation' : "No eligible models after filtering",
                   }

        # Filter out statistically invalid models
        valid_models = {mid: m for mid, m in models.items() if m.is_statistically_valid}

        if not valid_models:
            return {'best_model'  : None,
                    'ranking'     : [],
                    'scores'      : {},
                    'explanation' : f"No models passed statistical validation (total models: {len(models)})",
                    'n_candidates': len(models),
                    'n_valid'     : 0,
                   }

        if (len(valid_models) == 1):
            mid = next(iter(valid_models))
            
            return {'best_model'  : mid,
                    'ranking'     : [mid],
                    'scores'      : {mid: 1.0},
                    'explanation' : f"Only one statistically valid model: {mid}",
                    'n_candidates': len(models),
                    'n_valid'     : 1,
                   }

        # Use valid models for scoring
        self._compute_agreement(models = valid_models)

        scores      = self._score_models(models = valid_models)

        # Tie-breaking with tolerance
        ranked      = self._rank_with_tie_breaking(models = valid_models,
                                                   scores = scores,
                                                  )

        explanation = self._generate_explanation(ranked = ranked,
                                                 scores = scores,
                                                )

        return {'best_model'  : ranked[0].model_id,
                'ranking'     : [m.model_id for m in ranked],
                'scores'      : scores,
                'explanation' : explanation,
                'n_candidates': len(models),
                'n_valid'     : len(valid_models),
               }


    def _canonicalize(self, raw_results: Dict[str, Dict]) -> Dict[str, ModelResult]:
        """
        Canonicalize the model results for fair and traceable comparison
        """
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
        """
        Compute cross-model change-points' agreements
        """
        for m in models.values():
            others            = [o.change_points for o in models.values() if (o.model_id != m.model_id)]

            tc                = AgreementComputer.temporal_consensus(m.change_points, others)
            bd                = AgreementComputer.boundary_density(m.change_points, others)

            m.stability_score = 0.5 * tc + 0.5 * bd


    def _score_models(self, models: Dict[str, ModelResult]) -> Dict[str, float]:
        """
        Score the models based on their detections
        """
        scores = defaultdict(float)

        for metric in self.config.scoring_metrics:
            weight = self.config.metric_weights.get(metric, 0.0)

            if (weight == 0.0):
                continue

            raw  = {m.model_id: getattr(m, metric, 0.0) for m in models.values()}
            norm = MetricNormalizer.normalize(raw, self.config.metric_normalization.get(metric, 'minmax'), robust=True)

            for mid, val in norm.items():
                scores[mid] += weight * val

        for m in models.values():
            scores[m.model_id] += self.config.agreement_weight * m.stability_score

        return dict(scores)


    def _rank_with_tie_breaking(self, models: Dict[str, ModelResult], scores: Dict[str, float], tolerance: float = 0.01) -> List[ModelResult]:
        """
        Rank with tie-breaking tolerance
        """
        ranked = sorted(models.values(),
                       key     = lambda m: scores.get(m.model_id, 0.0),
                       reverse = True,
                      )

        # Apply tie-breaking if top scores are within tolerance
        if (len(ranked) >= 2):
            top_score    = scores[ranked[0].model_id]
            second_score = scores[ranked[1].model_id]

            if (abs(top_score - second_score) < tolerance):
                # Scores are tied - apply tie-breaking rules
                tied = [m for m in ranked if abs(scores[m.model_id] - top_score) < tolerance]

                if (len(tied) > 1):
                    tied = self._apply_tie_breaking(tied)
                    
                    # Replace tied models at top with sorted tied list
                    ranked = tied + [m for m in ranked if m not in tied]

        return ranked


    def _apply_tie_breaking(self, tied_models: List[ModelResult]) -> List[ModelResult]:
        """
        Apply sequential tie-breaking rules
        """
        for rule in self.config.tie_breaking_rules:
            if (len(tied_models) == 1):
                break

            if (rule == 'higher_agreement'):
                tied_models = sorted(tied_models, key=lambda m: m.stability_score, reverse=True)

            elif (rule == 'higher_effect_size'):
                tied_models = sorted(tied_models, key=lambda m: abs(m.mean_effect_size), reverse=True)

            elif (rule == 'fewer_change_points'):
                tied_models = sorted(tied_models, key=lambda m: len(m.change_points))

            elif (rule == 'simpler_model'):
                # PELT is simpler than BOCPD
                tied_models = sorted(tied_models, key=lambda m: 0 if m.family == 'pelt' else 1)

        return tied_models


    def _generate_explanation(self, ranked: List[ModelResult], scores: Dict[str, float]) -> str:
        """
        Generate explanations for the model selection
        """
        best = ranked[0]

        if (best.family == 'pelt'):
            signal_line = f"meaningful effect size (d={best.mean_effect_size:.2f}, n_significant={best.n_significant_cps})"

        else:
            signal_line = f"strong posterior support (mass={best.posterior_mass:.2f}, coverage={best.posterior_coverage:.2%})"

        # More detailed explanation
        explanation = (f"Selected model '{best.model_id}' ({best.family.upper()}) because it:\n"
                       f"- Achieved the highest overall score ({scores[best.model_id]:.3f})\n"
                       f"- Showed strong cross-model agreement (stability={best.stability_score:.2f})\n"
                       f"- Demonstrated {signal_line}\n"
                       f"- Detected {len(best.change_points)} change points at positions: {[f'{cp:.2%}' for cp in best.change_points[:5]]}"
                      )

        if (len(ranked) > 1):
            second       = ranked[1]
            gap          = scores[best.model_id] - scores[second.model_id]
            explanation += f"\n- Outperformed runner-up '{second.model_id}' by {gap:.3f} points"

        return explanation