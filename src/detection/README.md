# PHQ-9 Change Point Detection Module

## 🎯 Overview

This module provides **change point detection** for longitudinal PHQ-9 datasets. It implements two complementary algorithms with automated parameter tuning, statistical validation, and model selection.

**Scope**: Detect significant shifts in aggregated PHQ-9 statistics (e.g., coefficient of variation) that indicate population-level changes in depression symptom patterns.

---

## ✅ Key Capabilities

- **Dual-algorithm support**: PELT (offline, frequentist) and BOCPD (online, Bayesian)
- **Automated parameter tuning**: BIC-based penalty selection (PELT), cross-validated hazard tuning (BOCPD)
- **Rigorous statistical validation**: Hypothesis testing with multiple testing correction, effect size quantification
- **Model selection framework**: Cross-model agreement metrics, weighted composite scoring
- **Production-ready**: Config-driven, extensible, fully logged

---

## 🧪 Algorithms

### 1. PELT (Pruned Exact Linear Time)

| Aspect | Details |
|--------|---------|
| **Type** | Offline, frequentist |
| **Cost functions** | L1 (robust), L2 (least squares), RBF (nonlinear), AR (autoregressive) |
| **Penalty tuning** | Bayesian Information Criterion (BIC) |
| **Validation** | Mann-Whitney U test, Cohen's d effect size, FDR correction |
| **Complexity** | O(n) with pruning |
| **Reference** | Killick et al. (2012) |

**When to use**: Known study period, need interpretable frequentist statistics, offline analysis.


### 2.BOCPD (Bayesian Online Change Point Detection)

| Aspect | Details |
|--------|---------|
| **Type** | Online, Bayesian |
| **Likelihood** | Gaussian (Student-t planned) |
| **Hazard tuning** | Heuristic (fast) or predictive likelihood (accurate) |
| **Validation** | Posterior probability threshold + persistence filter |
| **Complexity** | O(n × r) where r = max run length |
| **Reference** | Adams & MacKay (2007) |

**When to use**: Real-time monitoring, streaming data, Bayesian inference preferred.

---

## 🚀 Quick Start

### Single Detector (PELT with auto-tuning)

```bash
python scripts/run_detection.py \
    --execution-mode single \
    --detectors pelt \
    --data data/processed/synthetic_phq9_aggregated.csv \
    --auto-tune-penalty
```

### Side-by-Side Comparison (Default)

```bash
python scripts/run_detection.py \
    --execution-mode compare \
    --detectors pelt bocpd \
    --data data/processed/synthetic_phq9_aggregated.csv
```

### Ensemble with Model Selection

```bash
python scripts/run_detection.py \
    --execution-mode ensemble \
    --detectors pelt bocpd \
    --data data/processed/synthetic_phq9_aggregated.csv
```

---

## 📐 Mathematical Framework

### PELT Cost Function

**Objective**: Minimize total cost + penalty for each change point

```
Cost(τ) = Σ[C(y_{τ_i:τ_{i+1}})] + β × K
```

- `C(·)`: Segment cost (L1, L2, RBF, AR)
- `β`: Penalty parameter (tuned via BIC)
- `K`: Number of change points

**BIC Formula** (FIXED in this version):
```
BIC = n × log(σ²) + p × log(n)
where p = 2 × n_segments (mean + variance per segment)
```

---

### BOCPD Posterior Update

**Run-length posterior** at time `t`:

```
P(r_t | y_{1:t}) ∝ P(y_t | r_{t-1}, y_{1:t-1}) × [
    P(r_t = 0) × Σ P(r_{t-1})           (change point)
    P(r_t = r_{t-1} + 1 | r_{t-1}) × P(r_{t-1})  (growth)
]
```

**Hazard function** (constant):
```
H(τ) = 1 / λ
where λ = expected run length
```

**FIXED ISSUES** in this version:
- ✅ Corrected log-space normalization
- ✅ Fixed sufficient statistics tracking (Welford's algorithm)
- ✅ Added numerical stability checks

---

## 🔧 Configuration

### 1. PELT Parameters

```python
# Penalty control
penalty              = 0.5                        # Fixed penalty (if auto_tune=False)
auto_tune_penalty    = True                       # BIC-based tuning
penalty_range        = (0.1, 10.0)                # Tuning search range

# Segmentation constraints
minimum_segment_size = 5                          # Minimum days per segment
jump                 = 1                          # Subsampling (1 = no subsampling)

# Cost function
pelt_cost_models     = ['l1', 'l2', 'rbf', 'ar']  # All variants tested
```

### 2. BOCPD Parameters

```python
# Hazard control
hazard_lambda          = 30.0          # Expected run length (days)
auto_tune_hazard       = True          # Automatic tuning
hazard_tuning_method   = 'heuristic'   # 'heuristic' or 'predictive_ll'
hazard_range           = (10.0, 300.0)

# Detection thresholds
cp_posterior_threshold = 0.6           # P(change point) threshold
bocpd_persistence      = 3             # Consecutive timesteps required
posterior_smoothing    = 3             # Gaussian smoothing (σ)

# Computational
max_run_length         = 500           # Maximum tracked run length
```


### 3. Statistical Testing

```python
alpha                       = 0.05       # Significance level
multiple_testing_correction = 'fdr_bh'   # 'bonferroni', 'fdr_bh', 'none'
effect_size_threshold       = 0.3        # Minimum Cohen's d
```

---

## 📊 Output Structure

```plaintext
results/detection/
├── all_model_results.json           # Complete outputs from all detectors
├── model_selection.json              # Selection results (if enabled)
├── best_model/
│   ├── model_result.json             # Best model's canonical result
│   ├── metadata.json                 # Selection metadata
│   └── *.png                         # Plots for best model
├── per_model/
│   ├── pelt_l1/
│   │   ├── change_points.json
│   │   ├── segments.json
│   │   ├── validation.json
│   │   └── tuning_results.json
│   └── bocpd_gaussian_heuristic/
│       ├── change_points.json
│       ├── hazard_tuning.json
│       └── validation.json
└── plots/
    ├── aggregated_cv_all_models.png  # Overlay of all detections
    ├── model_comparison_grid.png     # Side-by-side subplots
    └── bocpd_*_posterior.png         # BOCPD diagnostics
```

---

## 🧬 Model Selection

### Canonical Result Format

All detectors output a **unified result structure**:

```python
{
    "method": "pelt" | "bocpd",
    "variant": "pelt_l1" | "bocpd_gaussian_heuristic",
    "change_points": [45, 120, 210],       # Absolute indices
    "n_changepoints": 3,
    "validation": {
        "n_significant": 2,
        "overall_significant": True,
        "summary": {
            "mean_effect_size": 0.65,      # PELT only
            "mean_posterior_at_cp": 0.82,  # BOCPD only
        }
    },
    "tuning_results": {...}
}
```

### Selection Metrics

| Metric | PELT | BOCPD | Normalization |
|--------|------|-------|---------------|
| `n_significant_cps` | ✓ | — | MinMax |
| `mean_effect_size` | ✓ | — | MinMax |
| `posterior_mass` | — | ✓ | MinMax |
| `stability_score` | ✓ | ✓ | MinMax |

**Agreement metrics** (cross-model):
- **Temporal consensus**: % of change points within 2% tolerance of others
- **Boundary density**: Clustering of change points across models

### Composite Score

```
score = Σ(weight_i × normalized_metric_i) + agreement_weight × stability_score

Default weights:
- n_significant_cps: 0.30
- mean_effect_size:  0.30
- posterior_mass:    0.20
- agreement:         0.25
```

---

## 🎨 Visualizations

### 1. Aggregated CV with All Models
- Base signal (aggregated CV)
- Change points from all detectors overlaid
- Color-coded by detector family
- Saved: `plots/aggregated_cv_all_models.png`

### 2. Model Comparison Grid
- Side-by-side subplots for each detector variant
- Identical x-axis for fair comparison
- Change point count in titles
- Saved: `plots/model_comparison_grid.png`

### 3. BOCPD Diagnostics
- Run-length posterior heatmap
- Change point posterior probability time series
- Threshold line
- Saved: `plots/bocpd_*_posterior.png`

---

## 🔍 Statistical Validation

### PELT (Frequentist)

**Hypothesis test**: Do segment means differ significantly?

**Test selection** (FIXED in this version):
1. **Mann-Whitney U** (default, n ≥ 10): Non-parametric, robust
2. **T-test** (n ≥ 30): Large sample, CLT applies
3. **Permutation test** (n < 10): Exact, no assumptions

**Effect size**: Cohen's d with threshold = 0.3 (small-to-medium effect)

**Multiple testing correction**:
- **FDR (Benjamini-Hochberg)**: Default, controls false discovery rate
- **Bonferroni**: Conservative, controls family-wise error rate
- **None**: Use with caution

### BOCPD (Bayesian)

**Validation criteria**:
1. Posterior probability > threshold (default: 0.6)
2. Persistence filter: ≥ 3 consecutive timesteps

**Interpretation**:
- `P(change point) = 0.82` → Strong evidence for regime shift
- `coverage_ratio = 0.15` → 15% of timesteps have high CP probability

---

## 🧪 Ground Truth Evaluation

**Use case**: Validate detectors on synthetic data with known change points.

```python
from src.detection.ground_truth_evaluator import GroundTruthEvaluator

evaluator = GroundTruthEvaluator(true_changepoints = [50, 150, 250], 
                                 tolerance         = 5,
                                )

metrics   = evaluator.evaluate(detected_changepoints = [48, 155, 248])

# {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'hausdorff_distance': 5.0}
```

---

## ⚙️ Advanced Usage

### Custom Cost Function (PELT)

```python
from src.detection.pelt_detector import PELTDetector
from config.detection_config import ChangePointDetectionConfig


config   = ChangePointDetectionConfig(pelt_cost_models = ['ar'])
detector = PELTDetector(config     = config, 
                        cost_model = 'ar',
                       )

result   = detector.detect(signal = aggregated_signal)
```

### Custom Hazard Function (BOCPD)

```python
from src.detection.bocpd_detector import BOCPDDetector

# Override hazard lambda
detector = BOCPDDetector(config = config)
result   = detector.detect(signal        = signal, 
                           hazard_lambda = 50.0,
                          )
```

### Ensemble Combination

```python
from src.detection.model_selector import ModelSelector
from config.model_selection_config import ModelSelectorConfig

selector_config = ModelSelectorConfig(agreement_weight = 0.40,  # Prioritize consensus
                                      metric_weights   = {'n_significant_cps' : 0.25,
                                                          'mean_effect_size'  : 0.35,
                                                         }
                                     )

selector        = ModelSelector(selector_config)
selection       = selector.select(all_model_results)
```

---

## 📚 References

### Core Algorithms
- **PELT**: Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*.
- **BOCPD**: Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection. *arXiv preprint arXiv:0710.3742*.

### Statistical Methods
- **BIC**: Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*.
- **Mann-Whitney U**: Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastic ally larger than the other. *Annals of Mathematical Statistics*.
- **FDR**: Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society: Series B*.

### Clinical Context
- **PHQ-9**: Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001). The PHQ-9: validity of a brief depression severity measure. *Journal of General Internal Medicine*.

---

## 🔮 Future Enhancements

- **Multivariate change point detection**: Detect shifts across multiple aggregated metrics simultaneously
- **Adaptive hazard functions**: Time-varying hazard rates for BOCPD
- **Confidence intervals**: Bootstrap CIs for PELT, credible intervals for BOCPD
- **Real-time dashboard**: Live monitoring with streaming BOCPD
- **Student-t likelihood**: More robust to outliers than Gaussian

---

## 🙋 Troubleshooting

### "Signal too short for PELT"
**Cause**: Signal length < 2 × minimum_segment_size
**Fix**: Reduce `minimum_segment_size` or aggregate over longer periods

### "Penalty tuning failed: all values caused PELT failure"
**Cause**: Penalty range inappropriate for signal characteristics
**Fix**: Adjust `penalty_range`, try different cost function

### "BOCPD detects no change points"
**Cause**: `cp_posterior_threshold` too high or signal has no shifts
**Fix**: Lower threshold (e.g., 0.4), check signal visually

### "Model selection returns None"
**Cause**: No models passed structural validation
**Fix**: Review rejection reasons in validation reports

---

*This module provides the foundation for rigorous, reproducible change point detection in longitudinal mental health research.*