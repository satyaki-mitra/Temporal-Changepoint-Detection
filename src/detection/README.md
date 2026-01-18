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
- **Complete results persistence**: Structured directory output with all artifacts
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


### 2. BOCPD (Bayesian Online Change Point Detection)

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
    --dataset exponential \
    --auto-tune-penalty
```

### Side-by-Side Comparison (Default)

```bash
python scripts/run_detection.py \
    --execution-mode compare \
    --detectors pelt bocpd \
    --dataset exponential
```

### With Model Selection

```bash
python scripts/run_detection.py \
    --dataset exponential \
    --select-model
```

### Run All Datasets

```bash
python scripts/run_detection.py \
    --all-datasets \
    --select-model
```

### Custom Parameters

```bash
python scripts/run_detection.py \
    --dataset exponential \
    --hazard-lambda 75.0 \
    --posterior-threshold 0.15 \
    --bocpd-persistence 1 \
    --select-model
```

---

## 📁 Output Structure

```plaintext
results/detection_{dataset}/
├── all_model_results.json              # Complete results (all models)
├── per_model/                          # Individual model JSONs
│   ├── pelt_l1.json
│   ├── pelt_l2.json
│   ├── pelt_rbf.json
│   ├── pelt_ar.json
│   ├── bocpd_gaussian_heuristic.json
│   └── bocpd_gaussian_predictive_ll.json
├── change_points/                      # Change point CSVs
│   ├── pelt_l1_changepoints.csv
│   ├── bocpd_gaussian_heuristic_changepoints.csv
│   └── all_changepoints_comparison.csv # Cross-model comparison
├── statistical_tests/                  # Statistical validation
│   ├── pelt_l1_tests.json
│   ├── bocpd_gaussian_heuristic_tests.json
│   └── statistical_summary.csv
├── diagnostics/                        # Segment diagnostics
│   ├── pelt_l1_segments.png
│   ├── bocpd_gaussian_heuristic_segments.png
│   └── detection_summary.json
├── plots/                              # Visualizations
│   ├── aggregated_cv_all_models.png    # Overlay of all models
│   ├── model_comparison_grid.png       # Side-by-side comparison
│   ├── bocpd_gaussian_heuristic_posterior.png
│   └── bocpd_gaussian_predictive_ll_posterior.png
├── model_selection.json                # Selection results (if enabled)
└── best_model/                         # Model selection results
    ├── metadata.json                   # Selection explanation
    └── model_result.json               # Best model details
```

### What Each Directory Contains

#### **`per_model/`**
Individual JSON files with:
- Model configuration
- Detected change points (indices and normalized positions)
- Validation results
- Method-specific metadata

#### **`change_points/`**
CSV files for analysis:
- Per-model change points
- `all_changepoints_comparison.csv` - All models in one table

#### **`statistical_tests/`**
Statistical validation:
- Detailed test results (p-values, effect sizes)
- `statistical_summary.csv` - Cross-model summary

#### **`diagnostics/`**
Segment-level analysis:
- Segment means and residuals (PNG)
- `detection_summary.json` - Overall summary

#### **`plots/`**
Visual comparisons:
- All models overlaid on signal
- Side-by-side subplots
- BOCPD posterior heatmaps

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

### 2. BOCPD Parameters (Updated Defaults)

```python
# Hazard control
hazard_lambda          = 75.0           # Expected run length (FIXED: was 30.0)
auto_tune_hazard       = True          # Automatic tuning
hazard_tuning_method   = 'heuristic'   # 'heuristic' or 'predictive_ll'
hazard_range           = (30.0, 300.0) # FIXED: minimum raised to 30

# Detection thresholds
cp_posterior_threshold = 0.15           # FIXED: lowered from 0.6
bocpd_persistence      = 1              # FIXED: reduced from 3
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

## 🔍 Mathematical Framework

### PELT Cost Function

**Objective**: Minimize total cost + penalty for each change point

```
Cost(τ) = Σ[C(y_{τ_i:τ_{i+1}})] + β × K
```

- `C(·)`: Segment cost (L1, L2, RBF, AR)
- `β`: Penalty parameter (tuned via BIC)
- `K`: Number of change points

---

### BOCPD Posterior Update (FIXED)

**Run-length posterior** at time `t`:

```
P(r_t | y_{1:t}) ∝ P(y_t | r_{t-1}, y_{1:t-1}) × [
    P(r_t = 0) × Σ P(r_{t-1})           (change point)
    P(r_t = r_{t-1} + 1 | r_{t-1}) × P(r_{t-1})  (growth)
]
```

**Change point posterior** (CRITICAL FIX):
```python
# BEFORE (WRONG):
cp_posterior[t] = np.exp(log_R[t, 0])

# AFTER (CORRECT):
cp_posterior[t] = np.exp(log_cp - log_norm)
```

This fix ensures the CP posterior represents the **probability of transitioning to r=0**, not just the probability mass at r=0.

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

## 📊 Statistical Validation

### PELT (Frequentist)

**Test selection**:
1. **Mann-Whitney U** (default, n ≥ 10): Non-parametric, robust
2. **T-test** (n ≥ 30): Large sample, CLT applies
3. **Permutation test** (n < 10): Exact, no assumptions

**Effect size**: Cohen's d with threshold = 0.3

**Multiple testing correction**: FDR (Benjamini-Hochberg) by default

### BOCPD (Bayesian) - FIXED

**Validation criteria** (improved):
1. Posterior probability > threshold (default: 0.15, lowered from 0.6)
2. Persistence filter: ≥ 1 consecutive timesteps (reduced from 3)
3. **Peak detection**: Marks CP at the peak of persistent regions

**Fixed validation logic**:
- Now correctly identifies runs of consecutive exceedances
- Marks change point at maximum posterior within each run
- Better rejection reason messages

---

## 🧬 Model Selection

### Canonical Result Format

All detectors output a **unified result structure**:

```python
{
    "method": "pelt" | "bocpd",
    "variant": "pelt_l1" | "bocpd_gaussian_heuristic",
    "change_points": [45, 120, 210],       # PELT: indices, BOCPD: normalized
    "n_changepoints": 3,
    "validation": {
        "n_significant": 2,
        "overall_significant": True,
        "summary": {
            "mean_effect_size": 0.65,      # PELT only
            "mean_posterior_at_cp": 0.82,  # BOCPD only
        }
    }
}
```

### Selection Strategy

**Agreement-first approach**:
1. Compute cross-model temporal consensus
2. Weight models by agreement + internal metrics
3. Apply tie-breaking rules

**Tie-breaking order**:
1. Higher cross-model agreement
2. Higher effect size (PELT) / posterior mass (BOCPD)
3. Fewer change points (parsimony)
4. Simpler model (PELT preferred over BOCPD)

---

## 🐛 Troubleshooting

### BOCPD detects 0 change points

**Check diagnostics**:
```
BOCPD diagnostics:
  Max CP posterior: 0.0847
  Threshold: 0.15
  Points above threshold: 0
```

**Solutions**:
1. Lower threshold: `--posterior-threshold 0.10`
2. Adjust hazard lambda: `--hazard-lambda 60`
3. Reduce persistence: `--bocpd-persistence 1`

---

## 📚 Command Reference

### Basic Usage

```bash
# Single dataset with defaults
python scripts/run_detection.py --dataset exponential

# All datasets
python scripts/run_detection.py --all-datasets

# With model selection
python scripts/run_detection.py --dataset exponential --select-model
```

### Advanced Usage

```bash
# PELT only, custom parameters
python scripts/run_detection.py \
    --dataset exponential \
    --execution-mode single \
    --detectors pelt \
    --penalty 0.5 \
    --min-size 7

# BOCPD only, manual tuning
python scripts/run_detection.py \
    --dataset exponential \
    --execution-mode single \
    --detectors bocpd \
    --hazard-lambda 60.0 \
    --posterior-threshold 0.12 \
    --bocpd-persistence 1

# Custom data path
python scripts/run_detection.py \
    --data path/to/custom_data.csv \
    --output-dir results/custom_analysis
```

### Verification

```bash
# Check results completeness
python scripts/verify_results_structure.py results/detection_exponential

# Expected output:
# ✓ Passed: 45+
# ⚠ Warnings: 0
# ❌ Failed: 0
```

---

## 📖 References

### Core Algorithms
- **PELT**: Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*.
- **BOCPD**: Adams, R. P., & MacKay, D. J. (2007). Bayesian online changepoint detection. *arXiv preprint arXiv:0710.3742*.

### Statistical Methods
- **BIC**: Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*.
- **Mann-Whitney U**: Mann, H. B., & Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger than the other. *Annals of Mathematical Statistics*.
- **FDR**: Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society: Series B*.

### Clinical Context
- **PHQ-9**: Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001). The PHQ-9: validity of a brief depression severity measure. *Journal of General Internal Medicine*.

---
