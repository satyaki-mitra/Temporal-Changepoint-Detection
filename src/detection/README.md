# PHQ-9 Change Point Detection Module

## Overview

PELT-based change point detection with BIC optimization and statistical validation.

### Key Features

✅ **BIC-Based Penalty Tuning** - Automatic hyperparameter optimization  
✅ **Multiple Cost Functions** - L1 (robust), L2, RBF, AR  
✅ **Fixed Validation Logic** - Tests BEFORE vs AFTER (not adjacent segments)  
✅ **Multiple Testing Correction** - Bonferroni, FDR (Benjamini-Hochberg)  
✅ **Effect Size Calculation** - Cohen's d for clinical significance  

---

## Quick Start

### Basic Usage

```python
from config.detection_config import ChangePointDetectionConfig
from phq9_analysis.detection import ChangePointDetector

# Create configuration
config = ChangePointDetectionConfig(
    auto_tune_penalty=True,
    cost_model='l1',
    minimum_segment_size=5
)

# Run detection
detector = ChangePointDetector(config)
detector.load_data("data/raw/synthetic_phq9_data.csv")
results = detector.run_full_detection()

# Access results
print(f"Change points: {results['change_points']}")
print(f"Significant: {results['validation_results']['n_significant']}")
```

### Command-Line Usage

```bash
# Full analysis with auto-tuning
python scripts/run_detection.py --auto-tune

# Manual penalty
python scripts/run_detection.py --penalty 0.5

# Different cost function
python scripts/run_detection.py --auto-tune --cost-model l2

# Custom significance level
python scripts/run_detection.py --alpha 0.01 --correction bonferroni

# Custom data and output
python scripts/run_detection.py \\
    --data path/to/data.csv \\
    --output-dir results/my_detection \\
    --auto-tune
```

---

## PELT Algorithm

### Overview

Pruned Exact Linear Time (PELT) algorithm efficiently detects optimal change points using dynamic programming.

**Model:**
```
Y_t = μ_t + ε_t

where μ_t changes at unknown time points (change points)
```

**Objective:**
Minimize: Cost + Penalty × (Number of change points)

**Cost Functions:**
- **L1** (default): Robust to outliers, focuses on median
- **L2**: Assumes Gaussian distribution
- **RBF**: Kernel-based, non-parametric
- **AR**: Autoregressive model

---

## BIC-Based Penalty Tuning

### Problem

Manual penalty selection is difficult and subjective.

### Solution

Bayesian Information Criterion (BIC) automatically selects penalty that balances fit and complexity:

```
BIC = n·log(σ²) + K·log(n)

where:
- n = number of observations
- σ² = residual variance
- K = number of change points
```

**Lower BIC = Better model**

### Usage

```python
config = ChangePointDetectionConfig(auto_tune_penalty=True)
detector = ChangePointDetector(config)
results = detector.run_full_detection()

# View tuning results
print(f"Optimal penalty: {results['tuning_results']['optimal_penalty']}")
print(f"Optimal K: {results['tuning_results']['optimal_n_changepoints']}")
```

---

## Statistical Validation (FIXED)

### Critical Fix

**Original (WRONG):**
- Tested adjacent segments: Seg1 vs Seg2, Seg2 vs Seg3
- Missed overall change point significance

**Fixed (CORRECT):**
- Tests BEFORE vs AFTER each change point
- Proper hypothesis testing for distributional shifts

### Example

```
Data: [10, 10, 10, | 20, 20, 20]
Change point at index 3

Original: Compare [10,10,10] vs [20,20,20] ✓ (correct by accident)
BUT with 2 CPs: [10,10|15,15|20,20]
Original: Compare [10,10] vs [15,15] AND [15,15] vs [20,20]
Fixed: Compare [10,10] vs [15,15,20,20] AND [10,10,15,15] vs [20,20]
```

### Statistical Tests

1. **Wilcoxon Rank-Sum** (Primary)
   - Non-parametric
   - No normality assumption
   - Robust to outliers

2. **Independent t-test** (Fallback)
   - Used when data has constant values
   - Or sample size < 5

3. **Effect Size (Cohen's d)**
   - Small: |d| ≥ 0.3
   - Medium: |d| ≥ 0.5
   - Large: |d| ≥ 0.8

---

## Multiple Testing Correction

When testing multiple change points, correction prevents false positives:

### Methods

1. **FDR (Benjamini-Hochberg)** - Default, recommended
   - Controls False Discovery Rate
   - More powerful than Bonferroni
   - Good for exploratory analysis

2. **Bonferroni**
   - Controls Family-Wise Error Rate
   - Very conservative
   - Use for confirmatory analysis

3. **None**
   - No correction
   - Only use with single change point

### Example

```python
# With FDR correction (recommended)
config = ChangePointDetectionConfig(
    multiple_testing_correction='fdr_bh',
    alpha=0.05
)

# Bonferroni (conservative)
config = ChangePointDetectionConfig(
    multiple_testing_correction='bonferroni',
    alpha=0.05
)
```

---

## Configuration

### Full Example

```python
from config.detection_config import ChangePointDetectionConfig

config = ChangePointDetectionConfig(
    # Input/Output
    data_path=Path('data/raw/synthetic_phq9_data.csv'),
    results_base_directory=Path('results/detection'),
    
    # PELT Algorithm
    cost_model='l1',  # 'l1', 'l2', 'rbf', 'ar'
    penalty=0.5,
    auto_tune_penalty=True,
    penalty_range=(0.1, 10.0),
    minimum_segment_size=5,
    jump=1,
    
    # Statistical Testing
    alpha=0.05,
    multiple_testing_correction='fdr_bh',
    effect_size_threshold=0.3,
    
    # Visualization
    smoothing_window_size=10,
    figure_size=(15, 10),
    dpi=300,
)
```

---

## Output Files

### Directory Structure

```
results/detection/
├── aggregated_cv_data.csv              # Daily CV values
├── change_points/
│   ├── analysis_results.json           # Complete results
│   └── cluster_boundaries.csv          # Segment info
├── statistical_tests/
│   └── test_results.csv                # Hypothesis tests
└── plots/
    ├── aggregated_cv_plot.png          # CV over time
    ├── change_points_detected.png      # Scatter with segments
    ├── model_validation.png            # Diagnostics
    └── penalty_tuning.png              # BIC curve (if auto-tuned)
```

### JSON Results Example

```json
{
  "configuration": {
    "cost_model": "l1",
    "penalty": 0.523,
    "auto_tuned": true
  },
  "change_points": {
    "indices": [30, 60],
    "n_changepoints": 2
  },
  "validation": {
    "n_significant": 2,
    "overall_significant": true
  }
}
```

---

## Troubleshooting

### Issue: Too many change points

**Cause:** Penalty too low  
**Solution:** Increase penalty or use auto-tuning

### Issue: No change points detected

**Cause:** Penalty too high  
**Solution:** Decrease penalty or verify data has patterns

### Issue: Change points not significant

**Cause:** Small effect sizes or multiple testing correction  
**Solution:** 
- Check effect sizes (Cohen's d)
- Try less conservative correction (fdr_bh instead of bonferroni)
- Increase minimum_segment_size

### Issue: BIC tuning takes long

**Cause:** Testing many penalty values  
**Solution:** Reduce penalty_range or use manual penalty

---

## Testing

```bash
# Test penalty tuning
python -m phq9_analysis.detection.penalty_tuning

# Test statistical tests
python -m phq9_analysis.detection.statistical_tests

# Test visualizations
python -m phq9_analysis.detection.visualizations

# Run full detection
python scripts/run_detection.py --data data/raw/synthetic_phq9_data.csv
```

---

## References

1. **Killick, R., Fearnhead, P., & Eckley, I. A. (2012)**  
   *Optimal detection of changepoints with a linear computational cost.*  
   Journal of the American Statistical Association, 107(500), 1590-1598.

2. **Truong, C., Oudre, L., & Vayatis, N. (2020)**  
   *Selective review of offline change point detection methods.*  
   Signal Processing, 167, 107299.

3. **Benjamini, Y., & Hochberg, Y. (1995)**  
   *Controlling the false discovery rate: a practical and powerful approach.*  
   Journal of the Royal Statistical Society: Series B, 57(1), 289-300.

---

## License

MIT License - See project root LICENSE file

---

**Last Updated:** January 2025
"""