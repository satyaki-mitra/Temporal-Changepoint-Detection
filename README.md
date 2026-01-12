<div align="center">

# Temporal Change-Point Detection on Sparse Time Series Data (PHQ-9 Dataset)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **End-to-end system for detecting significant changes in mental health trajectories using PHQ-9 scores. Includes synthetic data generation, exploratory analysis, and multi-algorithm change-point detection with automatic model selection.**

</div>

---

## 🎯 Overview

This framework analyzes **Patient Health Questionnaire-9 (PHQ-9)** time series to identify significant shifts in depression severity. It combines:

1. **Synthetic Data Generation** - Realistic PHQ-9 trajectories with AR(1) autocorrelation, relapses, and dropout
2. **Exploratory Data Analysis** - Clustering, trend analysis, and statistical validation
3. **Data Validation** - Cross-dataset comparison based on Medical benchmarks and EDA findings
4. **Change-Point Detection** - 6 detector models (PELT + BOCPD) with automatic selection
5. **Model Selection** - Agreement-first scoring to identify best-performing detector

###  Pipeline Architecture

```
Generation  →  EDA   → Literature Validation → Detection → Model Selection
    ↓           ↓              ↓                   ↓              ↓
 3 datasets   3 EDAs     Compare & Select      Best data      Best model
```

--- 
## Problem Statement

> **The aim of this analysis is to detect significant changes in the Aggregated Patient Health Questionnaire-9 (PHQ-9) scores across 365 days using the Coefficient of Variation (CV) as the aggregated statistic. The sample dataset comprises 1000 patients identified by unique patient IDs, with each patient's PHQ-9 score recorded for 50 randomly selected days amongst the total 365 days of a year.**

### Key Challenges:

- **Sparse Data**: Each patient attempts the survey at most 6 times during the 365-day span
- **Missing Values**: Not all patients respond every day, resulting in NaN values
- **Heterogeneous Population**: Diverse baseline PHQ-9 scores across patients
- **Temporal Patterns**: Need to identify critical shifts in depression levels over time

---

## 📋 Context & Clinical Background

### About PHQ-9

The Patient Health Questionnaire-9 (PHQ-9) is a multipurpose instrument used for:

- `Diagnosis and screening of depression`
- `Monitoring and measuring severity of depression`
- `Clinical utility as a brief, self-report tool`
- `Repeated administration for tracking improvement or regression`

### PHQ-9 Severity Levels:

- `0-4`: Minimal depression
- `5-9`: Mild depression
- `10-14`: Moderate depression
- `15-19`: Moderately severe depression
- `20-27`: Severe depression

### Clinical Applications

- `Treatment response monitoring`
- `Relapse detection`
- `Clinical trial analytics`
- `Personalized intervention strategies`

---

### 🧩 Key Features

✅ **Clinically Grounded** - Literature-validated parameters (Kroenke et al., Rush et al.)  
✅ **Sparse Data Ready** - Handles irregular observations and high missingness  
✅ **Multi-Algorithm** - PELT (L1/L2/RBF/AR) + BOCPD (Gaussian)  
✅ **Auto-Tuned** - BIC penalty tuning (PELT), hazard optimization (BOCPD)  
✅ **Statistically Validated** - Wilcoxon tests, FDR correction, effect sizes  
✅ **Production Ready** - Pydantic configs, structured logging, reproducible outputs

---

## 📂 Project Structure

```
phq9_analysis/
├── config/                          # Configuration modules
│   ├── generation_config.py         # Data generation parameters
│   ├── eda_config.py                # EDA settings
│   ├── detection_config.py          # Detection algorithms
│   └── model_selection_config.py    # Model scoring/ranking
│
├── src/                             # Core package
│   ├── generation/                  # Synthetic data generation
│   │   ├── generator.py             # Main generator
│   │   ├── trajectory_models.py     # AR(1) + relapse models
│   │   └── validators.py            # Literature validation
│   │
│   ├── eda/                         # Exploratory analysis
│   │   ├── analyzer.py              # Main EDA orchestrator
│   │   ├── clustering.py            # KMeans, Agglomerative, Temporal
│   │   └── visualizations.py        # Plotting engine
|   | 
|   ├── validation/
|   |    ├── __init__.py
|   |    ├── literature_validator.py    # Main validator
|   |    ├── clinical_criteria.py       # Medical benchmarks
|   |    ├── comparator.py              # Cross-dataset comparison         
|   |    └── README.md
│   │
│   ├── detection/                   # Change-point detection
│   │   ├── detector.py              # Main orchestrator
│   │   ├── pelt_detector.py         # PELT implementation
│   │   ├── bocpd_detector.py        # BOCPD implementation
│   │   ├── penalty_tuning.py        # BIC-based PELT tuning
│   │   ├── hazard_tuning.py         # BOCPD hazard optimization
│   │   ├── statistical_tests.py     # Frequentist/Bayesian validation
│   │   ├── visualizations.py        # Detection plots
│   │   └── model_selector.py        # Best model selection
│   │
│   └── utils/                       # Utilities
│       └── logging_util.py          # Structured logging
│
├── scripts/                         # Execution scripts
│   ├── run_generation.py            # Generate synthetic data
│   ├── run_eda.py                   # Run exploratory analysis
│   └── run_detection.py             # Run detection pipeline
│
├── data/                              # Data storage
│   ├── raw/                           # 3 generated datasets
|   ├── finalized_data/                # Selected best dataset
│   |   ├── phq9_data_finalized.csv
│   |   └── selection_metadata.json
│   └── processed/                   # Processed data
│
├── results/                         # Analysis outputs
│   ├── generation/
|   ├── eda/
|   │   ├── exponential/
|   │   ├── gamma/
|   │   └── lognormal/
|   ├── validation/                      # Validation reports
|   │   ├── exponential_validation.json
|   |   ├── gamma_validation.json
|   |   ├── lognormal_validation.json
|   │   ├── comparison_report.json
|   |   └── dataset_comparison.png
|   ├── detection/                       # Only runs on finalized data
|   └── best_model/ 
│
├── logs/                            # Execution logs
├── docs/                            # Documentation
└── requirements.txt
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/phq9-changepoint-detection.git
cd phq9-changepoint-detection
pip install -r requirements.txt
```

### Basic Workflow

```bash
# 1. Generate synthetic data
python scripts/run_generation.py

# 2. Explore data
python scripts/run_eda.py

# 3. Detect change points (all 6 models + selection)
python scripts/run_detection.py --enable-selection
```

**Outputs:**
- `data/raw/synthetic_phq9_data.csv` - Generated data
- `results/eda/` - EDA plots and statistics
- `results/detection/` - All 6 model results
- `results/best_model/` - Selected model's change points

---

## 🔬 Complete Pipeline with Literature Validation

**What it does:**
1. Generates 3 datasets (exponential, gamma, lognormal relapses)
2. Performs EDA on each
3. **Validates against clinical literature** (Kroenke, STAR*D, etc.)
4. **Selects best dataset** based on compliance score
5. Runs detection on finalized data only
6. Selects best detection model

### Manual Step-by-Step
```bash
# 1. Generate all 3 distributions
for dist in exponential gamma lognormal; do
    python scripts/run_generation.py --relapse-dist $dist
done

# 2. EDA on each
for dist in exponential gamma lognormal; do
    python scripts/run_eda.py \
        --data data/raw/synthetic_phq9_data_$dist.csv \
        --output-dir results/eda/$dist
done

# 3. Literature validation & selection
python scripts/run_validation.py

# 4. Detection on finalized data
python scripts/run_detection.py \
    --data data/finalized_data/phq9_data_finalized.csv \
    --enable-selection
```

### Validation Criteria

Data validated against:
- **Kroenke et al. (2001)**: PHQ-9 autocorrelation (r=0.84)
- **Rush et al. (2006)**: STAR*D response rates (~47%)
- **Löwe et al. (2004)**: MCID thresholds
- **Fournier et al. (2010)**: Dropout meta-analysis
- **Clinical RCT standards**: Baseline severity, improvement trajectories

**Compliance Score**: Weighted combination of all criteria (threshold: 0.70)

---

## 📊 Modules

### 1. Data Generation

**Purpose**: Create realistic PHQ-9 trajectories with:
- AR(1) autocorrelation (ρ = 0.70)
- Exponential dropout (~18%)
- Relapse events (10% daily probability)
- Sparse, irregular sampling

**Usage:**
```bash
# Default configuration
python scripts/run_generation.py

# Custom parameters
python scripts/run_generation.py \
    --patients 500 \
    --days 180 \
    --baseline 18.0 \
    --recovery-rate -0.08
```

**Outputs:**
- `data/raw/synthetic_phq9_data.csv`
- `results/generation/validation_reports/validation_report.json`

**Key Parameters:**
```python
total_patients: 1000              # Sample size
total_days: 365                   # Study duration
baseline_mean_score: 16.0         # Initial severity
recovery_rate_mean: -0.06         # Daily improvement
ar_coefficient: 0.70              # Autocorrelation
relapse_probability: 0.10         # Daily relapse risk
dropout_rate: 0.18                # Study attrition
```

**See:** [phq9_analysis/generation/README.md](phq9_analysis/generation/README.md)

---

### 2. Exploratory Data Analysis (EDA)

**Purpose**: Statistical analysis and clustering to understand data structure.

**Usage:**
```bash
# Full analysis
python scripts/run_eda.py

# Temporal clustering
python scripts/run_eda.py --temporal

# Fixed K clusters (skip optimization)
python scripts/run_eda.py --n-clusters 4
```

**Outputs:**
- `results/eda/visualizations/` - Scatter, daily averages, clusters
- `results/eda/summary_statistics.csv` - Daily statistics
- `results/eda/cluster_characteristics.csv` - Cluster properties
- `results/eda/analysis_summary.json` - Overall results

**Key Features:**
- **Clustering**: KMeans, Agglomerative, Temporal-aware
- **Optimization**: Elbow, Silhouette, Gap Statistic
- **Validation**: Literature-based parameter checks

**See:** [phq9_analysis/eda/README.md](phq9_analysis/eda/README.md)

---

### 3. Change-Point Detection

**Purpose**: Detect significant shifts in aggregated PHQ-9 statistics using 6 models.

#### 3.1 Detectors

**PELT (Offline):**
- **PELT_L1**: Robust to outliers (MAD-based)
- **PELT_L2**: Least-squares minimization
- **PELT_RBF**: Kernel-based nonlinear detection
- **PELT_AR**: Autoregressive cost function

**BOCPD (Online Bayesian):**
- **BOCPD_Gaussian_Heuristic**: ACF-based hazard tuning
- **BOCPD_Gaussian_Predictive_LL**: Likelihood-optimized hazard

#### 3.2 Usage

```bash
# Run all 6 models + selection
python scripts/run_detection.py --enable-selection

# Single model
python scripts/run_detection.py \
    --execution-mode single \
    --detectors pelt \
    --cost-model l1

# Custom parameters
python scripts/run_detection.py \
    --penalty 2.0 \
    --posterior-threshold 0.7 \
    --alpha 0.01
```

#### 3.3 Outputs

**All Models:**
```
results/detection/
├── change_points/          # JSON per model
│   ├── pelt_l1.json
│   ├── pelt_l2.json
│   ├── ...
├── statistical_tests/      # Validation CSVs
│   ├── pelt_l1_validation.csv
│   ├── pelt_l1_summary.json
│   ├── ...
├── plots/
│   ├── pelt_comparison.png
│   ├── bocpd_comparison.png
│   ├── all_models_grid.png
│   └── bocpd_gaussian_*_posterior.png
└── run_metadata.json
```

**Best Model (if selection enabled):**
```
results/best_model/
├── change_points.json
├── validation.csv
└── selection_metadata.json
```

#### 3.4 Key Algorithms

**PELT:**
- Dynamic programming with pruning (O(n) complexity)
- BIC-based penalty tuning
- Wilcoxon rank-sum validation with FDR correction

**BOCPD:**
- Online Bayesian inference
- Hazard parameter tuning (heuristic or predictive likelihood)
- Posterior probability thresholding

**Model Selection:**
- Metrics: n_significant_cps, mean_effect_size, posterior_mass, stability_score
- Agreement-first strategy: rewards cross-model consensus
- Explainable ranking with rationale

**See:** [phq9_analysis/detection/README.md](phq9_analysis/detection/README.md)

---

## 🔧 Configuration

All modules use Pydantic for validated configurations:

```python
# Example: Detection config override
from config.detection_config import ChangePointDetectionConfig

config = ChangePointDetectionConfig(
    execution_mode='compare',
    detectors=['pelt', 'bocpd'],
    penalty=1.5,
    auto_tune_penalty=True,
    hazard_lambda=50.0,
    auto_tune_hazard=True,
    alpha=0.01,
    selection_enabled=True
)
```

**Config Files:**
- `config/generation_config.py` - Data parameters
- `config/eda_config.py` - Clustering/visualization
- `config/detection_config.py` - Detection algorithms
- `config/model_selection_config.py` - Scoring weights

---

## 📈 Typical Workflow

### End-to-End Pipeline

```bash
# 1. Generate 1000 patients, 365 days
python scripts/run_generation.py --patients 1000 --days 365

# 2. Explore data patterns
python scripts/run_eda.py --max-clusters 15

# 3. Detect change points with all models
python scripts/run_detection.py --enable-selection

# 4. Inspect best model
cat results/best_model/selection_metadata.json
```

### Custom Analysis

```python
from config.detection_config import ChangePointDetectionConfig
from phq9_analysis.detection.detector import ChangePointDetectionOrchestrator

# Load custom config
config = ChangePointDetectionConfig(
    data_path='my_real_data.csv',
    pelt_cost_models=['l1', 'l2'],
    auto_tune_penalty=True,
    hazard_tuning_method='predictive_ll',
    selection_enabled=True
)

# Run detection
orchestrator = ChangePointDetectionOrchestrator(config=config)
results = orchestrator.run()

# Best model is saved to results/best_model/
```

---

## 📊 Visualization Examples

### EDA Outputs

1. **Scatter Plot** - All PHQ-9 scores over time
2. **Daily Averages** - Mean trajectory with severity bands
3. **Cluster Results** - Temporal segments with boundaries
4. **Cluster Optimization** - Elbow + Silhouette curves

### Detection Outputs

1. **PELT Comparison** - All 4 PELT cost models overlaid
2. **BOCPD Comparison** - Both hazard tuning methods
3. **6-Panel Grid** - Individual model results with annotations
4. **BOCPD Posterior** - Run-length heatmap + CP posterior

---

## 🧪 Testing & Validation

### Generation Validation

Checks against literature benchmarks:
- Autocorrelation: 0.6-0.85 (Kroenke et al., 2001)
- Baseline severity: 14-18 (typical RCT enrollment)
- Response rate: 40-60% (STAR*D Level-1)
- Dropout: 10-20% (Fournier et al., 2010)

### Detection Validation

**PELT:**
- Wilcoxon rank-sum tests (p < 0.05)
- FDR correction (Benjamini-Hochberg)
- Cohen's d effect sizes (d ≥ 0.3)

**BOCPD:**
- Posterior probability thresholds (default: 0.6)
- Persistence filtering (3+ consecutive time points)

---

## 📚 Key References

1. **PELT Algorithm**  
   Killick, R., Fearnhead, P., & Eckley, I. A. (2012). *Optimal detection of changepoints with a linear computational cost.* JASA.

2. **BOCPD Algorithm**  
   Adams, R. P., & MacKay, D. J. (2007). *Bayesian online changepoint detection.* arXiv:0710.3742.

3. **PHQ-9 Validation**  
   Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001). *The PHQ-9: validity of a brief depression severity measure.* J Gen Intern Med.

4. **STAR*D Study**  
   Rush, A. J., et al. (2006). *Acute and longer-term outcomes in depressed outpatients requiring one or several treatment steps: STAR*D.* Am J Psychiatry.

5. **Meta-Analysis**  
   Fournier, J. C., et al. (2010). *Antidepressant drug effects and depression severity: a patient-level meta-analysis.* JAMA.

---

## 🔬 Clinical Applications

### Research
- **Clinical Trial Analytics** - Detect treatment effect timing
- **Biomarker Discovery** - Correlate change points with biological markers
- **Population Health** - Study community mental health dynamics

### Clinical Practice
- **Treatment Monitoring** - Track patient response in real-time
- **Relapse Detection** - Early identification of symptom worsening
- **Personalized Care** - Tailor interventions based on trajectories

### Healthcare Systems
- **Resource Allocation** - Predict high-demand periods
- **Quality Improvement** - Monitor clinic-wide trends
- **Policy Evaluation** - Assess intervention impact timing

---

## 🛠️ Requirements

**Python:** 3.8+

**Core:**
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `scipy` - Statistical tests
- `pydantic` - Configuration validation

**Detection:**
- `ruptures` - PELT implementation
- `scikit-learn` - Clustering, preprocessing

**Visualization:**
- `matplotlib` - Plotting
- `seaborn` - Statistical graphics

**Install all:**
```bash
pip install -r requirements.txt
```

---

## 📖 Documentation

- **Generation:** [phq9_analysis/generation/README.md](phq9_analysis/generation/README.md)
- **EDA:** [phq9_analysis/eda/README.md](phq9_analysis/eda/README.md)
- **Detection:** [phq9_analysis/detection/README.md](phq9_analysis/detection/README.md)
- **Architecture:** [docs/architecture.md](docs/architecture.md)
- **Literature:** [docs/literature_references.md](docs/literature_references.md)

---

## 📜 License

MIT License - See [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Satyaki Mitra**  
*Data Scientist | ML Researcher | Clinical AI*

---

## 🌟 Acknowledgments

This project synthesizes methods from:
- Killick et al. (PELT algorithm)
- Adams & MacKay (BOCPD algorithm)
- Kroenke et al. (PHQ-9 validation)
- Rush et al. (STAR*D clinical trial)

Special thanks to the open-source community for foundational libraries.

---

⭐ **If this project helps your research, please consider citing it and starring the repository!** ⭐