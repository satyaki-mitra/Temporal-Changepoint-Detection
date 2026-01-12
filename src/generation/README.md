# PHQ-9 Synthetic Data Generation – Technical Overview

---

## 📋 Table of Contents

1. [Purpose](#1-purpose)
2. [What Is PHQ-9?](#2-what-is-phq-9)
3. [Generated Dataset Structure](#3-generated-dataset-structure)
4. [Key Design Decisions](#4-key-design-decisions)
5. [Statistical Properties](#5-statistical-properties)
6. [Clinical Realism](#6-clinical-realism)
7. [Mathematical Model](#7-mathematical-model)
8. [Generation Pipeline](#8-generation-pipeline)
9. [Validation Framework](#9-validation-framework)
10. [Configuration & Usage](#10-configuration--usage)
11. [Module Structure](#11-module-structure)
12. [References](#12-references)

---

## 1. Purpose

This module tries to generate **clinically realistic synthetic PHQ-9 data** for longitudinal mental health research. The data mirrors real-world depression monitoring studies with:

- ✅ Sparse, irregular patient observations
- ✅ Temporal dependencies via gap-aware AR(1) processes
- ✅ Realistic missingness mechanisms (MCAR + informative dropout)
- ✅ Response pattern heterogeneity (early/gradual/late/non-responders)
- ✅ Plateau logic for symptom stabilization
- ✅ Configurable relapse distributions (exponential, gamma, lognormal)

**Use Case**: Population-level temporal change-point detection on aggregated depression severity metrics.

---

## 2. What Is PHQ-9?

The **Patient Health Questionnaire-9 (PHQ-9)** is a validated 9-item self-report instrument for:
- Depression screening
- Severity quantification (0–27 scale)
- Longitudinal symptom monitoring
- Treatment response evaluation

### PHQ-9 Severity Interpretation

| Score | Severity Level |
|-------|----------------|
| 0–4 | Minimal |
| 5–9 | Mild |
| 10–14 | Moderate |
| 15–19 | Moderately Severe |
| 20–27 | Severe |

**Clinical Context**: Widely used in outpatient psychiatry, clinical trials (e.g., STAR*D), and longitudinal cohort studies.

---

## 3. Generated Dataset Structure

### Population Characteristics
- **1000 patients** monitored over **365 days**
- **10–20 surveys per patient** (biweekly to monthly assessments)
- **~95% missingness** (sparse sampling by design)
- **Irregular observation days** (realistic clinic scheduling)
- **Heterogeneous** response patterns across patients

### Data Format
```
CSV Structure:
- Rows: 365 days (Day_1, Day_2, ..., Day_365)
- Columns: 1000 patients (Patient_1, Patient_2, ..., Patient_1000)
- Values: PHQ-9 scores [0, 27] or NaN (missing)
```

**Example**:
```
           Patient_1  Patient_2  Patient_3  ...
Day_1           17.3        NaN       15.8
Day_15           NaN       14.2        NaN
Day_30          12.1        NaN       13.5
...
```

### Metadata Sidecar
Each dataset is accompanied by a `.metadata.json` file containing:
- Generation timestamp & configuration hash
- Relapse distribution used
- Response pattern distribution (early/gradual/late/non-responders)
- All model parameters for reproducibility

**Example**: `synthetic_phq9_data_exponential.csv` → `synthetic_phq9_data_exponential.metadata.json`

---

## 4. Key Design Decisions

### 4.1 Why Gap-Aware AR(1)?

**Naive AR(1)** treats all consecutive observations as adjacent:
```
Patient A: Days [1, 2, 3] → High correlation (correct)
Patient B: Days [1, 50, 100] → High correlation (WRONG!)
```

**Gap-aware AR(1)** correctly adjusts:
```
α_effective = α^Δt

For Patient B with Δt=49 and α=0.7:
α_effective = 0.7^49 ≈ 0 (essentially uncorrelated)
```

This ensures:
- ✅ Nearby observations (Δt ≤ 7 days) maintain correlation
- ✅ Distant observations (Δt > 14 days) are nearly independent
- ✅ Matches real PHQ-9 dynamics in sparse data


### 4.2 Why Response Pattern Heterogeneity?

**Real-world evidence**:
- STAR*D: ~47% responders, ~28% remission → **~20% non-responders**
- Some patients respond in 4 weeks, others need 12+ weeks
- Treatment response timing varies widely

**Our Implementation**:
- 30% early responders (weeks 2-6)
- 35% gradual responders (weeks 6-12)
- 15% late responders (weeks 12-20)
- 20% non-responders (<50% improvement)

**Impact**: More realistic population-level statistics and individual trajectories.


### 4.3 Why Plateau Logic?

**Clinical Reality**:
- Patients don't improve indefinitely
- After response, symptoms **stabilize** (maintenance phase)
- Day-to-day variability **decreases** during stability

**Without Plateau**:
- Scores could trend to 0 over 365 days (unrealistic)
- Continuous noise prevents convergence

**With Plateau**:
- Scores stabilize at realistic levels (e.g., PHQ-9 = 5-8)
- Reduced noise during plateau (symptom stability)
- Matches real maintenance therapy patterns


### 4.4 Why High Missingness?

Real-world depression monitoring has:
- **Weekly assessments**: 52 surveys/year → 85% missingness
- **Biweekly**: 26 surveys/year → 93% missingness
- **Monthly**: 12 surveys/year → 97% missingness

Our **10–20 surveys/year** (95% missingness) matches **realistic clinical practice**, not idealized research protocols.


### 4.5 Why Three Relapse Distributions?

Different distributions model different relapse mechanisms:

| Distribution | Clinical Scenario | Characteristics |
|--------------|------------------|-----------------|
| **Exponential** | Random stressors | Memoryless, heavy tail |
| **Gamma** | Accumulated stress | Bounded, moderate tail |
| **Lognormal** | Major life events | Very heavy tail, rare extremes |

**Pipeline Strategy**:
1. Generate all 3 datasets
2. EDA analyzes each independently
3. **Data Validation** (Step 3) compares EDA results + statistical properties
4. Select best dataset for detection task


### 4.6 Why Soft Boundary Reflection?

**Problem with Hard Clipping**: `Artificial spikes at 0 and 27 (unnatural distribution)`

**Solution with Soft Reflection**:

```python

if (raw_baseline < 0):
    baseline = 0 + abs(raw_baseline) * 0.5

elif (raw_baseline > 27):
    baseline = 27 - (raw_baseline - 27) * 0.5

```

**Benefits**:
- ✅ Natural distribution near boundaries
- ✅ Preserves population mean
- ✅ No artificial concentration at 0/27


---

## 5. Statistical Properties

### 5.1 Expected Validation Results

| Metric | Expected Range | Typical Value | Notes |
|--------|----------------|---------------|-------|
| **Baseline Mean** | 13.0–19.0 | ~15.7 | Moderate-severe depression |
| **Autocorrelation (gap-aware)** | 0.30–0.70 | ~0.45 | Adjusted for sparse sampling |
| **Response Rate (12-week)** | 40%–70% | ~49% | STAR*D benchmark: 47% |
| **Improvement (12-week)** | 5.0–10.0 | ~7.8 | Points reduced from baseline |
| **Median Gap** | 15–30 days | ~20 days | Realistic clinic scheduling |
| **Missingness** | 92%–96% | ~95% | Structural + excess |
| **Dropout Rate** | 15%–25% | ~21% | STAR*D-aligned |


### 5.2 Distribution Characteristics
- **Right-skewed** (more severe scores early)
- **Non-Gaussian** (bounded, ordinal)
- **Heterogeneous variance** (response pattern differences)
- **Non-stationary** (plateau phases create local stationarity)


---

## 6. Clinical Realism

### Literature-Based Validation

| Aspect | Literature Reference | Implementation |
|--------|---------------------|----------------|
| **Test-retest reliability** | Kroenke et al. (2001): r=0.84 | Gap-aware AR(1) with α=0.70 |
| **Response rate** | STAR*D: 47% at 12 weeks | ~49% via recovery_rate=-0.06 |
| **MCID** | Löwe et al. (2004): ~5 points | noise_std=2.5 < MCID |
| **Dropout** | STAR*D: 21% Level 1 | Exponential timing, 21% rate |
| **Baseline severity** | RCT inclusion: PHQ-9 15–17 | μ=16.0, σ=3.0 |
| **Response heterogeneity** | Real trials show 20-35% non-responders | 20% non-responders in model |
| **Symptom stabilization** | Maintenance phase after response | Plateau logic at 6-16 weeks |

---

## 7. Mathematical Model

### 7.1 Latent Treatment Trajectory

Each patient follows a **linear recovery trend** with **response pattern modulation**:

```
μₜ = baseline + recovery_rate × (t - t₀)
```

- **baseline**: Initial PHQ-9 score (μ=16.0, σ=3.0)
- **recovery_rate**: Daily improvement rate (μ=-0.06 points/day)
  - **Adjusted by response pattern**:
    - Early responders: 1.3× faster
    - Gradual responders: 1.0× (baseline)
    - Late responders: 0.7× slower
    - Non-responders: 0.3× (minimal)
- **t**: Calendar day, **t₀**: Treatment start (day 1)

**Clinical Justification**: Linear trends approximate gradual antidepressant response over 3–12 months (STAR*D: ~47% response rate at 12 weeks).


### 7.2 Gap-Aware AR(1) Process

Observed scores incorporate **temporal persistence** while accounting for observation gaps:

```
Yₜ = α^Δt × Yₜ₋Δₜ + (1 - α^Δt) × μₜ + εₜ + Rₜ
```

| Term | Description | Default |
|------|-------------|---------|
| α | Autocorrelation coefficient | 0.70 |
| Δt | Days since last observation | Variable |
| ε | Gaussian measurement noise | σ=2.5 |
| R | Relapse shock | P(relapse)=0.10 |

**Key Feature**: `α^Δt` exponentially decays correlation for large gaps (e.g., α³⁰ ≈ 0 for 30-day gaps).

**Literature Basis**: Kroenke et al. (2001) reported PHQ-9 test-retest r=0.84 over 2 days. Our model adjusts for realistic longitudinal gaps (median ~20 days).


### 7.3 Plateau Logic

**Symptom stabilization** after treatment response:

```
If t < plateau_start:
    E[Yₜ] = baseline + recovery_rate × (t - t₀)

If t ≥ plateau_start:
    E[Yₜ]      = plateau_score (constant)
    noise_std = noise_std × 0.5  (reduced variability)
```

**Plateau Timing by Response Pattern**:
- Early responders: **Week 6** (42 days)
- Gradual responders: **Week 10** (70 days)
- Late responders: **Week 16** (112 days)
- Non-responders: **No plateau**

**Clinical Rationale**: Real patients don't improve indefinitely—symptoms stabilize after response. This models maintenance phase.


### 7.4 Relapse Modeling

Relapses simulate **stressful life events** or **treatment interruptions**:

- **Probability**: 10% per observation day
- **Magnitude**: Positive score increase (configurable distribution)
- **Three Distribution Options**:
  - **Exponential** (default): Heavy-tailed, mean=3.5 points
  - **Gamma**: Bounded, shape=2, scale=1.75
  - **Lognormal**: Very heavy-tailed, μ=log(3.5)−0.125

**Generated Datasets** (for pipeline):
```
data/raw/
├── synthetic_phq9_data_exponential.csv
├── synthetic_phq9_data_exponential.metadata.json
├── synthetic_phq9_data_gamma.csv
├── synthetic_phq9_data_gamma.metadata.json
├── synthetic_phq9_data_lognormal.csv
└── synthetic_phq9_data_lognormal.metadata.json
```

**Note**: All 3 datasets proceed to EDA → Final dataset selection happens after EDA + validation.


### 7.5 Missingness Mechanisms

#### MCAR (Missing Completely At Random)
- **8% of scheduled observations** randomly missed
- Independent of symptom severity
- Models missed appointments

#### Informative Dropout (MNAR)
- **21% of patients** drop out before study end (STAR*D-aligned)
- Dropout timing: exponential distribution (scale=0.3×study_days, offset=60 days)
- Later dropout more common (realistic attrition)

**Result**: ~95% overall missingness (4500 observations / 365,000 possible cells).


### 7.6 Response Pattern Heterogeneity

**Clinical Motivation**: Not all patients respond identically to treatment.

**Implementation**:
| Pattern | Probability | Recovery Rate Multiplier | Plateau Week |
|---------|-------------|-------------------------|--------------|
| Early Responder | 30% | 1.3× | Week 6 |
| Gradual Responder | 35% | 1.0× | Week 10 |
| Late Responder | 15% | 0.7× | Week 16 |
| Non-Responder | 20% | 0.3× | None |

**Impact on Data**:
- More realistic population heterogeneity
- Some patients improve quickly, others slowly or not at all
- Matches real-world clinical trial outcomes


### 7.7 Plateau Logic

**Clinical Motivation**: Symptoms stabilize during maintenance phase, not continuous improvement.

**Features**:
- Automatic plateau detection based on response pattern
- **50% noise reduction** during plateau (symptom stability)
- **2-week smooth transition** (gradual entry into plateau)

**Impact on Data**:
- More realistic long-term trajectories
- Prevents unrealistic score convergence to 0
- Models maintenance therapy phase


### 7.8 Centralized Clinical Constants

**File**: `config/clinical_constants.py`

**Purpose**: All magic numbers in one place with literature references.

**Examples**:
```python
STARD_DROPOUT_RATE = 0.21                    # Rush et al. (2006)
STARD_PRIMARY_ENDPOINT_DAYS = 84             # 12 weeks
PHQ9_MCID = 5.0                              # Löwe et al. (2004)
RESPONSE_PATTERN_PROBABILITIES = {
    'early_responder': 0.30,
    'gradual_responder': 0.35,
    'late_responder': 0.15,
    'non_responder': 0.20
}
```

**Benefits**:
- Easy parameter tuning
- Consistent across all modules
- Self-documenting code


### 7.9 STAR*D-Aligned Validation

**12-Week Endpoint Validation** (not 365-day):
- STAR*D primary outcome measured at 12-14 weeks
- New validators check improvement and response rate at day 84
- More clinically relevant than final observation

**Missingness Decomposition**:
- Separates structural sparsity (by design) from excess missingness
- Provides actionable warnings
- Example: "Expected 93% sparsity, observed 95% (2% excess - acceptable)"


### 7.10 Dataset Provenance Tracking

**Metadata JSON Sidecar**:
```json
{
  "generation_timestamp": "2025-01-13T10:30:00",
  "generator_version": "2.0.0",
  "random_seed": 2023,
  "config_hash": "a3f5c7e9b2d4f1a8",
  "relapse_config": {
    "distribution": "exponential",
    "probability": 0.10,
    "magnitude_mean": 3.5
  },
  "response_pattern_distribution": {
    "early_responder": 305,
    "gradual_responder": 348,
    "late_responder": 152,
    "non_responder": 195
  },
  "features": {
    "response_patterns_enabled": true,
    "plateau_logic_enabled": true
  }
}
```

**Benefits**:
- Full reproducibility
- Dataset tracking through pipeline
- EDA can extract response patterns for analysis

---

## 8. Generation Pipeline

### 8.1 Configuration

**Location**: `config/generation_config.py`

**Key Parameters**:
```python
total_patients = 1000
total_days                = 365
maximum_surveys_attempted = 20
min_surveys_attempted     = 10
ar_coefficient            = 0.70
baseline_mean_score       = 16.0
baseline_std_score        = 3.0
recovery_rate_mean        = -0.06
noise_std                 = 2.5
relapse_probability       = 0.10
dropout_rate              = 0.21  
enable_response_patterns  = True 
enable_plateau_logic      = True      
```

**All constants** imported from `clinical_constants.py` for maintainability.


### 8.2 Execution

**Generate Single Dataset**:
```bash
python scripts/run_generation.py \
  --relapse-dist exponential \
  --enable-response-patterns \
  --enable-plateau \
  --output data/raw/synthetic_phq9_data_exponential.csv \
  --seed 2023
```

**Generate All 3 Distributions** (for pipeline):
```bash
#!/bin/bash
for dist in exponential gamma lognormal; do
    python scripts/run_generation.py \
        --relapse-dist $dist \
        --enable-response-patterns \
        --enable-plateau \
        --output data/raw/synthetic_phq9_data_${dist}.csv \
        --seed 2023
done
```

**Output**:
- **Data**       : `data/raw/synthetic_phq9_data_{distribution}.csv`
- **Metadata**   : `data/raw/synthetic_phq9_data_{distribution}.metadata.json`
- **Validation** : `results/generation/validation_reports/validation_report_{timestamp}.json`
- **Logs**       : `logs/generation_{timestamp}.log`


### 8.3 Validation Reports

Each generation produces a **comprehensive JSON validation report**:

```json
{
  "overall_valid": true,
  "checks": {
    "score_range": {"min": 0.0, "max": 27.0, "valid": true},
    "autocorrelation": {
      "mean_gap_aware": 0.45,
      "expected_range": "[0.30, 0.70]",
      "in_expected_range": true,
      "median_temporal_gap": 20.3
    },
    "baseline": {
      "mean": 15.7,
      "expected_range": "[13.0, 19.0]",
      "in_expected_range": true
    },
    "improvement": {
      "mean_improvement_12week": 7.8,
      "mean_improvement_final": 9.1,
      "endpoint": "12 weeks (84 days)"
    },
    "response_rate": {
      "rate_12week": 0.49,
      "rate_final": 0.54,
      "expected_range": "[40%, 70%]",
      "in_expected_range": true
    },
    "missingness": {
      "total_missingness": 0.95,
      "structural_sparsity_expected": 0.93,
      "excess_missingness": 0.02,
      "interpretation": "Within expected range"
    }
  },
  "warnings": [],
  "errors": [],
  "relapse_statistics": {
    "total_relapses": 1247,
    "patients_with_relapses": 623,
    "mean_relapses_per_patient": 1.25
  },
  "dropout_statistics": {
    "dropout_rate": 0.21,
    "mean_dropout_day": 187.3
  },
  "generation_metadata": {
    "response_pattern_distribution": {...}
  }
}
```

---

## 9. Validation Framework

### 9.1 Validation Checks

**Automated validators** in `src/generation/validators.py`:

| Validator | Purpose | Expected Range |
|-----------|---------|----------------|
| `_validate_score_range()` | Ensure all scores in [0, 27] | Hard constraint |
| `_validate_autocorrelation()` | Gap-aware lag-1 correlation | [0.30, 0.70] |
| `_validate_baseline()` | Mean baseline severity | [13.0, 19.0] |
| `_validate_improvement_12week()` | STAR*D endpoint improvement | ≥3.0 points |
| `_validate_response_rate_12week()` | 12-week responder rate | [40%, 70%] |
| `_validate_missingness_decomposed()` | Structural + excess | Excess ≤10% |
| `_validate_distributions()` | Skewness, kurtosis | Diagnostic |


### 9.2 Validation Thresholds (Relaxed for Sparse Data)

| Metric | Dense Data | Sparse Data (Used) |
|--------|------------|-------------------|
| **Autocorrelation** | 0.6–0.85 | 0.3–0.7 |
| **Baseline Mean** | 14.0–18.0 | 13.0–19.0 |
| **Max Gap for r** | 7 days | 14 days |
| **Autocorr Window** | 14 days | 21 days |

**Rationale**: With median gaps of 20–30 days, observed autocorrelation is naturally lower than test-retest reliability (2-day gap).


### 9.3 Missingness Decomposition (NEW)

**Formula**:
```
Total Missingness   = Structural Sparsity + Excess Missingness

Structural Sparsity = 1 - (surveys_per_patient / total_days)
                    = 1 - (15 / 365) 
                    = ~0.93 (93%)

Excess Missingness  = Total - Structural
                    = 0.95 - 0.93
                    = 0.02 (2%)
```

**Interpretation**:
- **Structural** : By design (infrequent assessments)
- **Excess**     : From dropout + MCAR
- **Tolerance**  : -5% to +10% acceptable

**Validation Output**:
```json
{
  "total_missingness": 0.95,
  "structural_sparsity_expected": 0.93,
  "excess_missingness": 0.02,
  "interpretation": "Within expected range for sparse longitudinal studies"
}
```

---

## 10. Configuration & Usage

### 10.1 Basic Usage (Defaults)

```python
from src.generation.generator import PHQ9DataGenerator
from config.generation_config import DataGenerationConfig

# Use defaults (STAR*D-aligned, response patterns ON)
config           = DataGenerationConfig()
generator        = PHQ9DataGenerator(config)
data, validation = generator.generate_and_validate()
```


### 10.2 Custom Configuration

```python
config           = DataGenerationConfig(total_patients           = 2000,
                                        total_days               = 365,
                                        dropout_rate             = 0.25,  # Higher attrition
                                        relapse_distribution     = 'gamma',
                                        enable_response_patterns = True,
                                        enable_plateau_logic     = True,
                                        random_seed              = 42,
                                       )

generator        = PHQ9DataGenerator(config)
data, validation = generator.generate_and_validate()
```


### 10.3 CLI Usage

**Single Dataset**:
```bash
python scripts/run_generation.py \
    --relapse-dist exponential \
    --patients 1000 \
    --days 365 \
    --enable-response-patterns \
    --enable-plateau \
    --seed 2023
```

**Disable New Features** (v1.0 compatibility):
```bash
python scripts/run_generation.py \
    --relapse-dist exponential \
    --disable-response-patterns \
    --disable-plateau \
    --seed 2023
```

**Custom Parameters**:
```bash
python scripts/run_generation.py \
    --relapse-dist gamma \
    --baseline 17.0 \
    --recovery-rate -0.08 \
    --ar-coef 0.75 \
    --output data/raw/custom_dataset.csv
```


### 10.4 Batch Generation (Pipeline)

**Script**: `scripts/generate_all_distributions.sh`

```bash
# Generate all 3 relapse distributions for pipeline

SEED        = 2023
BASE_OUTPUT = "data/raw"

for DIST in exponential gamma lognormal; do
    echo "Generating ${DIST} distribution..."
    
    python scripts/run_generation.py \
        --relapse-dist $DIST \
        --enable-response-patterns \
        --enable-plateau \
        --output ${BASE_OUTPUT}/synthetic_phq9_data_${DIST}.csv \
        --seed $SEED \
        --log-level INFO
    
    echo "${DIST} complete"
    echo ""
done

echo "All datasets generated!"
echo "Next step: Run EDA on all 3 datasets"
```

**Run**:
```bash
chmod +x scripts/generate_all_distributions.sh
./scripts/generate_all_distributions.sh
```

---

## 11. Module Structure

### 11.1 File Organization

```plaintext
phq9_analysis/
├── config/
│   ├── generation_config.py          # Pydantic configuration with validation
│   └── clinical_constants.py         # Centralized constants & references (NEW)
├── src/
│   └── generation/
│       ├── __init__.py
│       ├── generator.py              # Main orchestrator (UPDATED: metadata)
│       ├── trajectory_models.py      # AR(1), response patterns, plateau (UPDATED)
│       ├── validators.py             # Clinical validation (UPDATED: 12-week, missingness)
│       └── README.md                 # This file
├── scripts/
│   ├── run_generation.py             # CLI entry point (UPDATED: new flags)
│   └── generate_all_distributions.sh # Batch generation script
├── data/
│   └── raw/
│       ├── synthetic_phq9_data_exponential.csv
│       ├── synthetic_phq9_data_exponential.metadata.json (NEW)
│       ├── synthetic_phq9_data_gamma.csv
│       ├── synthetic_phq9_data_gamma.metadata.json (NEW)
│       ├── synthetic_phq9_data_lognormal.csv
│       └── synthetic_phq9_data_lognormal.metadata.json (NEW)
└── results/
    └── generation/
        └── validation_reports/
            ├── validation_report_latest.json
            └── validation_report_20250113_103000.json
```


### 11.2 Core Classes

#### **`PatientTrajectory`** (dataclass)
Encapsulates individual patient parameters:
- Baseline severity
- Recovery rate (adjusted by response pattern)
- AR coefficient
- Noise level
- Response pattern ('early_responder', etc.)
- Plateau start day & phase tracking
- Relapse history

#### **`AR1Model`**
Gap-aware autoregressive score generation:
- Stateful (tracks last observation)
- Handles irregular gaps via `α^Δt`
- Configurable relapse distributions
- Plateau logic integration
- Reduced noise during plateau

#### **`PHQ9DataGenerator`**
Orchestrates full pipeline:
- Trajectory initialization with response patterns
- Missingness generation (MCAR + dropout)
- Survey scheduling (irregular)
- Score generation with AR(1)
- Metadata capture & persistence
- Validation and reporting

#### **`DataValidator`**
Literature-based validation:
- Gap-aware autocorrelation
- 12-week endpoint validation
- Missingness decomposition
- Baseline severity checks
- Response rate validation
- Distributional diagnostics


### 11.3 Key Functions

| Function | Module | Purpose |
|----------|--------|---------|
| `initialize_patient_trajectories()` | `trajectory_models.py` | Create all patient trajectories with response patterns |
| `assign_response_pattern()` | `trajectory_models.py` | Probabilistic pattern assignment |
| `adjust_recovery_rate_for_pattern()` | `trajectory_models.py` | Modulate recovery by pattern |
| `validate_against_literature()` | `generation_config.py` | Config-level validation |
| `get_expected_structural_sparsity()` | `clinical_constants.py` | Calculate design-based missingness |

---

## 12. References

References

### Primary Literature

1. **Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001)**. The PHQ-9: validity of a brief depression severity measure. *Journal of General Internal Medicine*, 16(9), 606-613.
   - **Used for**: Test-retest reliability (r=0.84), score interpretation

2. **Rush, A. J., et al. (2006)**. Acute and longer-term outcomes in depressed outpatients requiring one or several treatment steps: a STAR*D report. *American Journal of Psychiatry*, 163(11), 1905-1917.
   - **Used for**: Dropout rate (21%), response rate (47%), 12-week primary endpoint

3. **Löwe, B., et al. (2004)**. Monitoring depression treatment outcomes with the patient health questionnaire-9. *Medical Care*, 1194-1201.
   - **Used for**: Minimal clinically important difference (MCID ~5 points)

4. **Fournier, J. C., et al. (2010)**. Antidepressant drug effects and depression severity: a patient-level meta-analysis. *JAMA*, 303(1), 47-53.
   - **Used for**: Meta-analytic benchmarks, dropout comparison

5. **Fernandez, E., et al. (2015)**. Dropout rates in psychotherapy: A systematic review. *Clinical Psychology Review*.
   - **Used for**: Real-world dropout rates (18-30%)


### Additional Reading

- **Truong, C., Oudre, L., & Vayatis, N. (2020)**. Selective review of offline change point detection methods. *Signal Processing*.
  - Background on PELT algorithm (used in downstream detection)

- **Hamilton, J. D. (1994)**. *Time Series Analysis*. Princeton University Press.
  - AR(1) process foundations


---

> *This generation module produces a realistic foundation for population-level temporal analysis and change-point detection research, not synthetic perfection*

- `Clinically grounded PHQ-9 trajectories (STAR*D-aligned)`
- `Sparse, irregular longitudinal data (realistic clinic scheduling)`
- `Statistically defensible temporal dependencies (gap-aware AR(1))`
- `Literature-validated parameters (all constants referenced)`
- `Reproducible (seeded randomness + metadata tracking)`
- `Configurable relapse mechanisms (3 distributions)`
- `Heterogeneous response patterns (4 types)`
- `Realistic symptom stabilization (plateau logic)`