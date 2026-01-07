# PHQ-9 Data Generation Module

## Overview

This module generates clinically realistic synthetic PHQ-9 (Patient Health Questionnaire-9) data for depression research and algorithm development.

### Key Features

✅ **Temporal Autocorrelation** - AR(1) model captures day-to-day symptom stability  
✅ **Heterogeneous Trajectories** - Patient-specific recovery rates and baselines  
✅ **Realistic Missingness** - MCAR + informative dropout patterns  
✅ **Symptom Relapses** - Occasional temporary worsening  
✅ **Literature Validation** - All parameters validated against clinical papers  

---

## Quick Start

### Basic Usage

```python
from config.generation_config import DataGenerationConfig
from phq9_analysis.generation import PHQ9DataGenerator

# Create configuration
config = DataGenerationConfig()

# Generate data
generator = PHQ9DataGenerator(config)
data, validation = generator.generate_and_validate()

# Data is a pandas DataFrame (Days × Patients)
print(data.shape)  # e.g., (50, 100)
print(data.head())
```

### Command-Line Usage

```bash
# Generate with defaults
python scripts/run_generation.py

# Custom parameters
python scripts/run_generation.py \
    --patients 200 \
    --days 365 \
    --observation-days 75 \
    --ar-coef 0.75 \
    --baseline 17.0 \
    --seed 42

# Skip literature validation (faster)
python scripts/run_generation.py --skip-validation

# View all options
python scripts/run_generation.py --help
```

---

## Clinical Model

### AR(1) Autoregressive Model

The generator uses a first-order autoregressive model:

```
Y_t = α·Y_{t-1} + (1-α)·μ_t + ε_t + relapse_t
```

**Where:**
- `Y_t` = PHQ-9 score on day t
- `α` = Autocorrelation coefficient (~0.70)
- `Y_{t-1}` = Previous day's score
- `μ_t` = Expected score from treatment trend
- `ε_t` = Daily measurement noise
- `relapse_t` = Occasional symptom spike

### Treatment Trajectory

Expected score follows linear recovery:

```
μ_t = baseline + recovery_rate × (t - treatment_start)
```

**Typical Values:**
- `baseline` = 16 ± 4 (moderate-severe depression)
- `recovery_rate` = -0.10 points/day (50% reduction over ~80 days)

### Missing Data Mechanisms

1. **MCAR (Missing Completely At Random)**
   - Random missed appointments (~5% per observation)
   
2. **Informative Dropout**
   - Study dropout (~12% overall)
   - Exponentially distributed timing (later dropout more likely)

---

## Parameter Justification

All default parameters are based on clinical literature:

| Parameter | Value | Literature Reference |
|-----------|-------|---------------------|
| **AR Coefficient** | 0.70 | Kroenke et al. (2001): Test-retest r=0.84 |
| **Baseline Mean** | 16.0 | Typical moderate-severe RCT enrollment |
| **Recovery Rate** | -0.10/day | Rush et al. (2006): STAR*D 47% response |
| **Noise Std** | 2.5 | Löwe et al. (2004): MCID ~5 points |
| **Dropout Rate** | 12% | Fournier et al. (2010): Meta-analysis 13% |

---

## Configuration

### Full Configuration Example

```python
from config.generation_config import DataGenerationConfig

config = DataGenerationConfig(
    # Sample size
    total_patients=100,
    total_days=365,
    required_sample_count=50,
    maximum_surveys_attempted=7,
    random_seed=2023,
    
    # Temporal model
    ar_coefficient=0.70,
    baseline_mean_score=16.0,
    baseline_std_score=4.0,
    recovery_rate_mean=-0.10,
    recovery_rate_std=0.03,
    noise_std=2.5,
    
    # Relapses
    relapse_probability=0.10,
    relapse_magnitude_mean=3.5,
    
    # Missingness
    dropout_rate=0.12,
    mcar_missingness_rate=0.05,
)
```

### Parameter Constraints

All parameters are validated with Pydantic:

- `total_patients`: 50-500
- `total_days`: 180-730
- `ar_coefficient`: 0.5-0.9
- `baseline_mean_score`: 10.0-20.0
- `recovery_rate_mean`: -0.20 to -0.05
- `noise_std`: 1.0-4.0

Invalid values will raise `ValidationError`.

---

## Validation

### Automatic Validation

Generated data is automatically validated against:

1. **Score Range** - All scores in [0, 27]
2. **Autocorrelation** - Mean ≈ 0.70 (range 0.6-0.8)
3. **Baseline Severity** - Mean 14-20 (moderate-severe)
4. **Response Rate** - 40-60% with ≥50% reduction
5. **Improvement Trend** - Early > Late scores
6. **Missingness Pattern** - 15-30% missing

### Validation Report

```python
validation = generator.validate(data)

print(f"Valid: {validation['overall_valid']}")
print(f"Autocorrelation: {validation['checks']['autocorrelation']['mean']:.3f}")
print(f"Response Rate: {validation['checks']['response_rate']['rate']:.1%}")

# Print full report
from phq9_analysis.generation.validators import print_validation_report
print_validation_report(validation)
```

---

## Output Files

### Data File

**Path:** `data/raw/synthetic_phq9_data.csv`

**Format:**
```
Day,Patient_1,Patient_2,...,Patient_100
Day_1,18.2,15.6,...,19.1
Day_7,17.8,14.9,...,18.3
...
```

### Validation Report

**Path:** `results/generation/validation_reports/validation_report.json`

**Contents:**
```json
{
  "overall_valid": true,
  "checks": {
    "autocorrelation": {
      "mean": 0.712,
      "in_expected_range": true
    },
    "response_rate": {
      "rate": 0.48,
      "n_responders": 48
    }
  },
  "warnings": [],
  "errors": []
}
```

---

## Advanced Usage

### Custom Trajectory Models

```python
from phq9_analysis.generation.trajectory_models import PatientTrajectory, AR1Model

# Create custom trajectory
trajectory = PatientTrajectory(
    baseline=20.0,
    recovery_rate=-0.15,  # Fast responder
    noise_std=1.5,  # Low variability
    ar_coefficient=0.80  # High stability
)

# Generate scores manually
model = AR1Model(random_seed=42)
scores = []
for day in range(1, 366):
    score = model.generate_score(trajectory, day)
    scores.append(score)
```

### Batch Generation

```python
# Generate multiple datasets with different seeds
datasets = []
for seed in range(10):
    config = DataGenerationConfig(random_seed=seed)
    generator = PHQ9DataGenerator(config)
    data, _ = generator.generate_and_validate()
    datasets.append(data)

# Analyze cross-dataset properties
import pandas as pd
all_data = pd.concat(datasets, keys=range(10))
```

---

## Troubleshooting

### Validation Warnings

**Warning:** `Autocorrelation outside expected range`

**Solution:** Increase `ar_coefficient` (e.g., from 0.65 to 0.75)

---

**Warning:** `Response rate outside expected range`

**Solution:** Adjust `recovery_rate_mean`:
- Too low response → Decrease magnitude (e.g., -0.12 to -0.10)
- Too high response → Increase magnitude (e.g., -0.08 to -0.10)

---

**Warning:** `Baseline mean outside typical range`

**Solution:** Adjust `baseline_mean_score` to 15-17 for typical trials

---

### Common Errors

**Error:** `ValidationError: Sample count exceeds total days`

**Solution:** Ensure `required_sample_count` ≤ `total_days`

---

**Error:** `ValidationError: AR coefficient outside range [0.5, 0.9]`

**Solution:** Use realistic autocorrelation (0.6-0.8 typical)

---

## Testing

Run module tests:

```bash
# Test trajectory models
python -m phq9_analysis.generation.trajectory_models

# Test validators
python -m phq9_analysis.generation.validators

# Test generator
python -c "from phq9_analysis.generation import PHQ9DataGenerator; \
           from config.generation_config import DataGenerationConfig; \
           config = DataGenerationConfig(total_patients=10, total_days=50); \
           gen = PHQ9DataGenerator(config); \
           data, val = gen.generate_and_validate(); \
           print('✅ Test passed!')"
```

---

## References

### Primary Literature

1. **Kroenke, K., Spitzer, R. L., & Williams, J. B. (2001)**  
   *The PHQ-9: validity of a brief depression severity measure.*  
   Journal of General Internal Medicine, 16(9), 606-613.

2. **Rush, A. J., et al. (2006)**  
   *Acute and longer-term outcomes in depressed outpatients requiring one or several treatment steps: a STAR*D report.*  
   American Journal of Psychiatry, 163(11), 1905-1917.

3. **Löwe, B., et al. (2004)**  
   *Monitoring depression treatment outcomes with the Patient Health Questionnaire-9.*  
   Medical Care, 42(12), 1194-1201.

4. **Fournier, J. C., et al. (2010)**  
   *Antidepressant drug effects and depression severity: a patient-level meta-analysis.*  
   JAMA, 303(1), 47-53.

### Methodology References

5. **Killick, R., Fearnhead, P., & Eckley, I. A. (2012)**  
   *Optimal detection of changepoints with a linear computational cost.*  
   Journal of the American Statistical Association, 107(500), 1590-1598.

---

## Contributing

To add new features:

1. Update `trajectory_models.py` for new model components
2. Update `validators.py` for new validation checks
3. Update `generator.py` to use new features
4. Add tests and documentation

---

## License

MIT License - See project root LICENSE file

---

## Contact

For questions or issues:
- **GitHub Issues:** [project-url]/issues
- **Email:** [your-email]

---

**Last Updated:** January 2025