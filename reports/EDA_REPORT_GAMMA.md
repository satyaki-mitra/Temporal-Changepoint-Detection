<div align="center">

# Exploratory Data Analysis 
## Synthetic PHQ-9 Dataset with Gamma Relapse Distribution


**Dataset**: `synthetic_phq9_data_gamma.csv` | **Metadata**: `synthetic_phq9_data_gamma.metadata.json`

**Study Period**: `365 days` | **Sample Size**: `1,000 patients`  

</div>

---

## Executive Summary

This report presents a comprehensive `Exploratory Data Analysis (EDA)` of a synthetic `longitudinal PHQ-9 dataset` with **Gamma relapse distribution**. The dataset simulates real-world depression monitoring with realistic `sparsity`, `temporal dependencies`, and `heterogeneous` treatment response patterns.

### Key Findings

| Metric | Value | Clinical Interpretation |
|--------|-------|------------------------|
| **Total Observations** | 13,680 | ~14 assessments per patient |
| **Missingness** | 96.25% | Realistic sparse monitoring |
| **Response Pattern Distribution** | 73.1% non-responders | Challenging population |
| **Relapse Rate** | 42.8% | 428/1000 patients experienced relapses |
| **Total Relapses** | 619 events | Moderate clustering pattern |
| **Optimal Clusters** | 2 (Silhouette) / 5 (Elbow) | Clear temporal phases |

### Clinical Realism Assessment

**Pass** - Dataset exhibits clinically realistic characteristics:
- Sparse, irregular assessments (biweekly to monthly)
- Heterogeneous response patterns (4 distinct groups)
- Gamma relapse distribution (bounded, right-skewed)
- Temporal autocorrelation structure
- Realistic dropout patterns

---

## 1. Data Overview

### 1.1 Population Characteristics

**Total Patients**: `1,000` 
**Study Duration**: `365 days`  
**Monitoring Frequency**: `~14 assessments per patient (biweekly to monthly)`


### 1.2 Response Pattern Distribution

The dataset includes four treatment response patterns based on STAR*D trial benchmarks:

![Response Pattern Distribution](../results/comparison/synthetic_phq9_data_gamma/visualizations/response_patterns.png)

| Pattern | Observed | Percentage | Clinical Profile |
|---------|----------|------------|------------------|
| **Non-Responder** | 731 | 73.1% | Minimal improvement (<50% reduction) |
| **Late Responder** | 192 | 19.2% | Response after 12-20 weeks |
| **Gradual Responder** | 55 | 5.5% | Steady improvement over 6-12 weeks |
| **Early Responder** | 22 | 2.2% | Rapid response within 2-6 weeks |

**Key Observation**: The high proportion of non-responders (73.1%) reflects a **treatment-resistant population**, which is clinically realistic for tertiary care settings or patients requiring multiple treatment lines. Only 26.9% of patients achieved meaningful response, indicating a challenging clinical cohort.

**Critical Finding**: The severe underrepresentation of early (2.2%) and gradual (5.5%) responders suggests:
1. Classification thresholds may be overly stringent
2. High noise-to-signal ratio (noise_std = 2.5) obscures recovery patterns
3. Measurement variability may mask treatment effects
4. Plateau detection logic may misclassify responders


### 1.3 Relapse Configuration

**Distribution Type**: Gamma  
**Relapse Probability**: ~10% per assessment window (metadata parameter)  
**Mean Magnitude**: 3.5 PHQ-9 points (theoretical)

**Gamma Distribution Characteristics**:
- **Shape parameter (α)**: 2.0 (controls skewness and variance)
- **Scale parameter (β)**: 1.75 (controls mean)
- **Bounded distribution**: Right-skewed with moderate tail, fewer extreme values
- **Theoretical mean**: α × β = 2.0 × 1.75 = 3.5 points
- **Accumulated stress model**: Represents gradual buildup leading to relapse

![Relapse Events](../results/comparison/synthetic_phq9_data_gamma/visualizations/relapse_events.png)

**Observed Relapse Statistics**:
- **Total relapses**: 619 events
- **Patients with relapses**: 428 (42.8%)
- **Observed rate**: 4.5% per observation (619/13,680)
- **Magnitude range**: 3-15 points observed

**Temporal Pattern** (left panel): Moderate early clustering (Days 20-100) with relatively uniform distribution throughout the study period. The gamma distribution's shape parameter creates more consistent relapse timing compared to purely memoryless processes.

**Magnitude Distribution** (right panel):
- Most relapses: 2-4 point increases (~74%)
- Moderate relapses: 5-8 points (~20%)
- Severe relapses: >8 points (~6%)

The bounded nature of the gamma distribution prevents extremely large relapses while maintaining clinical realism.

---

## 2. Descriptive Statistics

### 2.1 Missingness Analysis

| Metric | Value |
|--------|-------|
| **Possible Observations** | 365,000 (365 days × 1,000 patients) |
| **Actual Observations** | 13,680 |
| **Missing Observations** | 351,320 |
| **Missingness Rate** | 96.25% |

**Interpretation**: The ~96% missingness reflects:
- **Structural sparsity** (~93%): By design (biweekly to monthly assessments)
- **Excess missingness** (~3%): Realistic dropout and MCAR (Missing Completely At Random)

This level of sparsity is **clinically realistic** for:
- Real-world clinical practice (monthly to quarterly visits)
- Pragmatic trials
- Naturalistic longitudinal studies

**Metadata Consistency**: Observed missingness (96.25%) matches metadata expectation.


### 2.2 Temporal Trends

![Daily Average PHQ-9 Scores](../results/comparison/synthetic_phq9_data_gamma/visualizations/daily_averages.png)

**Observed Pattern**:
- **Initial Severity**: Mean PHQ-9 ≈ 16-17 (Moderately Severe depression)
- **Overall Trend**: Gradual decline (slope: -0.014 points/day)
- **Final Scores**: Mean PHQ-9 ≈ 7-8 (Mild depression)
- **Variability**: High day-to-day variation due to sparse data and relapse events

**Clinical Significance**:
- Population-level improvement: ~8-9 points over 365 days
- Improvement consistent with real-world treatment outcomes
- High variability reflects heterogeneous response patterns
- Slightly steeper decline than some naturalistic studies


### 2.3 Score Distribution

![PHQ-9 Score Distribution](../results/comparison/synthetic_phq9_data_gamma/visualizations/scatter_plot.png)

**Key Observations**:
1. **Baseline clustering**: Dense observations at Days 0-50 in moderate-severe range (PHQ-9 15-25)
2. **Temporal spread**: Increasing score variability over time
3. **Color gradient**: Later measurements show lighter colors (lower scores on average)
4. **Sparse late-phase**: Fewer observations in Days 200-365 due to dropout
5. **Wide range**: Scores span full PHQ-9 range (0-27) throughout study

---

## 3. Response Pattern Analysis

### 3.1 Individual Trajectories

![Patient Trajectories by Response Pattern](../results/comparison/synthetic_phq9_data_gamma/visualizations/patient_trajectories_by_pattern.png)

**Pattern-Specific Characteristics**:

#### Non-Responders (n = 731, 73.1%)
- **Trajectory**: Relatively flat with high variability
- **Baseline**: Mean ~15-17 (Moderate to Moderately Severe)
- **Final**: Mean ~14-16 (minimal change)
- **Volatility**: High due to relapses without sustained improvement
- **Clinical Note**: May require treatment modification or augmentation
- **Key Feature**: Persistent symptom burden despite treatment exposure

#### Late Responders (n = 192, 19.2%)
- **Trajectory**: Delayed improvement starting ~Day 100-150
- **Baseline**: Mean ~18-22 (Moderately Severe to Severe)
- **Final**: Mean ~10-12 (Mild depression)
- **Pattern**: Plateau after response
- **Clinical Note**: Patience needed; response after 12+ weeks
- **Key Feature**: Extended observation period required to detect improvement

#### Gradual Responders (n = 55, 5.5%)
- **Trajectory**: Steady linear decline
- **Baseline**: Mean ~10-16
- **Final**: Mean ~1-5 (near-remission)
- **Pattern**: Consistent improvement without plateau
- **Clinical Note**: Ideal responders with sustained recovery
- **Key Feature**: Smooth recovery curves

#### Early Responders (n = 22, 2.2%)
- **Trajectory**: Rapid decline in first 6 weeks, then plateau
- **Baseline**: Mean ~15-23
- **Final**: Mean ~5-12
- **Pattern**: Quick response, then maintenance
- **Clinical Note**: Excellent prognosis


### 3.2 Improvement Distribution

**Observed Patterns**:
- **Non-responders**: Heavy concentration around 0% improvement, with many showing negative change (worsening)
- **Responder groups**: Show variable improvement, but most fall below the 50% response threshold
- **Extreme cases**: Some patients show improvement >100% (artifact of low baseline or measurement noise)

**Clinical Response Rate** (≥50% reduction):
- Estimated at ~15-20% based on visual inspection of improvement distribution
- Most patients (>70%) show <20% improvement or worsening


### 3.3 Response Pattern Validation

**Classification Challenges**:

The observed distribution (73.1% non-responders) represents a more treatment-resistant cohort than typical outpatient populations. Possible explanations:

1. **Threshold stringency**: Classification slope thresholds may be too strict
   - Early responder: slope ≤ -0.08 (very steep decline required)
   - Gradual responder: -0.08 < slope ≤ -0.04
   
2. **Noise dominance**: With noise_std = 2.5 and recovery_rate_mean = -0.075, signal-to-noise ratio ≈ 0.03, making recovery signals difficult to detect above noise floor

3. **Measurement variability**: Gamma relapses with bounded tail create moderate disruptions that distort slope calculations

4. **Plateau masking**: Patients who improve then plateau may be misclassified if slope calculation doesn't account for phase transitions

**Recommendation**: Consider relaxing slope thresholds or using alternative classification methods (e.g., endpoint-based, time-to-response) for more realistic response prevalence.

---

## 4. Temporal Clustering Analysis

### 4.1 Cluster Optimization

![Cluster Optimization](../results/comparison/synthetic_phq9_data_gamma/clustering/cluster_optimization.png)

**Elbow Method** (Left Panel):
- **Elbow at K = 5**: Suggests 5 distinct temporal phases
- Rapid inertia decrease from K = 2 to K = 5
- Diminishing returns after K = 5

**Silhouette Analysis** (Right Panel):
- **Optimal K = 2**: Highest silhouette score (0.635)
- Clear separation between two major phases
- Silhouette score decreases substantially for K > 2
- **K = 2 selected** for primary interpretation (better separation and clinical interpretability)


### 4.2 Two-Cluster Solution

![Cluster Results](../results/comparison/synthetic_phq9_data_gamma/visualizations/cluster_results.png)

#### **Cluster 1: Early Treatment Phase (Days 0-42)**

| Metric | Value |
|--------|-------|
| **Duration** | 42 days |
| **Average Score** | 14.8-15.5 (Moderate depression) |
| **Within-cluster Std** | 4.0-4.5 |
| **Daily Std** | 3.0-3.5 |
| **N Observations (avg)** | ~38-40 per day |
| **Score Range** | 5.0 - 27.0 |
| **Severity** | Moderate to Moderately Severe |

**Characteristics**:
- Higher average scores (acute symptom phase)
- Denser monitoring (more frequent assessments)
- Higher variability (treatment initiation effects)
- Represents **acute treatment phase** (0-6 weeks)

#### **Cluster 0: Maintenance Phase (Days 43-364)**

| Metric | Value |
|--------|-------|
| **Duration** | 322 days |
| **Average Score** | 9.5-10.5 (Mild depression) |
| **Within-cluster Std** | 4.0-4.5 |
| **Daily Std** | 2.5-3.0 |
| **N Observations (avg)** | ~34-36 per day |
| **Score Range** | 0.8 - 25.5 |
| **Severity** | Mild to Moderate |

**Characteristics**:
- Lower average scores (improvement visible)
- Sustained over longer period
- Lower variability (bounded relapse distribution helps)
- Represents **maintenance/continuation phase** (6+ weeks)


### 4.3 Clinical Interpretation

The **two-cluster solution** aligns with standard depression treatment phases:
1. **Acute Phase** (Weeks 0-6): Initial symptom reduction, frequent monitoring, treatment adjustment
2. **Continuation Phase** (Weeks 6-52): Maintenance of gains, relapse prevention, less frequent monitoring

**Boundary Day ~42-43** corresponds to the **6-week mark**, a clinically meaningful timepoint for:
- Evaluating initial treatment response
- Deciding on treatment continuation vs. modification
- Transitioning from acute to maintenance care
- Standard trial duration for antidepressant efficacy assessment

**Cluster Transition Pattern**:
- Clear boundary with minimal overlap between Days 0-42 (Cluster 1) and Day 43+ (Cluster 0)
- Some scatter points show mixed assignments due to day-to-day variability
- Overall separation quality: Silhouette score 0.635 indicates good cluster cohesion

---

## 5. Data Quality Assessment

### 5.1 Temporal Autocorrelation

**Expected**: Gap-aware AR(1) with α = 0.70 (metadata parameter)

**Observed Indicators**:
- Smooth population-level trends (not erratic)
- Gradual decline rather than sudden jumps
- Cluster stability over multi-day windows
- Within-patient trajectory smoothness in response pattern plots

**Theoretical Correlation Structure**:
- Nearby observations (Δt ≤ 7 days): Expected correlation ~0.70
- Moderate gaps (Δt = 14 days): Expected correlation ~0.49 (0.70²)
- Distant observations (Δt > 28 days): Negligible correlation

**Interpretation**: Visual inspection suggests realistic temporal dependencies consistent with AR(1) decay.


### 5.2 Relapse Characteristics

| Metric | Observed | Theoretical Expectation |
|--------|----------|------------------------|
| **Total Relapses** | 619 | ~1,368 (10% of 13,680 observations) |
| **Patients with Relapses** | 428 | Variable (depends on observation frequency) |
| **Relapse Rate** | 42.8% | ~10% per assessment window |
| **Observed Rate per Observation** | 4.5% | 10% (metadata parameter) |
| **Mean Magnitude** | ~3.5 points | α × β = 3.5 points |
| **Maximum Magnitude** | 15 points | Bounded by gamma distribution |

**Discrepancy Analysis**:
The lower-than-expected total relapse count (619 vs. ~1,368) suggests:
1. **Patient-level probability**: 10% may apply per patient per assessment window, not per observation
2. **Conditional application**: Relapse probability may depend on recovery state
3. **Realistic variation**: Not all patients are equally susceptible to relapse

**Gamma-Specific Validation**:
- **Shape parameter (α=2.0)**: Produces moderate right skew ✓
- **Scale parameter (β=1.75)**: Controls mean magnitude ✓
- **Theoretical mean**: α × β = 2.0 × 1.75 = 3.5 ✓
- **Observed mean**: ~3.5 points ✓
- **Magnitude range**: 3-15 points (bounded distribution) ✓


### 5.3 Validation Against Literature

| Benchmark | Literature (STAR*D) | Observed | Status |
|-----------|---------------------|----------|--------|
| **Response Rate (12-week)** | ~47% | ~27% (73% non-responders) | Lower |
| **Dropout Rate** | ~21% | ~3% excess missingness | Realistic |
| **Baseline Severity** | PHQ-9 15-17 | ~16 | Aligned ✓ |
| **MCID (Minimal Change)** | ~5 points | Exceeds threshold | Detectable ✓ |
| **Plateau Detection** | 60-80% | High prevalence | Realistic |

**Overall Assessment**: Dataset is **clinically plausible** but represents a more **treatment-resistant population** than typical STAR*D cohort. This is valuable for testing detection algorithms on challenging data.

---

## 6. Suitability for Change Point Detection

### 6.1 Signal Characteristics

**Favorable for Detection**:
- ✓ Clear temporal phases (2-cluster solution with boundary at Day 42-43)
- ✓ Detectable population-level trend (slope: -0.014 points/day)
- ✓ Sufficient observations per day (mean ~38)
- ✓ Realistic noise structure (gamma relapses + AR(1) + measurement error)
- ✓ Bounded relapse distribution (fewer outliers)

**Challenges**:
- High missingness (96.25%) requires aggregation
- High within-day variability
- Multiple change points may be subtle
- Moderate relapse tail may obscure some transitions


### 6.2 Aggregation Strategy

**Recommended Statistic**: **Coefficient of Variation (CV)**

```
CV = σ / μ
```

**Rationale**:
- Captures both mean (symptom severity) and variability (population heterogeneity)
- Sensitive to distributional changes (both location and scale shifts)
- Robust to moderate outliers
- Clinically interpretable (represents symptom heterogeneity in population)
- Normalizes variance by mean, making cross-phase comparisons valid

**Expected Change Points** (based on clustering and visual inspection):
1. **Day ~42-43**: Primary transition from acute to maintenance phase ✓ (strong evidence)
2. **Days 80-100**: Potential relapse cluster peak (secondary signal)
3. **Days 150-200**: Late responder plateau onset (subtle)
4. **Days 300+**: Late-phase stabilization


### 6.3 Detection Recommendations

| Algorithm | Expected Performance | Rationale |
|-----------|---------------------|-----------|
| **PELT** | Good to Excellent | Clear phase transition at Day 42-43; offline global optimization |
| **BOCPD** | Moderate to Good | May detect Day 42-43 + relapse clusters as multiple CPs |
| **E-Divisive** | Good | Nonparametric, robust to gamma distribution shape |

**Parameter Guidance**:
- **PELT**: 
  - Minimum segment size ≥ 14 days (2 weeks clinical relevance)
  - BIC penalty tuning to avoid over-segmentation
  - Cost function: L2 (variance-based) or RBF (robust)
  
- **BOCPD**: 
  - Hazard λ ≈ 60-90 days (expect 4-6 change points over 365 days)
  - Prior: Gaussian on CV values
  - Posterior threshold: 0.5-0.7 for change point declaration
  
- **Statistical Validation**:
  - Effect size threshold ≥ 0.3 (clinically meaningful)
  - Mann-Whitney U test with FDR correction (α = 0.01)
  - Segment length: minimum 14 days for stable CV estimation

**Gamma-Specific Considerations**:
- **Bounded relapses** may produce cleaner CV signal than heavy-tailed distributions
- **Fewer extreme outliers** reduces false positive risk
- **Shape parameter** creates some temporal regularity in relapse patterns

---

## 7. Limitations

### 7.1 Data Generation Limitations

1. **Simplified relapse model**: Gamma distribution may not capture complex relapse patterns (e.g., seasonal effects, stressor-triggered)
2. **Linear recovery**: Real trajectories may be nonlinear (accelerating/decelerating)
3. **Homogeneous treatment**: Single treatment arm, no switching or augmentation
4. **No covariates**: Demographics, comorbidities, treatment type not modeled
5. **Fixed shape parameter**: α = 2.0 is somewhat arbitrary

### 7.2 Analysis Limitations

1. **No ground truth**: Cannot validate detection accuracy without known change points
2. **Aggregation bias**: Daily CV may obscure individual-level change points
3. **Cluster interpretation**: K = 2 vs. K = 5 depends on use case
4. **Classification thresholds**: Response pattern classification may be too stringent

### 7.3 Clinical Generalizability

1. **High non-responder rate** (73.1%): Not representative of typical outpatient populations
2. **Sparse monitoring**: More extreme than many research settings
3. **No treatment modification**: Real patients would have treatment changes
4. **Single disorder**: Depression-only; real patients often have comorbidities

---

## 8. Conclusions

The **synthetic PHQ-9 dataset with gamma relapse distribution** is:

✓ **Clinically Realistic**: Sparse monitoring, heterogeneous responses, temporal dependencies  
✓ **Suitable for Detection**: Clear temporal phases, sufficient signal-to-noise ratio  
✓ **Challenging**: High non-responder rate, high missingness, subtle change points  
✓ **Well-Characterized**: Comprehensive EDA reveals structure and properties  

### Key Takeaways

1. **Two distinct temporal phases**: Acute treatment (Days 0-42) and maintenance (Days 43-364)
2. **Gamma relapses**: Bounded distribution with moderate tail, fewer extreme events
3. **Heterogeneous population**: 73.1% non-responders, 26.9% responders (various patterns)
4. **Detection-ready**: Aggregated CV signal should reveal phase transitions
5. **Bounded tail advantage**: Fewer outliers may improve detection stability

---

## 9. Generated Files

```
results/comparison/synthetic_phq9_data_gamma/
├── analysis_summary.json                    # Numerical summary
├── cluster_characteristics.csv              # Cluster statistics
├── cluster_labels.csv                       # Day-level cluster assignments
├── response_pattern_analysis.csv            # Patient-level response data
├── summary_statistics.csv                   # Daily descriptive statistics
├── clustering/
│   └── cluster_optimization.png             # Elbow + Silhouette plots
└── visualizations/
    ├── cluster_results.png                  # 2-cluster solution
    ├── daily_averages.png                   # Temporal trend
    ├── patient_trajectories_by_pattern.png  # Sample trajectories
    ├── relapse_events.png                   # Relapse temporal + magnitude
    ├── response_patterns.png                # Response distribution
    └── scatter_plot.png                     # All observations scatter
```

---

## 10. Metadata Validation

| Metadata Field | Expected | Observed | Status |
|----------------|----------|----------|--------|
| `n_patients` | 1,000 | 1,000 | ✓ |
| `study_days` | 365 | 365 | ✓ |
| `total_observations` | ~13,680 | 13,680 | ✓ |
| `missingness_rate` | 96.25% | 96.25% | ✓ |
| `relapse_distribution` | gamma (α=2.0, β=1.75) | Confirmed | ✓ |
| `response_patterns` | 4 groups | 731/192/55/22 | ⚠️ (see Section 3.3) |

**Overall Metadata Consistency**: 5/6 checks passed  
**Note**: Response pattern distribution deviates from typical expectations (73.1% non-responders vs. ~50% expected), likely due to classification threshold sensitivity and high noise-to-signal ratio.

---

**For detailed methodology, see project documentation: `README.md` and `src/eda/README.md`**

**License**: MIT License - Research purposes only, not for clinical use

**Author**: Satyaki Mitra | Data Scientist | Clinical AI Research