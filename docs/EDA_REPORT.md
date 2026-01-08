# Exploratory Data Analysis (EDA) Report  
**Project:** PHQ-9 Longitudinal Mental Health Analysis  
**Dataset Type:** Synthetic, Clinically Grounded  
**Analysis Scope:** Population-level temporal behavior  
**Generated On:** 2026-01-08  

---

## 1. Objective of This Analysis

The goal of this Exploratory Data Analysis (EDA) is to:

- Understand the **temporal behavior** of PHQ-9 depression scores at the population level
- Identify **distinct severity regimes** over time
- Validate whether the synthetic data behaves in a **clinically and statistically realistic** manner
- Assess suitability of the dataset for downstream temporal and statistical modeling

This analysis is intentionally **descriptive and regime-oriented**, not predictive.

---

## 2. Dataset Overview

### 2.1 Data Shape

| Attribute | Value |
|---------|------|
| Number of days | 365 |
| Number of patients | 1,000 |
| Data matrix shape | 365 × 1,000 |
| Total observations | 4,600 |
| Missing values | 98.74% |

> Each patient contributes only a small number of PHQ-9 assessments, intentionally reflecting real-world adherence and dropout patterns seen in mental-health studies.

---

## 3. Clinical Context: PHQ-9 Scale

The Patient Health Questionnaire-9 (PHQ-9) is a validated instrument for assessing depression severity.

| Score Range | Severity |
|------------|----------|
| 0–4 | Minimal |
| 5–9 | Mild |
| 10–14 | Moderate |
| 15–19 | Moderately Severe |
| 20–27 | Severe |

All analyses respect this bounded, ordinal clinical scale.

---

## 4. Global Score Distribution

### 4.1 Scatter Distribution Across Time

The scatter visualization of all observed PHQ-9 scores shows:

- High variance and frequent **moderate–severe scores** in early days
- Progressive **compression toward lower scores** over time
- A sharp decline in the frequency of scores ≥ 20 (severe)

**Interpretation:**  
The population starts in a higher-severity, unstable state and gradually transitions toward lower, more stable symptom levels.

---

## 5. Daily Aggregate Trends

### 5.1 Daily Mean PHQ-9 Score

- A clear **negative linear trend** is observed
- Estimated slope: **−0.038 per day**

This implies a **gradual population-level improvement** in depressive symptoms over the study period.

### 5.2 Severity Band Transitions

Over time, the daily average crosses:

- From **Moderate / Moderately Severe**
- Into **Mild**
- And eventually stabilizes near **Minimal–Mild**

This confirms **non-stationarity**, a critical property of real longitudinal mental-health data.

---

## 6. Feature Engineering for Clustering

Clustering was performed on **daily-level engineered features**, not raw patient vectors, to avoid sparsity artifacts.

### 6.1 Daily Feature Set

For each day:

- Mean PHQ-9 score
- Standard deviation
- Coefficient of variation (CV)
- Number of observations
- Proportion of severe cases (PHQ-9 ≥ 20)

This representation captures **severity, dispersion, and observation density** in a clinically interpretable way.

---

## 7. Optimal Cluster Selection

Multiple cluster-selection methods were evaluated:

| Method | Suggested K |
|------|-------------|
| Elbow Method | 6 |
| Silhouette Analysis | **2** |

**Final Choice:** **K = 2**, based on:
- Maximum silhouette score
- Clean temporal separation
- Interpretability as clinical regimes

---

## 8. Temporal Regime Identification

### Cluster 0 — High-Severity / Volatile Phase

- Dominates early study period
- Higher mean PHQ-9 scores
- Larger variance
- Greater proportion of severe cases
- Represents **acute or unstable depression phase**

### Cluster 1 — Low-Severity / Stable Phase

- Dominates later study period
- Lower mean and variance
- Mostly minimal to mild scores
- Represents **recovery or stabilization phase**

Cluster transitions align closely with the global downward trend, validating their temporal meaning.

---

## 9. Cluster Characteristics Summary

Key differentiators between clusters:

- Mean severity
- Score dispersion
- Observation density
- Temporal continuity

No long-range interleaving of clusters is observed, indicating **strong temporal coherence** rather than noise-driven segmentation.

---

## 10. Statistical Properties Observed

The dataset exhibits the following intentional properties:

- Strong sparsity with informative missingness
- Bounded, non-Gaussian score distribution
- Temporal non-stationarity
- Population-level improvement trend
- Regime-like behavior rather than IID samples

These properties mirror real-world depression monitoring data.

---

## 11. Limitations (Explicitly Acknowledged)

- No patient-level trajectory inference is attempted
- No causal claims are made
- Clusters represent **population regimes**, not individuals
- Missingness prevents traditional longitudinal modeling assumptions

These limitations are **by design**, not deficiencies.

---

## 12. Artifacts Generated

```
results/eda
├── analysis_summary.json
├── cluster_characteristics.csv
├── clustering
│   └── cluster_optimization.png
├── summary_statistics.csv
└── visualizations
    ├── cluster_results.png
    ├── daily_averages.png
    └── scatter_plot.png
```

All outputs are reproducible and auditable.

---

## 13. Final Conclusion

This EDA demonstrates that the synthetic PHQ-9 dataset:

- Behaves in a **clinically realistic** manner
- Exhibits **clear temporal regimes**
- Supports **robust population-level analysis**
- Is well-suited for advanced temporal methods (e.g., change-point detection, regime modeling)

The data is **fit for purpose** and analytically defensible.

---