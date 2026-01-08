# PHQ-9 Synthetic Data Generation — Conceptual Overview

## 1. Purpose of This README

This document explains **the data itself**, not the implementation.

Its goal is to ensure that a reader can understand:

- What PHQ-9 represents clinically
- How the synthetic data is generated
- What modeling and statistical assumptions are made
- What mathematical structure governs the data
- What statistical properties the dataset exhibits
- What kind of analyses the data is suitable for

A reader should be able to reason about the dataset **without seeing any code**.

---

## 2. What Is PHQ-9?

The **Patient Health Questionnaire-9 (PHQ-9)** is a validated clinical instrument used to:

- Screen for depression
- Quantify depression severity
- Monitor symptom change over time
- Evaluate treatment response

It is widely used in:
- Clinical trials
- Outpatient mental health care
- Longitudinal monitoring studies

### PHQ-9 Score Interpretation

| Score Range | Clinical Meaning |
|-----------|------------------|
| 0–4 | Minimal depression |
| 5–9 | Mild depression |
| 10–14 | Moderate depression |
| 15–19 | Moderately severe depression |
| 20–27 | Severe depression |

PHQ-9 is:
- Self-reported
- Bounded (0–27)
- Repeated over time
- Non-Gaussian in practice

This generator produces **longitudinal PHQ-9 trajectories**, not isolated measurements.

---

## 3. What Kind of Dataset Is Generated?

The generated dataset represents:

- A **population of patients** (default: 1000)
- Observed over a **365-day study window**
- With **sparse and irregular survey completion**

### Core Characteristics

- Each patient completes **2–7 PHQ-9 surveys total**
- Observations occur on **irregular days**
- Missing values are expected and meaningful
- Patients differ in:
  - Baseline severity
  - Treatment response rate
  - Symptom stability
  - Dropout behavior

This structure mirrors **real-world mental health monitoring**, not idealized daily data.

---

## 4. Core Modeling Assumptions

### Clinical Assumptions

- Patients enter care with moderate to severe depression
- Symptoms improve gradually under treatment
- Treatment response is heterogeneous
- Symptoms exhibit day-to-day persistence
- Temporary relapses occur
- Some patients discontinue follow-up

### Statistical Assumptions

- PHQ-9 scores are bounded and non-Gaussian
- Observations are irregular in time
- Data is missing due to both random and informative mechanisms
- Temporal dependence decays with observation gaps
- Population statistics are computed from observed values only

---

## 5. Latent Treatment Trajectory

Each patient follows an expected recovery trend:

\[
\mu_t = \text{baseline} + \text{recovery\_rate} \times (t - t_0)
\]

Where:
- `baseline` is initial PHQ-9 severity
- `recovery_rate < 0` represents improvement
- `t` is the calendar day
- `t_0` is treatment start

This represents **gradual symptom improvement**, not abrupt change.

---

## 6. Temporal Dependency (Gap-Aware AR(1))

Observed scores follow a **gap-aware autoregressive process**:

\[
Y_t =
\alpha^{\Delta t} Y_{t-\Delta t}
+
(1 - \alpha^{\Delta t}) \mu_t
+
\varepsilon_t
+
\text{relapse}_t
\]

### Why Gap Awareness Matters

- PHQ-9 symptoms are temporally persistent
- Observation gaps reduce correlation
- Treating sparse data as daily data overestimates stability
- This formulation preserves realism under irregular sampling

### Components

| Term | Meaning |
|----|--------|
| \( \alpha \) | Symptom persistence (≈ 0.7) |
| \( \Delta t \) | Days since last observation |
| \( \varepsilon_t \) | Measurement noise |
| `relapse_t` | Temporary symptom worsening |

---

## 7. Relapse Modeling

Relapses represent:

- Stressful life events
- Treatment interruptions
- Temporary symptom worsening

They are modeled as:
- Low-probability events
- Positive score shocks
- Heavy-tailed magnitude

Relapses introduce:
- Nonlinearity
- Realistic volatility
- Clinically plausible instability

---

## 8. Missing Data Mechanisms

Two missingness mechanisms are intentionally introduced:

### 1. Missed Appointments (MCAR)

- Random missed surveys
- Independent of symptom severity

### 2. Informative Dropout (MNAR)

- Some patients stop responding entirely
- Dropout timing follows an exponential distribution
- Later dropout is more common than early dropout

Together, these create **high sparsity**, consistent with real PHQ-9 usage.

---

## 9. Statistical Properties of the Generated Data

The dataset exhibits:

- Bounded support (0–27)
- Right-skewed distributions
- Heterogeneous variance
- Moderate but attenuated autocorrelation
- Non-stationary population trends
- Strong between-patient heterogeneity
- High missingness by design

The data is **not IID**, **not Gaussian**, and **not complete**.

---

## 10. Intended Use Cases

This dataset is suitable for:

- Temporal change-point detection
- Aggregated instability metrics (CV, variance)
- Population-level trend analysis
- Robust statistical modeling
- Clinical analytics demonstrations

It is **not intended** for:
- Supervised accuracy benchmarks
- Fully observed time-series models
- IID statistical assumptions

---

## 11. Summary

This generation module produces:

- Clinically grounded PHQ-9 trajectories
- Sparse, irregular longitudinal data
- Statistically defensible behavior
- A realistic foundation for downstream temporal analysis

The objective is **clinical realism and modeling correctness**, not synthetic perfection.