<div align="center">

# Detecting Regime Shifts in Mental Health Monitoring Data
## A Rigorous End-to-End Pipeline for Change Point Detection in Sparse Longitudinal PHQ-9 Data

> **Key Insight:** We can detect population-level mental health shifts from sparse clinical data—identifying when treatments work, policies impact outcomes, and interventions are needed—without forecasting individual trajectories.

**Author:** Satyaki Mitra  
**Domain:** Healthcare AI | Clinical Informatics | Time Series Analysis  
**Methods:** Bayesian & Frequentist Change Point Detection  

</div>

---

## 🎯 The Problem: When Conventional Time Series Methods Fail

### **The Clinical Reality**

Imagine you're monitoring 1,000 patients with depression over a year. Each patient completes the PHQ-9 questionnaire—a validated 9-item depression screening tool—every 2-8 weeks. Your goal isn't to predict what score each patient will get next week. Instead, you need to answer a fundamentally different question:

> **"When did the population's mental health dynamics fundamentally change?"**

This could signal:
- A new treatment protocol taking effect
- Seasonal patterns affecting symptom severity
- Policy changes impacting care quality
- Environmental stressors (e.g., pandemic, economic crisis)

### **Why Traditional Methods Don't Work**

**The Data Challenge:**
- **96% missingness** - Patients assessed biweekly to monthly, not daily
- **Irregular intervals** - Real-world scheduling varies (2 weeks, 3 weeks, 8 weeks)
- **Individual sparsity** - Each patient has only ~14 observations over 365 days
- **High heterogeneity** - Some patients improve rapidly, others worsen, most stay the same


**What This Rules Out:**

| Method | Why It Fails |
|--------|--------------|
| **ARIMA/SARIMA** | Requires regular intervals and stationarity assumptions |
| **LSTM/RNN** | Needs dense sequences; 96% missingness destroys gradient flow |
| **Prophet** | Designed for forecasting with seasonality, not structural breaks |
| **VAR/State-Space** | Assumes known dynamics; we seek unknown breakpoints |
| **Traditional ML** | Insufficient per-patient data for individual modeling |

**The Key Insight:**  
We don't need to forecast individual trajectories. We need to detect when the **population distribution** shifts—a fundamentally different problem requiring a different approach.

---

## 💡 The Solution: Change Point Detection on Aggregated Statistics

### **The Core Idea**

Instead of modeling individual patients (impossible with 96% missingness), we:

1. **Aggregate daily** - Compute population statistics each day (mean, variance, coefficient of variation)
2. **Detect shifts** - Identify when these statistics change fundamentally
3. **Validate rigorously** - Use statistical tests to separate signal from noise

**Example:**
```
Day   Mean PHQ-9   CV (σ/μ)   Interpretation
0     16.2         0.35       Baseline: High severity, high variance
24    15.8         0.31       ← Potential change point (early responders diverging)
57    14.1         0.39       ← Major change point (peak heterogeneity)
133   11.9         0.44       ← Maintenance phase (sustained high variance)
```

### **Why This Works**

**Advantages:**
- **Handles sparsity:** Daily aggregates have ~38 patients per day (sufficient statistics)
- **No individual forecasting needed:** We analyze population-level patterns
- **Clinically meaningful:** Population shifts indicate system-wide changes
- **Statistically testable:** Can validate with hypothesis tests (Mann-Whitney U, permutation tests)

**What To Detect:**
- Transition from homogeneous (everyone similar) to heterogeneous (responders vs. non-responders)
- Timing of treatment effects (when do outcomes start diverging?)
- Maintenance phase consolidation (when does the population stabilize?)

---

## 🚧 The Ground Truth Problem: A Blessing in Disguise

### **Reality: No Ground Truth Exists**

In real-world mental health monitoring:
- **No labeled change points** - We don't know the "true" dates when population dynamics shifted
- **No external validation dataset** - Each clinical setting is unique
- **Subjective clinical judgment** - Experts may disagree on when changes occurred

**This is actually common in many real-world ML problems:**
- Anomaly detection (no labeled anomalies)
- Clustering (no true cluster assignments)
- Topic modeling (no ground truth topics)
- **Change point detection** (no known transition times)

### **How This Has Been Addressed**

**1. Synthetic Data with Known Properties (Not Known Change Points)**

The data have been generated with:
- **Known parameters:** AR(1) coefficient = 0.70, recovery rate = -0.075 points/day
- **Known response patterns:** 30% early, 35% gradual, 15% late, 20% non-responders
- **Known relapse distribution:** Exponential (λ=1/3.5) or Gamma (α=2, β=1.75)

But **intentionally any change point hasn't been hard-coded**. They emerge naturally from:
- Response heterogeneity (some improve, some don't)
- Plateau logic (responders stabilize)
- Relapse dynamics (random setbacks)

**This is the gold standard for methodological research** - Through the data-generating process, the algorithm itself discovers patterns without being told where to look.

**2. Cross-Method Validation**

Instead of comparing to "ground truth," the data have been validated through:
- **Convergence:** Do multiple algorithms detect the same change points?
- **Statistical rigor:** Are detected changes significant (p<0.05 after FDR correction)?
- **Effect sizes:** Are changes large enough to matter clinically (Cohen's d ≥ 0.3)?
- **Clinical plausibility:** Do detected times align with known treatment phases (STAR*D)?

**Result:** Day 57 detected by **all 4 PELT variants** with **p=0.010** and **d=2.37**—that's strong evidence, even without "ground truth."

**3. Literature Benchmarking**

The data have been validated against clinical milestones from STAR*D (largest depression treatment trial):

| Detected CP | Day | Week | STAR*D Milestone | Alignment |
|-------------|-----|------|------------------|-----------|
| CP1 | 24 | 3.5 | Early response (weeks 2-4) | ✓ Strong |
| CP2 | 57 | 8.0 | Acute response peak (weeks 6-8) | ✓ Exact |
| CP3 | 133 | 19.0 | Plateau phase (weeks 12-20) | ✓ Strong |

**The Takeaway:**  
No ground truth isn't a limitation—it's the reality of unsupervised methods. Our validation strategy (convergence + statistics + clinical alignment) is **more rigorous** than having arbitrary labels.

---

## 🔬 The Three-Module Pipeline: From Synthesis to Detection

### **Module 1: Data Generation** (`src/generation/`)

**Purpose:** Create clinically realistic synthetic PHQ-9 data for methodological validation.

**Why Synthetic Data?**
- Real PHQ-9 data is **privacy-restricted** (HIPAA, GDPR)
- Need **control over parameters** for methodological testing
- Want **reproducibility** (same random seed = same results)
- Require **known data-generating process** for validation

**What Makes It Realistic?**

1. **Gap-Aware AR(1) Temporal Model**
   ```
   PHQ9(t) = α^Δt × PHQ9(t-1) + ε
   where α = 0.70 (temporal correlation)
         Δt = days since last observation
   ```
   - Correlation decays exponentially with gap length
   - Nearby observations (Δt=7 days) → correlation ~0.70
   - Distant observations (Δt=30 days) → correlation ~0.03

2. **Four Response Patterns** (based on STAR*D)
   - **Early responders** (30%): Rapid improvement in 2-6 weeks
   - **Gradual responders** (35%): Steady decline over 6-12 weeks
   - **Late responders** (15%): Delayed response after 12+ weeks
   - **Non-responders** (20%): Minimal or no improvement

3. **Plateau Logic**
   - Responders stabilize after improvement (variance < threshold for 14 days)
   - Scores fluctuate around plateau level
   - Realistic maintenance phase behavior

4. **Three Relapse Distributions**
   - **Exponential:** Memoryless, random stressors, heavy tail
   - **Gamma:** Stress accumulation, bounded tail, shape parameter α=2
   - **Lognormal:** Rare catastrophic events, multiplicative processes

5. **Realistic Missingness**
   - **Structural sparsity:** 93% missing by design (biweekly to monthly)
   - **Informative dropout:** 21% patients drop out (STAR*D-aligned)
   - **MCAR component:** Random 3% missingness

**Validation Results:**
```
Autocorrelation: 0.36 ± 0.44 (14-day gap) ✓ Expected: 0.30-0.70
Baseline Severity: PHQ-9 14.6 ± 4.2 ✓ Expected: 13-19
Missingness: 96.2% ✓ Expected: ≤96.5%
Data Integrity: No violations ✓
```

**Output:**
- 3 datasets (exponential, gamma, lognormal)
- 1,000 patients × 365 days each
- Metadata sidecar JSON (full provenance tracking)

---

### **Module 2: Exploratory Data Analysis** (`src/eda/`)

**Purpose:** Validate data quality, characterize temporal structure, and rank distributions for downstream analysis.

**Why EDA Before Detection?**

You wouldn't run a regression without checking assumptions. Similarly, we need to:
- Verify data quality (no corruption, realistic ranges)
- Understand temporal structure (clustering, trends)
- Identify response patterns (validate against metadata)
- Compare distributions (select best for detection)

**What To Analyze:**

1. **Data Validation**
   - Score range: All values in [0, 27] ✓
   - Temporal ordering: No future dates ✓
   - Patient consistency: 1,000 patients across all days ✓
   - Metadata alignment: Observed matches expected ✓

2. **Temporal Clustering**
   - **Method:** KMeans on daily features (mean, CV, severity percentiles)
   - **Optimal K:** Silhouette analysis suggests K=2
   - **Boundary:** Day 42-43 (week 6) across all distributions
   - **Clinical interpretation:** Acute treatment (0-6 weeks) vs. maintenance (6+ weeks)

3. **Response Pattern Classification**
   ```python
   Classification Logic:
   - Early responder: slope ≤ -0.08, improvement ≥ 50%
   - Gradual responder: -0.08 < slope ≤ -0.04, improvement ≥ 50%
   - Late responder: -0.04 < slope ≤ -0.02, improvement ≥ 50%
   - Non-responder: improvement < 50% OR slope > -0.02
   ```
   
   **Observed distribution:** 71-73% non-responders (intentional treatment-resistant population)

4. **Relapse Detection**
   - Criteria: ≥3-point increase, 7-30 day gap since last observation
   - Results: 580-619 events, 41-43% of patients affected
   - Temporal pattern: Early clustering (Days 20-100) for exponential

5. **Multi-Dataset Comparison**
   - **Scoring dimensions:** Temporal stability (35%), clinical realism (30%), statistical quality (20%), metadata consistency (15%)
   - **Winner:** Gamma distribution (91.36/100 composite score)
   - **Key advantage:** Best temporal stability (87.1%) due to bounded tail

**Key Findings:**

| Metric | Exponential | Gamma | Lognormal |
|--------|-------------|-------|-----------|
| Total Observations | 13,718 | 13,680 | 13,711 |
| Optimal K (Silhouette) | 2 | 2 | 2 |
| Cluster Boundary | Day 42 | Day 42-43 | Day 42 |
| Non-Responder % | 71.8% | 73.1% | 72.9% |
| Relapse Rate | 43.4% | 42.8% | 41.2% |
| **Composite Score** | 91.20 | **91.36** | 91.20 |

**Visualizations Generated:**
- Daily average trends with severity bands
- Cluster optimization (elbow + silhouette)
- Response pattern distribution
- Patient trajectory samples by pattern
- Relapse temporal + magnitude histograms
- Cross-dataset comparison dashboard

---

### **Module 3: Change Point Detection** (`src/detection/`)

**Purpose:** Detect and rigorously validate significant temporal shifts in population dynamics.

**Why Two Algorithms?**

Different algorithms have different strengths:

| Algorithm | Type | Strengths | Weaknesses |
|-----------|------|-----------|------------|
| **PELT** | Offline, frequentist | Exact optimization, interpretable, proven | Requires full data upfront |
| **BOCPD** | Online, Bayesian | Real-time capable, uncertainty quantification | Sensitive to priors, computationally expensive |

**Using both provides:**
- **Convergence validation:** If both detect the same CPs, strong evidence
- **Robustness testing:** Check sensitivity to methodological choices
- **Complementary perspectives:** Frequentist hypothesis testing + Bayesian posterior probabilities

**Algorithm Details:**

**1. PELT (Pruned Exact Linear Time)**

**How it works:**
- Minimizes: `Cost = Σ[segment costs] + β × n_changepoints`
- Penalty β tuned via BIC (Bayesian Information Criterion)
- Pruning: Removes impossible candidates for O(n) complexity

**Cost functions tested:**
- **L1 (median-based):** Robust to outliers, O(n) complexity ✓ **Winner**
- **L2 (mean-based):** Least-squares, sensitive to outliers
- **RBF (kernel):** Nonlinear, captures complex patterns, O(n²) complexity
- **AR (autoregressive):** Models temporal correlation explicitly

**Parameter tuning:**
```python
Penalty range: [0.01, 10.0] (50 values, log scale)
BIC optimization: penalty = 0.1677 (minimizes BIC = -2015.3)
Minimum segment: 5 days (clinically meaningful window)
```

**Results (Gamma dataset, PELT-L1):**

| Change Point | Day | Week | CV Before | CV After | Change | Cohen's d | P-value (FDR) |
|--------------|-----|------|-----------|----------|--------|-----------|---------------|
| CP1 | 24 | 3.5 | 0.262 | 0.296 | +13% | 1.86 | 0.033 ✓ |
| **CP2** | **57** | **8.0** | **0.313** | **0.387** | **+24%** | **2.37** | **0.010 ✓** |
| CP3 | 133 | 19.0 | 0.348 | 0.438 | +26% | 2.41 | 0.010 ✓ |

**2. BOCPD (Bayesian Online Change Point Detection)**

**How it works:**
- Maintains run-length posterior: `P(r_t | y_{1:t})`
- Updates belief about "time since last change point"
- Detects CP when posterior mass concentrates at r=0

**Hazard function:**
```
P(r_t = 0 | r_{t-1}) = 1/λ
where λ = expected run length (e.g., 75 days = ~5 CPs per year)
```

**Implementation:**
- **Student-t likelihood (df=3):** Robust to heavy-tailed CV distribution
- **Hazard tuning:** Converges to λ=100 (both heuristic and predictive methods)
- **Hazard range:** [100, 500] with enforced minimum of 50
- **Adaptive threshold:** mean + 1.5σ (≈0.011), data-driven
- **Result:** 2 CPs detected at Days 32, 69 (early-phase transitions)

**Statistical Validation:**

For each detected change point:

1. **Hypothesis Test**
   - Mann-Whitney U test (non-parametric, robust)
   - Null hypothesis: CV distributions before/after are identical
   - Alternative: Distributions differ
   - **Result:** All 3 CPs have p<0.05

2. **Multiple Testing Correction**
   - FDR (False Discovery Rate) via Benjamini-Hochberg
   - Controls expected proportion of false discoveries
   - **Result:** All 3 CPs survive correction (p<0.05 after FDR)

3. **Effect Size**
   - Cohen's d: Standardized mean difference
   - Threshold: d ≥ 0.3 (clinically meaningful)
   - **Result:** d = 1.86, 2.37, 2.41 (very large effects)

**Model Selection:**

**Composite scoring** (weights optimized for clinical detection):
```
Score = 0.30 × (n_significant / n_total)     [Significance]
      + 0.30 × (mean_effect_size / max)      [Effect Size]
      + 0.25 × (cross-model agreement)       [Stability]
      + 0.15 × (parsimony: fewer CPs better) [Simplicity]
```

**Rankings (Gamma dataset):**

| Rank | Model | Score | CPs | Significant | Mean d |
|------|-------|-------|-----|-------------|--------|
| 1 | **PELT-L1** | **0.806** | 3 | 3 (100%) | 2.21 |
| 2 | PELT-RBF | 0.806 | 3 | 3 (100%) | 2.21 |
| 3 | PELT-L2 | 0.718 | 3 | 2 (67%) | 1.67 |
| 4 | PELT-AR | 0.591 | 2 | 1 (50%) | 1.73 |
| 5 | BOCPD-H | 0.340 | 2 | Bayesian | N/A |
| 6 | BOCPD-P | 0.340 | 2 | Bayesian | N/A |

**Winner: PELT-L1** (chosen over tied PELT-RBF for computational efficiency)

---

## 📊 Key Results: What Has Been Found

### **1. Dataset Quality: All Three Excellent**

**Validation Summary:**

| Check | Exponential | Gamma | Lognormal | Status |
|-------|-------------|-------|-----------|--------|
| Autocorrelation | 0.360 ± 0.438 | 0.365 ± 0.443 | 0.363 ± 0.443 | ✓ PASS |
| Baseline Severity | 14.63 ± 4.24 | 14.67 ± 4.25 | 14.65 ± 4.22 | ✓ PASS |
| Missingness | 96.24% | 96.25% | 96.24% | ✓ PASS |
| Score Range | [0, 26.88] | [0, 26.75] | [0, 26.98] | ✓ PASS |

**Consistency:** <1% variation across distributions—excellent reproducibility.

**Winner:** Gamma (91.36/100 composite score) for best temporal stability (87.1%).

---

### **2. Three Major Change Points Detected**

**Timeline:**
```
Day 0    Day 24    Day 57      Day 133              Day 365
  |        |         |            |                    |
  └────────┼─────────┼────────────┼────────────────────┘
  Baseline  CP1      CP2          CP3
  
  Phase 1   Phase 2   Phase 3       Phase 4
  ────────  ────────  ────────────  ─────────────────
  Homog.    Early     Peak          Maintenance
  Low CV    Response  Divergence    High CV
  (0.26)    (+19%)    (+24%)        (+26%)
```

**Clinical Interpretation:**

**Phase 1 (Days 0-24): Baseline Homogeneity**
- CV = 0.262 (low variability)
- Population has similar symptom severity
- Pre-treatment or early initiation

**Phase 2 (Days 25-57): Early Response Divergence**
- CV = 0.313 (+19%)
- ~30% early responders improving
- Non-responders stable or worsening
- First stratification visible

**Phase 3 (Days 58-133): Peak Heterogeneity**
- CV = 0.387 (+24%)
- Maximum outcome divergence
- Gradual responders (~35%) reaching plateau
- Clear responder/non-responder split
- **Most critical monitoring period**

**Phase 4 (Days 134-365): Maintenance**
- CV = 0.438 (+26%)
- Sustained high variability
- Late responders (~15%) reaching plateau
- Some early responders relapsing
- Permanent population stratification

**Overall:** Population variability increased **67%** from baseline to maintenance.

---

### **3. PELT vs BOCPD: Complementary Paradigms**

**Cross-Model Agreement:**

| Change Point | PELT-L1 | PELT-L2 | PELT-RBF | PELT-AR | BOCPD | Agreement |
|--------------|---------|---------|----------|---------|-------|-----------|
| **Day 57-58** | ✓ | ✓ | ✓ | ✓ | ✗ | **100% (PELT)** |
| Day 24 | ✓ | ✓ | ✓ | ✗ | ✗ | 75% |
| Day 133 | ✓ | ✗ | ✓ | ✗ | ✗ | 50% |
| Day 149 | ✗ | ✓ | ✗ | ✓ | ✗ | 50% |
| Days 32, 69 | ✗ | ✗ | ✗ | ✗ | ✓ (2 CPs) | **0% (different phase)** |

**Key Findings:**
- **Day 57 is universally agreed upon** by all 4 PELT variants
- **BOCPD detects different phase** - early transient shifts (Days 32, 69) vs PELT's sustained regime changes
- **PELT variants converge** despite different cost functions → robust
- **Paradigm difference:** Sequential (BOCPD) vs global optimization (PELT)

**BOCPD Characteristics:**
- Uses **Student-t likelihood (df=3)** for robust handling of heavy-tailed CV data
- Hazard tuning converges to **λ=100** (both heuristic and predictive methods agree)
- **Adaptive posterior threshold** (mean + 1.5σ ≈ 0.011) - data-driven, not fixed
- Detects **2 early-phase change points** (Days 32, 69)
- Different sensitivity than PELT: sequential algorithm captures transient shifts vs. sustained regime changes

---

### **4. Alignment with Clinical Literature**

**Comparison to STAR*D (Largest Depression Trial):**

| STAR*D Milestone | Expected Week | Detected CP | Alignment |
|------------------|---------------|-------------|-----------|
| Early response begins | Week 2-4 | Day 24 (Week 3.5) | ✓ Strong |
| Acute response peak | Week 6-8 | Day 57 (Week 8) | ✓ Exact |
| Plateau consolidation | Week 12-20 | Day 133 (Week 19) | ✓ Strong |

**Conclusion:** Detected change points **strongly align** with established clinical treatment phases, providing external validation.

---

### 🔑 **Key Takeaways**

- **Population > Individual:** With sparse data, aggregate statistics reveal what individual trajectories cannot
- **Three Phases Matter:** Early response (Week 3-4), peak heterogeneity (Week 8), and maintenance (Week 19)
- **PELT Wins:** For this application, offline frequentist methods outperform online Bayesian approaches
- **Gamma Distribution:** Provides the most temporally stable synthetic data for validation

---

### 🎯 **Conclusion & Impact**

**This pipeline demonstrates that even with extremely sparse, real-world clinical data, we can detect meaningful population-level shifts. This approach can help health systems identify when treatments are working, when policies impact outcomes, and when to intervene at a population level—all without needing to forecast individual patient trajectories.**

> *Interested in the code or want to try it on your own data? Check out the [GitHub repository](https://github.com/satyaki-mitra/phq9-changepoint-detection).*

---