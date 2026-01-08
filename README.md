# Temporal Change-Point Detection on PHQ-9 Data  
**Multi-Model, Statistically-Validated, Explainable Framework**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **An evidence-first, multi-model change-point detection system for analyzing population-level mental health trajectories using PHQ-9 scores.**  
>  
> Supports *offline* and *online* detectors (PELT, BOCPD), statistical validation, cross-model agreement, and explainable model selection.

---

## 🎯 Problem Statement

Mental-health data collected via **PHQ-9 questionnaires** is:

- sparse
- irregular
- heterogeneous
- clinically sensitive

The goal of this project is to **identify statistically and clinically meaningful temporal shifts** in population-level PHQ-9 behavior across time — *not merely detect mathematical discontinuities*.

This repository implements a **config-driven framework** that:

- Aggregates sparse PHQ-9 observations
- Detects change points using **multiple complementary algorithms**
- Validates detected changes statistically or probabilistically
- Selects the *best model* using transparent, explainable criteria

---

## 🧠 Key Design Philosophy

### ❌ What this system is NOT
- A black-box classifier
- A single-algorithm demo
- An accuracy-optimized ML benchmark

### ✅ What this system IS
- A **decision-support system**
- A **screening + validation pipeline**
- A **human-interpretable statistical framework**
- A **portfolio-grade system design artifact**

---

## 🧩 Supported Detection Paradigms

| Detector | Type | Strength |
|--------|------|---------|
| **PELT** | Offline, Frequentist | Global optimal segmentation, interpretable |
| **BOCPD** | Online, Bayesian | Real-time detection, uncertainty-aware |

Each detector:
- runs independently
- produces self-contained results
- does *not* know about other detectors

Cross-model reasoning happens **only at selection time**.

---

## 📐 Aggregation Strategy

### Why Coefficient of Variation (CV)?

\[
\text{CV}(t) = \frac{\sigma_t}{\mu_t}
\]

Chosen because:

- normalizes across heterogeneous baselines
- captures *instability*, not just mean shifts
- robust to sparse participation
- interpretable at population level

---

## 🧪 Validation Strategy

### PELT (Frequentist)
- Before-vs-after hypothesis testing
- Automatic test selection (t-test / Wilcoxon)
- Multiple-testing correction
- Effect size filtering (Cohen’s d)

### BOCPD (Bayesian)
- Posterior change-point probability
- Persistence-based validation
- Posterior mass summaries

---

## 🧠 Model Selection (New)

Instead of trusting a single algorithm, the system:

1. Runs **multiple detector variants**
2. Scores each model on comparable metrics
3. Computes **cross-model agreement**
4. Produces a ranked, explainable outcome

This avoids:
- overfitting to one detector
- arbitrary hyperparameter trust
- silent failures

---

## 📂 Project Structure (Current)

```text
phq9_analysis/
├── config/
│   ├── detection_config.py
│   └── model_selection_config.py
│
├── detection/
│   ├── detector.py              # Executor only
│   ├── pelt_detector.py
│   ├── bocpd_detector.py
│   ├── penalty_tuning.py
│   ├── statistical_tests.py
│   ├── visualizations.py
│   └── model_selector.py
│
├── scripts/
│   └── run_detection.py
│
├── data/
├── results/
├── logs/
└── docs/
    ├── architecture.md
    ├── usage_guide.md
    └── literature_references.md
```

---

## 🚀 Quick Start

```bash
python scripts/run_detection.py
```

For detailed usage and configuration, see docs/usage_guide.md.

---

## 📜 License

MIT License

---

## 🙋 Author

Satyaki Mitra
Senior Data Scientist | Applied Statistics | Clinical AI

---

