# Usage Guide

## Basic Execution

```bash
python scripts/run_detection.py
```

Runs:
	•	PELT + BOCPD
	•	validation
	•	visualization
	•	model selection

---

## Run Only One Detector

```bash
python scripts/run_detection.py \
  --execution-mode single \
  --detectors pelt
```

---

## Change PELT Cost Function

```bash
python scripts/run_detection.py \
  --cost-model l2
```

---

## Adjust Statistical Strictness

```bash
python scripts/run_detection.py \
  --alpha 0.01 \
  --correction bonferroni
```

---

## Output Control

```bash
python scripts/run_detection.py \
  --output-dir results/experiment_01
```

---

## Recommended Workflow

- 1.	Run default comparison
- 2.	Inspect agreement plots
- 3.	Review model selector explanation
- 4.	Lock configuration
- 5.	Archive results

---

## Common Pitfalls

- Do not over-tune penalties
- Do not ignore validation summaries
- Do not trust raw change-point counts

---
