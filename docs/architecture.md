# System Architecture

This system follows a **strict separation of concerns** to ensure:

- reproducibility
- explainability
- extensibility
- auditability

---

## High-Level Architecture

```text
          ┌─────────────────────────┐
          │   run_detection.py       │
          │  (CLI / entry point)     │
          └────────────┬────────────┘
                       │
                       ▼
          ┌─────────────────────────┐
          │ ChangePointDetection     │
          │ Orchestrator             │
          │ (detector.py)            │
          └────────────┬────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼                              ▼
┌───────────────┐             ┌───────────────┐
│ PELTDetector  │             │ BOCPDDetector │
│ (offline)     │             │ (online)      │
└──────┬────────┘             └──────┬────────┘
       │                              │
       ▼                              ▼
┌───────────────┐             ┌───────────────┐
│ Frequentist   │             │ Bayesian      │
│ Validation    │             │ Validation    │
└──────┬────────┘             └──────┬────────┘
       └──────────────┬──────────────┘
                      ▼
            ┌────────────────────┐
            │ ModelSelector       │
            │ (model_selector.py)│
            └────────────────────┘

```

## Design Rules

- 1. Detectors Are Isolated
	•	No detector imports another
	•	No shared state
	•	No hidden agreement logic

- 2. Orchestrator Is Dumb
	•	Executes
	•	Collects
	•	Visualizes
	•	Does not decide

- 3. Selection Is Explicit
	•	All ranking logic lives in model_selector.py
	•	All weights live in model_selection_config.py

⸻

## Why This Matters

This design:
	•	prevents accidental coupling
	•	allows algorithm substitution
	•	supports regulatory review
	•	mirrors production ML governance