# Agentic AI System for PDE Discovery

## 🎯 Overview
This system uses a multi-agent AI framework (AutoGen) to automatically discover governing partial differential equations (PDEs) from spatiotemporal data. It combines Vision-Language Models (VLMs), Large Language Models (LLMs), and symbolic regression (PySINDy) in an iterative, reinforcement learning-like loop.

---

## 🏗️ System Architecture

### **Multi-Agent Pipeline Flow**

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INITIATES SYSTEM                         │
│              (Provides contour/surface plots)                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT 1: Contour_Plot_Analyser (VLM)                           │
│  ────────────────────────────────────────                        │
│  • Analyzes visual patterns in plots                             │
│  • Identifies: shocks, waves, dispersion, nonlinearity          │
│  • Outputs: Structured text description                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT 2: LLM (PDE Hypothesis Generator) ⭐ CORE ROLE            │
│  ────────────────────────────────────────                        │
│  ROLE: Mathematical Physicist & Pattern Interpreter              │
│                                                                   │
│  INPUT:                                                           │
│    - VLM's visual analysis (e.g., "high dispersion, nonlinear    │
│      advection, chaotic waves")                                  │
│    - Feedback from Critic (if iteration > 1)                     │
│                                                                   │
│  PROCESS:                                                         │
│    1. Maps visual cues to PDE terms:                             │
│       • "Dispersion" → u_xxxx, u_xxx                             │
│       • "Nonlinear advection" → u*u_x                            │
│       • "Diffusion" → u_xx                                        │
│    2. Draws from knowledge base of canonical PDEs:               │
│       • Kuramoto-Sivashinsky: u_t = -u*u_x - u_xx - u_xxxx       │
│       • Burgers: u_t = -u*u_x + ν*u_xx                           │
│       • KdV, Fisher-KPP, etc.                                    │
│    3. Generates 10 diverse candidate PDEs combining:             │
│       • Known PDE templates                                       │
│       • Novel term combinations                                   │
│       • Physically plausible structures                           │
│                                                                   │
│  OUTPUT: 10 equations (e.g., u_t = -u*u_x - u_xx - u_xxxx)       │
│                                                                   │
│  CONSTRAINTS:                                                     │
│    - Uses ONLY allowed symbols (u, u_x, u_xx, u_xxx, u_xxxx)    │
│    - NO code generation (purely symbolic)                         │
│    - Balances exploration (new terms) & exploitation (proven)    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT 3: Engineer (Code Generator)                              │
│  ────────────────────────────────────────────                    │
│  • Receives 10 PDEs from LLM                                     │
│  • Writes Python script using PySINDy:                           │
│    1. Loads u(x,t), x, t data                                    │
│    2. Computes derivatives (u_t, u_x, u_xx, u_xxx, u_xxxx)      │
│    3. For each PDE:                                              │
│       - Fits coefficients via least-squares                      │
│       - Calculates Relative L2 Error                             │
│    4. Saves results to CSV (scores, errors, boundaries)          │
│  • Output: Executable Python code + fitted PDEs with scores      │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT 4: Executor (Code Runner)                                 │
│  ────────────────────────────────────────────                    │
│  • Runs Engineer's code locally (no Docker)                      │
│  • Returns: Exit code, output, errors                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT 5: Scientist (Quality Control)                            │
│  ────────────────────────────────────────────                    │
│  • Checks execution success (exitcode: 0)                        │
│  • Validates outputs:                                             │
│    - Files exist (score.csv, error.csv, etc.)                    │
│    - Scores are numerical                                         │
│  • Reviews physical plausibility:                                 │
│    - Do terms make sense? (e.g., u_xxxx for dispersion)          │
│    - Are coefficients reasonable?                                 │
│  • Signals: ROGER CRITIC (pass) or ROGER ENGINEER (fix code)     │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  AGENT 6: Critic (Decision Maker) 🎓 GATEKEEPER                  │
│  ────────────────────────────────────────────                    │
│  ROLE: Final arbiter of truth                                    │
│                                                                   │
│  EVALUATES:                                                       │
│    1. All 10 fitted PDEs (not just best score)                   │
│    2. Mathematical validity (well-posed, consistent)             │
│    3. Physical plausibility (terms represent real processes)     │
│    4. Fit quality (L2 error, boundary behavior)                  │
│                                                                   │
│  DECISION LOGIC:                                                  │
│    IF any PDE meets ALL criteria:                                │
│       • L2 Error ≤ 0.01                                          │
│       • Mathematically sound                                      │
│       • Physically meaningful                                     │
│       • Stable boundaries                                         │
│    THEN:                                                          │
│       → Signal TERMINATE (success!)                              │
│    ELSE:                                                          │
│       → Analyze failures (e.g., "missing dispersion term")       │
│       → Generate specific feedback for LLM                        │
│       → Signal ROGER LLM (iterate)                               │
│                                                                   │
│  FEEDBACK EXAMPLES:                                               │
│    - "High errors suggest missing u_xxxx for dispersion"         │
│    - "Coefficient on u*u_x too large; reduce nonlinearity"       │
│    - "Add damping term u_xx to stabilize"                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
        ┌───────────────┐      ┌──────────────┐
        │  TERMINATE    │      │  BACK TO LLM │
        │  (Success!)   │      │  (Iterate)   │
        └───────────────┘      └──────┬───────┘
                                      │
                                      └─────────────┐
                                                    │
                    ┌───────────────────────────────┘
                    │
                    ▼
            ┌───────────────────────────────────────┐
            │  LLM Receives Critic Feedback          │
            │  - Incorporates suggestions            │
            │  - Generates 10 NEW PDEs               │
            │  - Cycle repeats (max 100 rounds)     │
            └────────────────────────────────────────┘
```

---

## 🧠 The LLM's Role in Detail

### **Why Do We Need an LLM for PDE Discovery?**

Traditional symbolic regression (e.g., genetic programming, LASSO) explores term combinations blindly. **The LLM brings domain knowledge** to guide the search:

1. **Physical Intuition**: Knows that:
   - Burgers equation has `u*u_x` (shock formation)
   - KS equation needs `u_xxxx` (dispersion stabilization)
   - Diffusion requires `u_xx` (smoothing)

2. **Pattern Recognition**: Maps visual cues → math:
   - VLM says "chaotic waves" → LLM proposes `-u*u_x - u_xxxx`
   - VLM says "smooth diffusion" → LLM proposes `ν*u_xx`

3. **Iterative Learning**: Unlike static methods, the LLM:
   - Receives feedback ("error too high, add u_xxxx")
   - Adjusts next generation of PDEs
   - Converges faster than blind search

### **What the LLM Does NOT Do**
- ❌ Does NOT run simulations
- ❌ Does NOT compute derivatives
- ❌ Does NOT fit coefficients
- ✅ ONLY generates symbolic PDE forms (e.g., `u_t = term1 + term2`)

---

## 📊 Scoring System

### Relative L2 Error Metric
```python
error = ||u_t_true - u_t_predicted|| / ||u_t_true||
```

| Error Range | Interpretation | Action |
|-------------|----------------|--------|
| < 0.01 | ✅ Excellent | Accept if physically valid |
| 0.01–0.03 | 🟡 Good | Review carefully |
| 0.03–0.05 | 🟠 Acceptable | Likely needs refinement |
| > 0.05 | 🔴 Poor | Reject, iterate |

---

## 📁 File Structure

```
pdefinder/
├── llm4ed_agentic_ai_new_vlm_v1.ipynb  # Main notebook
├── Bayesian_PDE_Discovery_DATA/         # Data files
│   └── kuramoto_sivishinky.mat
├── surface_contour_plots/               # Visualization inputs
│   ├── KS_contour_896.png
│   └── KS_surface_896.png
├── u.npy, x.npy, t.npy                 # Loaded data arrays
├── score.csv                            # PDE scores (output)
├── error.csv                            # Error fields (output)
└── pde_sim.py                           # Last executed code
```


## 📚 Key References

1. **SINDy**: Brunton et al. (2016) - Sparse Identification of Nonlinear Dynamics
2. **AutoGen**: Microsoft Research - Multi-agent conversation framework


---



## 📝 Quick Start

```bash
# 1. Install dependencies
pip install autogen-agentchat pysindy scipy numpy

# 2. Set API key in notebook
api_key = "your-nvidia-api-key"

# 3. Run all cells sequentially

# 4. Monitor output for:
#    - VLM analysis
#    - 10 generated PDEs
#    - Fitted coefficients + scores
#    - TERMINATE signal (success!)
```

---

**Last Updated**: October 2025

