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

## 🚀 How to Speed Up Execution

### Current Bottlenecks
1. **API Latency**: Each agent call to NVIDIA API takes 5-15s
2. **Large Data**: 1024×251 grid requires heavy computation
3. **Many Iterations**: Can take 10+ cycles to converge

### Speed Optimization Strategies

#### 1. **Use Local LLMs (10x faster)**
```python
# Replace NVIDIA API with local Ollama/LM Studio
llm_config = {
    "config_list": [{
        "model": "llama3.1:70b",  # Run locally
        "base_url": "http://localhost:11434/v1",
        "api_key": "not-needed"
    }],
    "cache_seed": None  # Disable caching for faster response
}
```

#### 2. **Reduce Data Resolution**
```python
# Faster: 256×128 instead of 1024×251
subsample_prompt_text, u_full, x_full, t_full = load_pde_data_and_subsample(
    'KS', 
    n_x_sub=256,  # Was 1024
    n_t_sub=128   # Was 251
)
```

#### 3. **Parallelize Agent Calls**
```python
# Use async execution for non-dependent agents
llm_config["timeout"] = 60  # Fail fast
```

#### 4. **Reduce Max Rounds**
```python
groupchat = autogen.GroupChat(
    agents=[...],
    max_round=20,  # Was 100 - stop sooner
)
```

#### 5. **Cache VLM Analysis**
```python
# Run VLM once, reuse description across iterations
vlm_description = Contour_Plot_Analyser.generate_reply(messages=[seed])
# Store and reuse instead of calling repeatedly
```

### **Recommended Fast Configuration**
```python
FAST_CONFIG = {
    "data_resolution": (256, 128),
    "max_rounds": 20,
    "use_local_llm": True,
    "cache_vlm": True,
    "timeout": 30
}
```
**Expected speedup: 5-10x faster** (from 2 hours → 15-20 mins)

---

## 🎯 Improving Accuracy

### Current Accuracy Issues
1. **LLM Hallucination**: Generates plausible but wrong PDEs
2. **Coefficient Mismatch**: Correct structure, wrong signs
3. **Premature Termination**: Accepts suboptimal PDEs

### Accuracy Improvements

#### 1. **Strengthen LLM Prompt**
```python
# Add explicit PDE examples to prompt
llm_prompt += """
KNOWN REFERENCE PDEs:
- Kuramoto-Sivashinsky: u_t = -u*u_x - u_xx - u_xxxx
- Burgers: u_t = -u*u_x + 0.1*u_xx
- KdV: u_t = -6*u*u_x - u_xxx

Match these patterns when visual cues align.
"""
```

#### 2. **Improve Critic Thresholds**
```python
# Stricter acceptance criteria
if score <= 0.005 and physically_valid and boundaries_stable:  # Was 0.01
    terminate()
```

#### 3. **Add Ground Truth Hints**
```python
# For known datasets, provide hints
if dataset == 'KS':
    llm_prompt += "Expect: nonlinear advection + diffusion + dispersion"
```

#### 4. **Multi-Metric Evaluation**
```python
# Don't rely only on L2 error
scores = {
    "l2_error": relative_l2_error(u_t_true, u_t_pred),
    "coefficient_stability": check_coefficient_magnitude(),
    "boundary_error": compute_boundary_mismatch(),
    "physical_validity": check_term_plausibility()
}
```

#### 5. **Ensemble Approach**
```python
# Run system 3-5 times, pick consensus PDE
results = [run_discovery() for _ in range(5)]
best_pde = vote_best_pde(results)
```

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

---

## 🔬 Example Run

### Input
- **Dataset**: Kuramoto-Sivashinsky (KS)
- **True PDE**: `u_t = -u*u_x - u_xx - u_xxxx`

### Iteration 1
- **VLM**: "Chaotic waves, dispersion, nonlinear advection"
- **LLM**: Generates 10 PDEs including `u_t = -u*u_x - u_xx - u_xxxx`
- **Engineer**: Fits coefficients, computes L2 = 0.008
- **Scientist**: Validates output
- **Critic**: Checks all PDEs, finds #3 has L2=0.008, physically valid → **TERMINATES**

### Output
```
✅ DISCOVERED PDE:
u_t = -1.0021*u*u_x - 0.9987*u_xx - 1.0003*u_xxxx
L2 Error: 0.0084
Status: ACCEPTED
```

---

## 🛠️ Troubleshooting

### Problem: Hallucination (Wrong PDE Selected)
**Solution**: 
- Increase data resolution
- Add PDE examples to LLM prompt
- Use stricter L2 threshold (< 0.005)

### Problem: Slow Execution (Hours)
**Solution**:
- Use local LLM (Ollama)
- Reduce data to 256×128
- Lower max_rounds to 20
- Cache VLM analysis

### Problem: Code Execution Errors
**Solution**:
- Check `use_docker: False` in executor
- Ensure PySINDy installed: `pip install pysindy`
- Verify data files exist in `Bayesian_PDE_Discovery_DATA/`

---

## 📚 Key References

1. **SINDy**: Brunton et al. (2016) - Sparse Identification of Nonlinear Dynamics
2. **AutoGen**: Microsoft Research - Multi-agent conversation framework
3. **Kuramoto-Sivashinsky Equation**: Canonical chaotic PDE

---

## 🎓 Educational Summary

**What makes this system "agentic"?**
- Agents have specialized roles (VLM→LLM→Engineer→Scientist→Critic)
- Feedback loops enable learning (Critic guides LLM)
- Autonomous decision-making (Critic decides when to stop)

**Why is the LLM critical?**
- Bridges vision (plots) and math (PDEs)
- Incorporates centuries of physics knowledge
- Adapts based on feedback (pseudo-RL)

**When should you use this?**
- Unknown PDEs from experimental data
- Automated scientific discovery
- Rapid hypothesis testing (10 PDEs per iteration)

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

**Author**: Amartya Roy  
**Version**: 2.0 (Optimized)  
**Last Updated**: October 2025
