# 🎯 Agentic AI for PDE Discovery

An autonomous multi-agent system that discovers governing partial differential equations (PDEs) from spatiotemporal data using vision-language models and symbolic regression.

---

## 🚀 Quick Start

```python
# 1. Set your NVIDIA API key in the notebook (Cell 5)
CONFIG = {
    "api_key": "your-nvidia-api-key-here",
    "dataset_name": "KS",  # or "Burgers"
}

# 2. Run all cells in llm4ed_agentic_ai_CLEAN.ipynb

# 3. Find discovered PDE in FINAL_RESULT.txt
```

---

## 📖 What This Does

**Input**: Spatiotemporal data `u(x,t)` from PDEs (e.g., Kuramoto-Sivashinsky, Burgers equation)

**Output**: Discovered governing equation with fitted coefficients
```
Example: u_t = -1.023*u*u_x - 0.997*u_xx - 1.001*u_xxxx
```

**Method**: Multi-agent AI system with visual analysis → symbolic hypothesis → numerical fitting → visual validation

---

## 🤖 How It Works

### Multi-Agent Pipeline

```
VLM (Vision) → LLM (Symbolic) → Engineer (Code) → Executor (Run) → 
Scientist (QC) → VLM Validator (Visual) → Critic (Decision)
```

### Agent Roles

1. **VLM (Contour_Plot_Analyser)**: Analyzes spatiotemporal plots, identifies patterns (shocks, waves, dispersion)

2. **LLM (Hypothesis Generator)**: Maps visual cues to symbolic PDE terms, generates 10 candidate equations
   - Example: "dispersion" → u_xxxx, "nonlinearity" → u*u_x

3. **Engineer**: Writes Python code to fit each candidate PDE using PySINDy
   - Creates custom PDE libraries for each hypothesis
   - Generates 3-subplot comparison plots (ground truth vs predicted vs error)

4. **Executor**: Runs the generated code, saves comparison plots

5. **Scientist**: Quality control - checks if plots were generated successfully

6. **VLM Validator**: Visually analyzes the 10 comparison plots
   - Rates match quality: EXCELLENT / GOOD / ACCEPTABLE / POOR
   - Identifies top 3 best matches

7. **Critic**: Final decision maker
   - If EXCELLENT match found → TERMINATE and save result
   - If POOR matches → Send feedback to LLM for new hypotheses

---

## 📂 Key Files

### Input
- `Bayesian_PDE_Discovery_DATA/kuramoto_sivishinky.mat` - KS equation data
- `Bayesian_PDE_Discovery_DATA/burgers.mat` - Burgers equation data
- `surface_contour_plots/KS_*.png` - Visualization plots

### Notebook
- `llm4ed_agentic_ai_CLEAN.ipynb` - Main notebook (run this!)

### Output
- `FINAL_RESULT.txt` - **Discovered PDE with coefficients** ⭐
- `pde_comparison_0.png` to `pde_comparison_9.png` - Visual comparison plots
- `fitted_pdes.txt` - All fitted equations from PySINDy
- `u.npy`, `x.npy`, `t.npy` - Processed data

---

## ⚙️ Configuration

Edit `CONFIG` dictionary in Cell 5:

```python
CONFIG = {
    # Resolution (higher = more accurate but slower)
    "n_x_sub": 1024,      # Spatial points (256-1024)
    "n_t_sub": 512,       # Temporal points (128-512)
    
    # Search parameters
    "max_rounds": 30,     # Max iterations (20-50)
    
    # Dataset
    "dataset_name": "KS", # "KS" or "Burgers"
    
    # Term restrictions
    "restrict_terms": True,  # Remove polynomial terms for cleaner search
    
    # API
    "api_key": "nvapi-...",
    "llm_model": "qwen/qwen3-coder-480b-a35b-instruct",
    "vlm_model": "microsoft/phi-4-multimodal-instruct",
}
```



## 🔧 How PySINDy Fitting Works

For each candidate PDE from the LLM:

1. **Create Custom Library**: Parse PDE terms (e.g., `u*u_x`, `u_xx`, `u_xxxx`)
2. **Train-Test Split**: 60% train, 40% test
3. **Fit Coefficients**: Use `ps.STLSQ` sparse regression
4. **Predict**: Generate `u_dot_predicted` on test set
5. **Visualize**: 3-subplot comparison (ground truth | predicted | error)

**Key**: The LLM proposes symbolic structure, PySINDy finds optimal coefficients.

---

## 📊 Visual Validation Process

Instead of just using numerical L2 error, the system uses **visual analysis**:

1. Engineer generates comparison plots for each PDE
2. VLM Validator examines all 10 plots visually
3. Looks for:
   - Pattern similarity between ground truth and predicted
   - Error map uniformity (small random noise = good)
   - Physical plausibility

**Why visual?** Catches systematic errors that numerical metrics might miss.

---

## 🐛 Troubleshooting

### "No NVIDIA API key"
- Get key from: https://build.nvidia.com/
- Set in `CONFIG["api_key"]`

### "No plots generated"
- Check `surface_contour_plots/` folder exists
- Verify dataset name matches available .mat files

### "All PDEs rated POOR"
- Increase `n_x_sub` and `n_t_sub` for better signal
- Lower `restrict_terms` to allow more term types
- Check if dataset is noisy

### "Takes too long"
- Reduce resolution: `n_x_sub=256, n_t_sub=128`
- Reduce max_rounds: `max_rounds=20`

---

## 📝 Dependencies

```bash
pip install autogen-agentchat autogen-ext pysindy scipy numpy matplotlib pillow
```

- **AutoGen**: Multi-agent orchestration
- **PySINDy**: Sparse regression and PDE fitting
- **NVIDIA API**: LLM/VLM inference

---

## 🎓 Citation

If you use this code, please cite:

```
@software{pde_discovery_agentic,
  title = {Agentic AI for PDE Discovery},
  author = {Your Name},
  year = {2024},
  description = {Multi-agent system for discovering PDEs using VLMs and symbolic regression}
}
```

---



## 🔗 Related Work

- **PySINDy**: https://github.com/dynamicslab/pysindy
- **AutoGen**: https://github.com/microsoft/autogen
- **SINDy Paper**: Brunton et al. (2016) - Discovering governing equations from data

---


