# How Your Literature Review Maps to Implementation

This document shows exactly which findings from your literature review (§1-9) become which code modules and dissertation chapters.

---

## LIT REVIEW SECTION → CODE MODULE → DISSERTATION CHAPTER

### §2: The Problem Domain (Class Imbalance Mechanism)

**Lit review key findings:**
- Global fraud losses: $34B in 2023 (Baisholan et al. 2025)
- Fraud rates: 0.13% to 3.6% depending on dataset
- Why imbalance degrades classifiers: majority class bias in decision boundary
- Accuracy becomes misleading (99.83% trivial classifier scores ~99.8%)

**Code implementation:**
- `src/data_loader.py` § `validate_data()` → logs fraud rate, confirms 0.172%
- `src/evaluation.py` → NEVER uses accuracy as primary metric (uses AUC-PR)

**Dissertation writing:**
- **Chapter 1, §1.1-1.2:** Problem statement (why fraud detection matters, why imbalance is hard)
- **Chapter 3, §3.1:** Dataset description (show Figure 1: class distribution chart)
- **Chapter 2, §2.1:** Problem framing from lit review §2

---

### §3.1-3.3: Data-Level Resampling Methods

**Lit review key findings:**
- Four SMOTE variants independently identify same root problem (outliers corrupt interpolation)
- Modest, dataset-conditional gains between methods
- No single technique consistently wins (Brandt & Lanzén 2020, Gnip et al. 2021)
- Undersampling trades-off: compact training but variance increases, information loss
- Hybridization (SMOTE+Tomek) shows promise (Batista et al. 2004)

**Techniques you'll implement (Month 5-6):**
```python
# src/imbalance_methods.py (when you write it)

class ResamplingSuite:
    def random_undersampling(X_train, y_train):
        # Random Undersampling
    
    def random_oversampling(X_train, y_train):
        # Random Oversampling
    
    def smote(X_train, y_train):
        # SMOTE (Fernández et al. 2018)
    
    def smote_tomek(X_train, y_train):
        # SMOTE + Tomek (hybrid)
    
    def adasyn(X_train, y_train):
        # ADASYN (He et al. 2008) — NEED PDF
    
    def soa_s(X_train, y_train):
        # Selective Oversampling (Gnip et al. 2021)
```

**Dissertation writing:**
- **Chapter 2, §2.1-2.3:** Lit review on resampling taxonomy
- **Chapter 3, §3.2:** Experimental design — which techniques, how implemented
- **Chapter 5, §5.1-5.2:** Results tables comparing techniques

**Why this matters:** Lit review's conclusion ("no consistent winner") is your **justification for doing a rigorous comparison instead of proposing a novel method**. This is not a weakness — it's the whole point.

---

### §4.1-4.4: Algorithm-Level Approaches

**Lit review key findings:**
- Cost-sensitive learning: 97.7% of studies show improvement (but publication bias alert!)
- Focal loss: emerging as principal response to fixed-cost problem
- Ensembles are workhorses: Random Forest in 24/44 studies, XGBoost in 12/44
- Reinforcement learning: explores temporal structure, shows recall gains but slower training
- Ensemble conclusion: implicit argument that robustness + efficiency matter for deployment

**Techniques you'll implement:**

```python
# src/imbalance_methods.py (continued)

class CostSensitiveMethods:
    def class_weight_lr(X_train, y_train):
        # Logistic Regression with class_weight='balanced'
        # Lit ref: Araf et al. (2024)
    
    def class_weight_rf(X_train, y_train):
        # Random Forest with class_weight='balanced_subsample'
    
    def focal_loss_xgb(X_train, y_train):
        # XGBoost with focal loss (via imbalance-xgboost)
        # Lit ref: Wang et al. (2020), Liu et al. (2022)
    
    def focal_loss_lgb(X_train, y_train):
        # LightGBM with focal loss
        # Lit ref: Liu et al. (2022)

class EnsembleMethods:
    def balanced_rf(X_train, y_train):
        # Balanced Random Forest (resampling ensemble)
    
    def rusboost(X_train, y_train):
        # RUSBoost: Random Undersampling + AdaBoost
        # Lit ref: Seiffert et al. (2010) — NEED PDF
    
    def easy_ensemble(X_train, y_train):
        # EasyEnsemble: Multiple undersampled sets + ensemble
        # Lit ref: Liu et al. (2009) — NEED PDF
    
    def xgb_scale_pos_weight(X_train, y_train):
        # XGBoost with scale_pos_weight = n_majority / n_minority
    
    def lgb_is_unbalanced(X_train, y_train):
        # LightGBM with is_unbalanced=True
```

**Dissertation writing:**
- **Chapter 2, §2.4:** Cost-sensitive learning (cite Araf et al. 97.7% stat, discuss publication bias)
- **Chapter 2, §2.5:** Ensemble methods as pragmatic trade-off
- **Chapter 3, §3.3-3.5:** Which techniques implemented and why

---

### §5.1-5.3: Evaluation Methodology (CRITICAL)

**Lit review key findings:**
- AUC-PR: 11.4% adoption despite universal agreement it's appropriate under extreme skew
- Your adoption of AUC-PR places you in top 11% on evaluation rigor
- Accuracy reported in 88.6% of studies (misleading under imbalance)
- ROC-AUC in 47.7% (less suitable than AUC-PR for rare events)
- MCC in 20.5% (robust to imbalance, underused)
- Statistical significance testing rare: bootstrap CIs (Brandt & Lanzén), Friedman-Nemenyi (Peykani et al.)
- Validation design: stratified k-fold can induce covariate shift; DOB-SCV alternative exists
- **CRITICAL:** Resampling must happen INSIDE fold, after train/test split

**Code implementation:**

```python
# src/evaluation.py (already created)

class FraudEvaluator:
    def compute_metrics(self, threshold=0.5):
        # Returns: auc_pr, precision, recall, f1, auc_roc, mcc, specificity
        # PRIMARY METRIC: auc_pr (lit review §5.1)
    
    def bootstrap_confidence_interval(self, metric_func, n_iterations=1000):
        # 95% CI (lit review §5.2 — Brandt & Lanzén method)
    
    def plot_pr_curve(self):
        # Visualization (lit review emphasis on AUC-PR)
    
    def plot_roc_curve(self):
        # For comparison, not primary (lit review caution)

# src/data_loader.py (already created)

class FraudDataset:
    def get_cross_validation_splits(self, n_splits=5):
        # Stratified k-fold
        # RESAMPLING HAPPENS INSIDE THIS LOOP (lit review §5.3)
```

**Validation pattern (lit review §5.3):**
```python
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # CORRECT: Resampling happens here (inside fold)
    X_train_resampled, y_train_resampled = resample(X_train, y_train)
    
    # Train model, evaluate on X_val (unbalanced, represents real distribution)
```

**Dissertation writing:**
- **Chapter 3, §3.4:** Evaluation metrics (cite Baisholan et al. Table 2)
  ```markdown
  ## 3.4 Evaluation Metrics
  
  Following Baisholan et al. (2025), we adopt AUC-PR as the PRIMARY metric.
  Literature shows AUC-PR is appropriate for extreme imbalance, but only 11.4% 
  of studies use it (vs. 47.7% using ROC-AUC). We use AUC-PR to align with 
  evaluation theory and place our work in top 11% on this dimension.
  
  Secondary metrics: Precision, Recall, F1, MCC (robust), Specificity.
  
  Statistical significance: Friedman-Nemenyi test for comparing 10-12 techniques.
  Bootstrap 95% confidence intervals per Brandt & Lanzén (2020).
  ```

- **Chapter 3, §3.5:** Cross-validation design (cite Fernández et al. §5.3)
  ```markdown
  ## 3.5 Cross-Validation and Leakage Prevention
  
  Following Fernández et al. (2018), we use stratified k-fold cross-validation
  with resampling applied INSIDE each fold, after the train/val split. This
  prevents information leakage and ensures evaluation reflects real-world
  class distribution (imbalanced).
  ```

- **Chapter 5, §5.1-5.3:** Results (populate tables as Month 5-6 completes)
  - Table 1: Baseline models (auc_pr, precision, recall, f1)
  - Table 2: Technique comparison (mean ± std across 5 folds)
  - Figure 1: PR curves for top 3 techniques
  - Statistical significance: Friedman test p-value and technique rankings

---

### §6: Comparative Findings (No Clear Winner)

**Lit review key finding:**
"No resampling technique consistently wins. Technique choice interacts with classifier, 
dataset characteristics, imbalance degree and hyperparameters. Marginal differences 
between well-tuned methods frequently smaller than variance from interactions."

**What this means for your dissertation:**
- You're NOT trying to prove "Technique X is always best"
- You ARE showing "Here's how 12 techniques perform on credit card fraud with controlled conditions"
- Your novel contribution is systematic comparison + latency analysis, not a new algorithm

**Dissertation positioning:**
- **Chapter 1, §1.3:** Your contribution statement
  ```markdown
  ## 1.3 Dissertation Contributions
  
  Rather than proposing a novel imbalance-handling technique, this dissertation
  makes three contributions to the fraud detection literature:
  
  1. **Rigorous comparative evaluation** of 12 existing techniques under 
     controlled conditions (fixed pipeline, significance testing, primary 
     metric AUC-PR) — addressing lit review Gap 3 (§7).
  
  2. **First systematic latency characterization** comparing accuracy vs. 
     inference time across all 12 techniques — addressing lit review Gap 1 (§7).
  
  3. **Production implementation** with working deployed API, addressing 
     practical deployment constraints (lit review Gap 1).
  
  This aligns with Brandt & Lanzén (2020), Gnip et al. (2021), and Baisholan 
  et al. (2025), which conclude that rigorous evaluation of existing methods 
  is more valuable than incremental algorithmic novelty.
  ```

- **Chapter 6, §6.1:** Discussion — cite Brandt & Lanzén, Gnip et al. on why comparison is harder than it looks

---

### §7: Research Gaps (Your Novelty)

**Lit review identifies 6 gaps; you address 3:**

| Gap | Lit Review Finding | Your Implementation | Chapter |
|-----|-------------------|--------------------|---------| 
| **Gap 1** | Only 2/21 sources report latency; none measure latency of imbalance-handling techniques themselves | `src/inference_profiler.py` (Month 7-8) → profile inference time per model, create accuracy vs latency plot | Ch.5 §5.4 |
| **Gap 2** | AUC-PR in 11.4% of studies despite consensus; cost-measures absent | `src/evaluation.py` uses AUC-PR as primary + threshold-specific reporting | Ch.3 §3.4 |
| **Gap 3** | Cross-study comparison impossible; hyperparameters confound technique choice | Fixed random seed, identical CV, identical tuning budget, single dataset, single hardware | Ch.3 §3.2 |
| Gap 4 | Explainability discussed, not implemented | `notebooks/02_baseline.ipynb` → SHAP plots on final model (scope-limited) | Ch.5 §5.5 |
| Gap 5 | Concept drift acknowledged, untested | Temporal split as sensitivity check if time permits (future work) | Ch.7 §7.2 |
| Gap 6 | MLOps absent from peer-reviewed corpus | Basic MLflow tracking (no full ML pipeline) | Ch.4 §4.3 |

**Dissertation writing:**
- **Chapter 7, Conclusion:** Explicitly map your contributions to lit review gaps
  ```markdown
  ## 7.1 Addressing Literature Gaps
  
  This dissertation directly addresses three gaps identified in the systematic review:
  
  1. **Latency Analysis (Gap 1):** Chapter 5 §5.4 provides the first systematic 
     measurement of inference latency across imbalance-handling techniques. 
     Figure 8 shows accuracy-latency trade-off space.
  
  2. **Evaluation Rigor (Gap 2):** By adopting AUC-PR as primary metric, we align 
     with evaluation theory (Baisholan et al. 2025) and represent top 11% of 
     published work on this dimension.
  
  3. **Controlled Comparison (Gap 3):** Fixed pipeline, significance testing, 
     and single-dataset evaluation address the confounding in cross-study 
     comparisons noted by Gnip et al. (2021).
  ```

---

### §8: Corpus Issues to Resolve (Before Month 5)

**Your action (THIS WEEK):**

| Issue | Your Action | Impact | Timeline |
|-------|-------------|--------|----------|
| Missing ADASYN PDF | Download He et al. 2008 | Can't write Ch.3 §3.2 without it | THIS WEEK |
| Missing RUSBoost PDF | Download Seiffert et al. 2010 | Can't describe ensemble in Ch.3 | THIS WEEK |
| Missing EasyEnsemble PDF | Download Liu et al. 2009 | Can't describe ensemble in Ch.3 | THIS WEEK |
| Wrong SMOTE citation | Replace with Fernández et al. 2018 | Avoid factual error in bibliography | THIS WEEK |
| Textbook not read | Read He & Ma (2013) Ch.1-2 only (~2 hrs) | Grounds Ch.2 in foundations | Before Month 5 |

**Lit review §9 says:** "If you cite Chawla et al. (2002) from a secondary source, that's a factual error."

---

## Quick Reference: Lit Review → Chapter Mapping

Print this table. Keep it on your desk while writing.

| Dissertation Chapter | Writes From | Lit Review Sections | Key Citations |
|-----|---|---|---|
| Ch.1: Introduction | Problem framing, research question, contribution | §2, §7 | Baisholan et al., Ruchay et al. |
| Ch.2: Literature Review | Resampling taxonomy, cost-sensitive methods, ensembles, evaluation theory | §3, §4, §5, §6 | Fernández et al., Araf et al., Baisholan et al. |
| Ch.3: Methodology | Dataset, splits, features, models, evaluation metrics, CV design | §5.1-5.3, §9 | Baisholan et al., Brandt & Lanzén, Fernández et al. |
| Ch.4: System Architecture | API design, deployment rationale | §7 Gap 6 (grey lit) | AWS, Idris (Medium) |
| Ch.5: Results | Technique comparison, latency analysis, statistical tests | §6, §7 Gap 1 | Your experiments, Papanastassiou et al. |
| Ch.6: Discussion | Why comparison matters, publication bias, hybrid deployment | §6.1, §6.2, §4.4 | Brandt & Lanzén, Papanastassiou et al. |
| Ch.7: Conclusion | Summary, gaps addressed, future work | §7 | Your lit review |

---

## Monthly Cadence: How Lit Review Guides Your Timeline

**Month 3-4 (Apr-May):** "Establish the data & evaluation baseline" — lit review §5
- Implement: data_loader.py, evaluation.py
- Write: Ch.1 problem statement, Ch.3 methodology

**Month 5-6 (Jun-Jul):** "Run systematic comparison" — lit review §3, §4, §6
- Implement: imbalance_methods.py (12 techniques)
- Write: Ch.5 preliminary results, Ch.2 technique taxonomy

**Months 7-8 (Aug-Sep):** "Build production system" — lit review §7 Gap 1, Gap 6
- Implement: api.py (FastAPI), inference_profiler.py
- Write: Ch.4 system architecture

**Months 9-10 (Oct-Nov):** "Complete analysis" — lit review §5.2, §6.1
- Implement: Friedman-Nemenyi testing, latency plots
- Write: Ch.5 results complete, Ch.6 discussion

**Month 11 (Dec):** "Intensive writing sprint" — synthesis
- Write: All chapters polished, Ch.7 conclusion, abstract
- Cite: Everything back to lit review §1-7

---

## This Should Take ~15 Minutes to Read

Bookmark this. Reference it while coding. Every time you implement a function, ask:
- "Which lit review section justifies this?"
- "Which dissertation chapter discusses this?"
- "Which citation do I cite?"

If you can't answer all three, add a comment to your code and note it for writing later.

Good luck. Your lit review is excellent and comprehensive. Use it. 🚀
