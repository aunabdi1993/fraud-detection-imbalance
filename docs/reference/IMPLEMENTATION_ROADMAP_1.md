# Implementation & Writing Roadmap

**Goal:** Execute dissertation Month 3-4 (Data & Baseline) with parallel writing scaffolding.

**Constraint:** 6-7 hours/week available

**Current state:** Week 1 (setup phase)

---

## MONTH 3-4: Data & Baseline (Apr-May, 2026)

### WEEK 9-10: Data Setup & Validation (7-8 hours total)

**Code tasks (4-5 hours):**
- [ ] Download Kaggle ULB European Credit Card Fraud dataset
- [ ] Implement `FraudDataset` class (src/data_loader.py) with:
  - `load_data()`: Validate 284,807 transactions, 0.172% fraud rate
  - `validate_data()`: Check dataset matches benchmarks
  - `train_val_test_split()`: Stratified 70/15/15 split (lit review §5.3)
  - `normalize_features()`: StandardScaler fit on training only (prevent leakage)
  - `save_split()`: Persist arrays to disk
- [ ] Test that cross-validation splits maintain class distribution
- [ ] Log all outputs to MLflow (experimentation tracking)

**Writing tasks (2-3 hours):**
- [ ] Start `dissertation/chapters/ch3_methodology.md` **skeleton**:
  ```markdown
  # Chapter 3: Methodology
  
  ## 3.1 Dataset Description
  [Will write this week from code validation]
  - Source: ULB European Credit Card Fraud (Kaggle)
  - 284,807 transactions, 0.172% fraud rate
  - 30 PCA-transformed features (V1-V28), Time, Amount
  - Class imbalance: See Figure 1 (from EDA Week 11-12)
  
  ## 3.2 Train/Val/Test Split
  [From code: stratified 70/15/15]
  
  ## 3.3 Feature Normalization
  [From code: StandardScaler on training only]
  
  ## 3.4 Baseline Models [PLACEHOLDER for Week 15-16]
  
  ## 3.5 Evaluation Metrics [PLACEHOLDER for Week 16]
  ```
- [ ] Add reference to lit review §5.3 (Fernández et al. 2018): explain why stratified split matters
- [ ] Create `dissertation/tables/table_dataset.csv`:
  ```
  Metric,Value,Reference
  Transactions,284867,ULB Kaggle
  Fraud Rate,0.172%,Baisholan et al. (2025) Table 1
  Features,30 PCA,Original dataset
  Train Size,199465,70% stratified
  Val Size,42723,15% stratified
  Test Size,42623,15% stratified
  ```

**Deliverable:** Working data loader, Chapter 3 skeleton, dataset table for results chapter

---

### WEEK 11-12: Exploratory Data Analysis (6-7 hours)

**Code tasks (3-4 hours):**
- [ ] Create `notebooks/01_eda.ipynb` with:
  - Load split data
  - **Class distribution analysis** → Figure 1 (bar chart: 0.172% fraud)
  - **Feature distributions** (violin plots for top 10 most predictive V-features)
  - **Correlation matrix** → heatmap showing feature relationships
  - **Temporal patterns** (fraud rate over Time feature)
  - **Missing values** check (should be 0)
  - **Statistics table** (mean, std, min, max per class per feature)
- [ ] Save all figures to `dissertation/figures/`
- [ ] Generate `experiments/eda_summary.txt` with:
  ```
  Dataset shape: (199465, 30)
  Fraud cases: 345 (0.172%)
  Legitimate: 199120 (99.827%)
  
  Features: 30 PCA-transformed (V1-V28), Time, Amount
  Correlated features (>0.5): [list top pairs]
  
  Temporal patterns: [fraud rate evolution if visible]
  ```

**Writing tasks (2-3 hours):**
- [ ] Write `dissertation/chapters/ch3_methodology.md` §3.1 (Dataset Description) — ~400 words:
  - Use figures from EDA
  - Cite lit review: Ruchay et al. (2023) on why imbalance degrades classifiers
  - Cite Baisholan et al. (2025) on ULB dataset prevalence
  - Include table of train/val/test splits
  
- [ ] Create `dissertation/chapters/ch1_introduction.md` **preliminary section** on problem:
  ```markdown
  # Chapter 1: Introduction
  
  ## 1.1 Financial Fraud and Class Imbalance [WRITE THIS WEEK]
  - Global fraud losses: $34 billion in 2023 (Baisholan et al. 2025)
  - Fraud rate in real systems: 0.172% (ULB dataset) to 3.6% (proprietary)
  - Why imbalance matters: [cite lit review §2.2]
    - Classifier bias toward majority class
    - Accuracy becomes misleading (99.83% trivial classifier)
  - Your data: Figure 1 shows extreme imbalance
  
  ## 1.2 Research Questions [PLACEHOLDER]
  ## 1.3 Dissertation Contributions [PLACEHOLDER for Month 10]
  ```

**Deliverable:** 10-15 EDA figures, Chapter 1 problem statement drafted, Chapter 3 §3.1 complete

---

### WEEK 13-14: Feature Engineering (5-6 hours)

**Code tasks (3-4 hours):**
- [ ] Implement `src/preprocessing.py` with:
  - **RFM features** (if temporal data sufficient):
    - Recency: days since last transaction
    - Frequency: # transactions in last 30 days
    - Monetary: total amount in last 30 days
  - **Velocity features**:
    - Transaction rate (per hour, per day)
    - Amount rate (total amount per transaction)
  - **Aggregation features**:
    - Rolling window stats (mean, std of amount)
  - **Note:** Keep it simple — lit review §3 notes "proven features only"
- [ ] Store feature engineering pipeline in scikit-learn Pipeline
- [ ] Document feature creation in code comments with lit review references
- [ ] Generate `experiments/features_created.txt`:
  ```
  Raw features: 30 (from PCA-transformed V1-V28)
  Engineered features: [number]
  
  RFM features: [yes/no, if applicable]
  Velocity features: [list]
  
  Total features after engineering: [number]
  ```

**Writing tasks (1-2 hours):**
- [ ] Write `dissertation/chapters/ch3_methodology.md` §3.2 (Feature Engineering) — ~300 words:
  - What features were created and why
  - Cite lit review §3 on feature selection principles
  - Show example RFM calculation if used
  - Note: Keep this section SHORT — focus on simplicity, not novelty

**Deliverable:** Feature engineering pipeline, feature list documented

---

### WEEK 15-16: Baseline Models (7-8 hours)

**Code tasks (4-5 hours):**
- [ ] Implement `src/baseline_models.py` with three classifiers:
  1. **Logistic Regression** (sklearn)
     - Fit on training split
     - Get probability predictions
  2. **Decision Tree** (sklearn.tree.DecisionTreeClassifier)
     - No imbalance handling yet, just baseline
  3. **Random Forest** (sklearn.ensemble.RandomForestClassifier)
     - n_estimators=100 (standard)
     - Fixed random_state=42
  
- [ ] For each model:
  - Train on X_train, y_train
  - Evaluate on X_val, y_val (no leakage)
  - Get y_pred_proba for evaluation
  - Log metrics to MLflow: auc_pr, auc_roc, precision, recall, f1, mcc
  
- [ ] Implement `src/evaluation.py` (already created) with:
  - `compute_metrics()` at threshold 0.5
  - `plot_pr_curve()` and `plot_roc_curve()`
  - Save all curves to `dissertation/figures/`

- [ ] Create baseline results table:
  ```
  Model,AUC-PR,AUC-ROC,Precision,Recall,F1,MCC
  LogisticRegression,?,?,?,?,?,?
  DecisionTree,?,?,?,?,?,?
  RandomForest,?,?,?,?,?,?
  ```

- [ ] Test cross-validation setup:
  - Run 5-fold CV on training data (no resampling yet)
  - Verify folds maintain class distribution
  - Log fold-level metrics

**Writing tasks (2-3 hours):**
- [ ] Complete `dissertation/chapters/ch3_methodology.md`:
  - §3.3: Model Selection — explain choice of LR, DT, RF as baselines
  - §3.4: Experimental Design — cross-validation strategy (lit review §5.3)
  - §3.5: Evaluation Metrics (draft):
    ```markdown
    ## 3.5 Evaluation Metrics
    
    We adopt the evaluation framework from Baisholan et al. (2025) §5.1,
    which finds that field practice lags evaluation theory.
    
    **Primary metric: AUC-PR**
    - Reason: [lit review §5.1]
    - AUC-PR used in 11.4% of studies despite near-universal agreement it's appropriate
    - We use it to place our work in top 11% on evaluation rigor
    
    **Secondary metrics:**
    - Precision, Recall, F1 (standard for imbalanced classification)
    - AUC-ROC (for comparison with literature, though less suitable under extreme imbalance)
    - MCC (Matthews Correlation Coefficient — robust to imbalance per Brandt & Lanzén 2020)
    
    **Statistical significance:**
    - Friedman-Nemenyi test for comparing 10-12 techniques (Peykani et al. 2025)
    - Bootstrap 95% confidence intervals (Brandt & Lanzén 2020)
    
    **Threshold optimization:**
    - Not fixed at 0.5; optimized for F1 and business objectives
    - Operational constraint: latency (Chapter 5)
    ```

- [ ] Start `dissertation/chapters/ch5_results.md` **skeleton** with table placeholders:
  ```markdown
  # Chapter 5: Results & Evaluation
  
  ## 5.1 Baseline Model Performance
  [Table will be populated in Month 5-6]
  
  Table 1: Baseline Models (No Imbalance Handling)
  | Model | AUC-PR | Precision | Recall | F1 | MCC |
  
  ## 5.2 Class Imbalance Technique Comparison
  [To be completed]
  ```

**Deliverable:** Three trained baseline models, baseline results table, Chapter 3 complete, Chapter 5 skeleton created

---

## MONTH 5-6: Model Comparison (Jun-Jul)

This is where lit review findings (§6, §7) drive implementation.

### High-Level Overview (You'll implement this next):

**Techniques to compare** (from lit review §3-4):

1. **Resampling** (§3):
   - Random Undersampling
   - Random Oversampling
   - SMOTE (Fernández et al. 2018)
   - SMOTE+Tomek (Batista et al. 2004 via Fernández)
   - ADASYN (He et al. 2008 — NEED PDF)
   - SOA-S (Gnip et al. 2021)

2. **Ensemble/Cost-Sensitive** (§4):
   - Balanced Random Forest
   - RUSBoost (Seiffert et al. 2010 — NEED PDF)
   - EasyEnsemble (Liu et al. 2009 — NEED PDF)
   - XGBoost with scale_pos_weight
   - LightGBM with is_unbalanced
   - XGBoost with Focal Loss (via imbalance-xgboost)

3. **Base classifiers** (tested with each technique):
   - Logistic Regression
   - Random Forest
   - XGBoost

**Evaluation** (from lit review §5-6):
- 5-fold cross-validation
- Resampling INSIDE fold (after split) to prevent leakage
- Friedman-Nemenyi test for significance
- AUC-PR as primary metric
- Bootstrap CIs

---

## RECOMMENDED WEEKLY SCHEDULE (Ongoing)

Map to your 6-7 hours/week:

```
Monday evening (1.5 hrs):    Code: implement that week's algorithms
Tuesday evening (1.5 hrs):   Code: run experiments, log to MLflow
Wednesday evening (1.5 hrs): Code: analysis and visualization
Thursday evening (2 hrs):    Writing: convert code findings to prose
Friday morning (1 hr):       Review: weekly debrief, plan next week
```

**Weekly writing rule:**
Every time you implement a new technique, write 200-300 words about it in the corresponding chapter section. Don't batch writing until December.

---

## How to Use This Roadmap

1. **Copy-paste tasks** into your task manager (Todoist, NotionI, paper)
2. **Estimated hours** shown — adjust if you're faster/slower
3. **"Deliverable" at end** of each section is what gets you unstuck if you fall behind
4. **Every code module maps to a chapter** — see README.md table
5. **Each writing section is <500 words** — not the full chapter, just that week's piece

---

## Common Pitfalls (From Lit Review)

1. **"I'll write it all in December"** — Don't. You'll run out of time. Write 200-300 words every Thursday.
   - Reference: Revised plan, Month 11 is INTENSIVE for a reason

2. **"Just one more resampling variant"** — No. Stick to the 12 listed above.
   - Reference: Lit review §6.1 — "no technique consistently wins", marginal differences between well-tuned methods

3. **Forgetting resampling must happen INSIDE fold** — You'll get data leakage.
   - Reference: Lit review §5.3, Fernández et al. (2018)

4. **Using Accuracy as primary metric** — You'll mislead yourself. Use AUC-PR.
   - Reference: Baisholan et al. (2025) Table 2, lit review §5.1

5. **No significance testing** — Results uninterpretable (see Ruchay et al. 0.0001 accuracy gain).
   - Reference: Lit review §5.2, your evaluation.py already has Friedman-Nemenyi

---

## Success Metrics (At End of This Month 3-4 Phase)

- [ ] Dataset loaded, validated, split (70/15/15 stratified)
- [ ] Three baseline models trained and evaluated
- [ ] Baseline results table with AUC-PR, Precision, Recall, F1, MCC
- [ ] Chapters 1, 3 drafted (500 + 600 words)
- [ ] Chapter 5 skeleton created with table placeholders
- [ ] 10-15 EDA figures in `dissertation/figures/`
- [ ] All code commented with lit review references
- [ ] MLflow tracking all experiments

If you hit all these: You're on track. If you miss >2: Adjust scope per revised plan §WHAT YOU'RE CUTTING.

---

## Questions to Ask Your Supervisor (Before Month 5)

Email after Week 16:

```
Dear [Supervisor],

I've completed the data loading and baseline modeling phase 
(Weeks 9-16). Key findings:

- Dataset validated: 284,807 transactions, 0.172% fraud rate
- Baseline AUC-PR: [LR: X, DT: Y, RF: Z]
- Methodology: Stratified 5-fold CV, AUC-PR as primary metric, 
  Friedman-Nemenyi for significance

Before I proceed to comparing imbalance-handling techniques (Month 5-6), 
I want to confirm:

1. Is the baseline approach sound (stratified CV, AUC-PR primary)?
2. Should I proceed with the 12 techniques listed in my plan, or would 
   you recommend adjustment?
3. Any other methodological feedback before I commit to 6 more weeks 
   of experimentation?

Best regards,
[Your name]
```

This gives your supervisor visibility and you course-correcting feedback *before* you've spent weeks on work that might need changing.

---

## Resources

- **Lit Review:** `/mnt/project/literature_review.pdf` (maps to every decision)
- **Dissertation Plan:** `/mnt/project/revised_dissertation_plan.pdf` (timeline + milestones)
- **Code Templates:** `src/data_loader.py`, `src/evaluation.py` (already created)
- **Dataset:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Key Papers to Have Accessible:**
  - Fernández et al. (2018) — evaluation and resampling methodology
  - Baisholan et al. (2025) — metric selection justification
  - Peykani et al. (2025) — Friedman-Nemenyi test
  - Ruchay et al. (2023) — baseline on ULB dataset (for comparison)

---

**Last reminder:** Done is better than perfect. Get Week 15-16 baseline working, write 200 words, move on to Month 5. You've got this. 🚀
