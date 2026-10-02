# Literature Review: Class Imbalance Handling for Financial Fraud Detection

**Research question:** *How do different class imbalance handling techniques compare in performance and practical deployability for financial fraud detection?*

**Corpus:** 21 substantive sources from the Pre-reading folder (18 peer-reviewed, 3 grey literature), plus two textbooks and one survey.

> **How to use this document.** This is a *working* review at roughly 5,000 words — longer than the 2,000-word Chapter 2 in your plan. It is written so you can cut it down: §2–§6 are the chapter, §7 (gaps) is the bridge into your Introduction's contribution claim, and §8–§9 are notes for you rather than for the examiner.

---

## 1. Scope and corpus

The review covers 21 sources spanning 2018–2026, organised around four themes: (i) the fraud detection problem and the nature of class imbalance; (ii) data-level resampling; (iii) algorithm-level cost-sensitive and ensemble methods; (iv) evaluation practice and deployment.

Three sources are grey literature — Ravaglia (2022) and Idris (2025) on Medium, and the AWS solution repository — and one is a vendor engineering blog (AWS, 2025). These are used only for practitioner framing and architectural reference, never as evidence for a performance claim. This distinction is made explicit because an examiner will otherwise ask.

Coverage is uneven, and the review says so where it matters. Of the 44 studies synthesised in the field's most recent systematic review, 32 (72.7%) use the same ULB European Credit Card dataset that this dissertation uses (Baisholan et al., 2025). The corpus therefore reflects a literature that is deep on one benchmark and thin on everything else.

---

## 2. The problem domain: fraud detection and why imbalance is the central obstacle

### 2.1 Scale and trajectory

Global fraud losses exceeded $34 billion in 2023, the highest in seven years, and the threat profile is shifting rather than merely growing: digital document forgeries rose 244% year-on-year and deepfakes now account for 40% of biometric fraud incidents (Baisholan et al., 2025). Ruchay et al. (2023) note that manual review of suspicious activity produces unreliable results and consumes disproportionate time and resources, which is the standing economic argument for automation.

The methodological arc across the corpus is consistent: rule-based systems, then statistical classifiers, then supervised ML, then deep and sequential architectures. Darwish et al. (2025) characterise rule-based systems as interpretable and regulator-friendly but requiring continuous manual updating as fraud tactics evolve — a maintenance cost that motivated the shift to learned models. Cheah et al. (2023) make the same point from the data side: the evolving behaviour of fraudsters and growing dataset sizes made manual rule identification impractical.

### 2.2 Why imbalance degrades classifiers

Fraud rates in the corpus range from 0.13% (PaySim) through 0.172% (ULB European) to 3.6% (Papanastassiou et al.'s proprietary banking data) and 3.5% (IEEE-CIS). Ruchay et al. (2023) give the mechanism plainly: most classification algorithms assume a balanced label distribution, so a skewed prior biases the decision boundary toward the majority class.

The consequence is that accuracy becomes actively misleading. Ruchay et al. (2023) report accuracies of 0.9979–0.9999 across eleven algorithms on the ULB dataset — a spread that conveys almost nothing, since a trivial always-legitimate classifier scores 99.83%. Gnip et al. (2021) state the same problem generally: a model that classifies majority samples correctly and minority samples incorrectly will still show high accuracy.

Critically, skew alone is not the whole story. Fernández et al. (2018) argue that imbalance degrades performance in conjunction with *data intrinsic characteristics* — if two classes are severely imbalanced but cleanly separable, classification remains easy. The genuine difficulty arises from **small disjuncts** (minority concepts fragmented into rare sub-clusters), **class overlap**, **noise**, **lack of data**, and **dataset shift**. This framing matters for your dissertation because it predicts that resampling method rankings will be *dataset-dependent* rather than universal — which is exactly what the comparative evidence shows (§6.1).

Kennedy et al. (2024) add a compounding constraint specific to fraud: labels are expensive. Accurate fraud labels require corroborative manual investigation by financial experts, which is costly and slow, and some fraud types are not evident even after that analysis. Label noise is therefore a realistic assumption, not a corner case.

---

## 3. Data-level approaches: resampling

### 3.1 The SMOTE lineage

SMOTE interpolates between a minority instance and its *k* nearest minority neighbours to generate synthetic examples, rather than duplicating existing ones. Fernández et al. (2018) — writing at the technique's fifteen-year mark — explain the original motivation: random oversampling only increases the effective weight of minority instances and leads to overfitting, so synthetic generation was introduced to supply genuinely new information and improve generalisation. SMOTE is now the de facto benchmark in imbalanced learning and one of the most influential preprocessing algorithms in data mining.

The corpus contains four distinct SMOTE-improvement strategies, all targeting different failure modes:

| Variant | Failure mode addressed | Reported result |
|---|---|---|
| **ADASYN** (He et al., 2008) | Uniform generation ignores instance difficulty | Generates more samples for hard-to-learn instances |
| **ISMOTE** (Li et al., 2025) | Interpolation distorts local density; synthetic points cluster too tightly | +13.07% F1, +16.55% G-mean, +7.94% AUC (relative), 13 datasets, 3 classifiers |
| **ExtSMOTE family** (Matharaarachchi et al., 2024) | Outliers *within* the minority class corrupt interpolation | Dirichlet ExtSMOTE best on F1, MCC and PR-AUC vs. SMOTE and competitive variants |
| **SOA-S / SOA-A** (Gnip et al., 2021) | Same — isolates representative minority samples via outlier detection *before* oversampling | Outperformed SMOTE and ADASYN in the majority of cases across 8 datasets |

Two observations. First, three of the four independently identify the *same* root problem — SMOTE's interpolation is corrupted by minority-class outliers and by uneven local density. That convergence is a genuine finding worth stating in your review. Second, the reported gains are consistently modest and dataset-conditional. Gnip et al. (2021) found SOA's advantage grew with the imbalance ratio, and Li et al. (2025) explicitly caveat that their test datasets were at "medium-low" imbalance levels — which is a serious external-validity limitation for a 0.172% fraud problem.

### 3.2 Undersampling and hybrids

Fernández et al. (2018) summarise undersampling's trade-off: it produces a compact balanced training set and reduces learning cost, but increases classifier variance, produces warped posterior probabilities, and may discard informative examples. At high imbalance ratios so many majority examples must be removed that a lack-of-data problem is induced.

Ruchay et al. (2023) advocate Tomek links as a *cleaning* step followed by random undersampling, on the reasoning that removing boundary noise from the majority class before sampling reduces information loss. Their empirical support is weak, though: Tomek links removed 248 records and improved Random Forest accuracy by 0.0001. They present this as justification for using the technique throughout. **This is a good example to cite when arguing for statistical significance testing** — an improvement of 1e-4 in accuracy on a 0.172% fraud dataset is indistinguishable from noise, and no significance test is reported.

Batista et al. (2004), reported via Fernández et al. (2018), found hybridisations of SMOTE with undersampling to outperform other resampling techniques — the strongest available support for including SMOTE+Tomek and SMOTE+ENN in your comparison.

### 3.3 Generative and optimisation-based generation

Cheah et al. (2023) evaluate SMOTE, GAN, and two hybrids (SMOTE+GAN, GANified-SMOTE) with FNN, CNN and FNN+CNN classifiers. GANs alone are ill-suited to imbalanced tabular data since they were designed for image generation from random noise; feeding SMOTE-generated samples into the GAN instead of random noise (following Sharma et al.'s SMOTifed-GAN) improves on both. Their headline finding, however, is a methodological one and directly relevant to your design: **classifier hyperparameters affected classification performance regardless of which data generation technique was applied.** They also concede the study may not simulate real-world conditions with evolving fraud behaviour.

Darwish et al. (2025) use Artificial Bee Colony optimisation to guide synthetic fraud generation, claiming a 10% accuracy improvement over state-of-the-art. Treat this claim cautiously — accuracy is the wrong headline metric here, and the comparison baselines are not fully specified.

Kennedy et al. (2024) address the adjacent problem of *unlabelled* data, using autoencoder reconstruction error to synthesise class labels (labelling the top-500 highest-error instances as fraud). All six supervised classifiers trained on synthesised labels outperformed Isolation Forest on AUPRC. This is out of scope for your dissertation but is a good citation for the "label quality" limitation in your Discussion.

---

## 4. Algorithm-level approaches

### 4.1 Cost-sensitive learning

Cost-sensitive learning (CSL) penalises minority-class misclassification more heavily rather than altering the data. Araf et al. (2024) provide the most comprehensive treatment — 173 papers from 2010–2022 — and identify CSL's structural advantages: it does not alter the data distribution, so it preserves dataset integrity and permits full use of available data; it is computationally efficient relative to resampling; and it explicitly encodes asymmetric misclassification cost, which is precisely the structure of the fraud problem.

Two weaknesses recur. **Misclassification costs are unknown.** Designing a cost matrix requires domain expertise that is frequently unavailable, a limitation echoed across nine prior reviews. The pragmatic workaround — set majority cost to 1 and minority cost to the imbalance ratio — is widely adopted but arbitrary. **Overfitting the minority class.** Inadequately defined, heavily minority-weighted costs cause excessive adaptation and reduced generalisation.

Araf et al. (2024) report that 169 of 173 selected studies (97.7%) found CSL improved on cost-insensitive alternatives. **Do not report this figure uncritically.** A 97.7% positive-result rate across a body of literature is a textbook publication-bias signature; the honest reading is that CSL is *reliably reported* to help, not that it reliably helps. This is a strong point for your Discussion chapter.

Mienye & Sun (2021) modified the objective functions of logistic regression, decision tree, XGBoost and random forest, using inverse class distribution as the weighting heuristic and three repeats of 10-fold cross-validation. Cost-sensitive versions outperformed standard versions on precision, recall, F-measure and AUC — with the important caveat that *accuracy decreased*, which they correctly attribute to additional majority-class misclassifications. They recommend future work combining CSL with resampling and comparing against each used individually. **That combination is a legitimate slot in your experimental matrix.**

### 4.2 Focal loss

Focal loss down-weights easy examples so training concentrates on hard ones, adapting cost dynamically rather than fixing it — Araf et al. (2024) frame this as the principal response to the fixed-cost criticism.

Wang et al. (2020) implement weighted cross-entropy and focal loss for XGBoost as the `imbalance-xgboost` package, deriving the first- and second-order gradients required. This is the practical route to focal-loss XGBoost in your implementation, and it is scikit-learn compatible. Liu et al. (2022) embed focal loss in LightGBM for credit scoring and — unusually for this corpus — pair it with interpretability, using feature importance for global explanation ("what drives predictions") and partial dependence plots for local explanation ("how features affect predictions"). Albalawi & Dardouri (2025) apply focal loss in a deep model for credit card fraud specifically, obtaining the highest precision of the models compared.

Peykani et al. (2025) apply CorrOV-CSEn across MLP, random forest, gradient boosted trees, XGBoost, CatBoost and AdaBoost for business failure prediction, comparing via the Friedman–Nemenyi test. CatBoost achieved sensitivity of 0.909 — **but all models exhibited relatively low precision.** This precision collapse under aggressive cost-sensitivity is a finding your latency-and-deployability framing should engage with directly: a model that catches 91% of fraud while flooding investigators with false positives is not deployable regardless of its recall.

### 4.3 Ensembles

Ensembles are the workhorse of the applied literature. Baisholan et al. (2025) found Random Forest in 24 of 44 studies, decision trees in 13, XGBoost in 12, LightGBM in 10, with bagging/stacking/voting meta-ensembles in 10. They interpret this concentration as a pragmatic trade-off between recall-oriented performance under imbalance and retaining computational efficiency and interpretability for deployment — which is, in effect, the field implicitly making your dissertation's argument without measuring it.

Suguna et al. (2025) compare nine individual classifiers against six homogeneous ensembles on churn data. Performance rose from 61% to 79% after SMOTE, with AdaBoost achieving F1 of 87.6%. Two methodological notes: they explicitly state that balancing was applied **only to the training set**, which is correct practice and unfortunately not universal in this corpus; and their headline improvement is reported in accuracy terms, which is the weaker choice.

**Gap in your corpus:** the foundational RUSBoost (Seiffert et al., 2010) and EasyEnsemble (Liu et al., 2009) papers are on your reading list but the PDFs are absent. Since your Chapter 5 plans to evaluate both, you need these before writing Chapter 3.

### 4.4 Sequential and reinforcement approaches

Papanastassiou et al. (2026) offer the corpus's sharpest challenge to the standard framing. Tree ensembles treat transactions as i.i.d. tabular points, which renders them blind to the temporal and sequential structure of sophisticated fraud — fraud often unfolds as a trajectory (a low-risk phase, then escalation) rather than a single anomalous point. Their RLFD framework uses a DQN with LSTM encoders and asymmetric rewards over client-centric chronological windows.

Their results are instructive beyond the method itself:

- GBT achieved fraud recall of 0.226; RLFD achieved 0.549 on the same proprietary banking data.
- GBT achieved *superior* ROC-AUC (0.886 vs 0.773) while failing the business objective. They conclude that reliance on global metrics like ROC-AUC and accuracy under severe imbalance is dangerous, and that future comparisons should prioritise threshold-dependent metrics.
- **RLFD required ~2× GBT training wall-clock time, but inference latency remained comparable**, since the trained Q-network processes sequence windows in constant time. This is the corpus's clearest articulation of the training-cost/inference-cost distinction your latency analysis depends on.
- The advantage disappeared on a static tabular benchmark (UCI): RLFD accuracy 0.802 vs GBT 0.821. They conclude RLFD adds value only where rich sequential signals exist.
- They argue against winner-takes-all model selection, proposing hybrid parallel deployment — GBT as a high-precision primary filter with an RL agent intercepting sequential attacks that bypass static rules.

The last point is worth borrowing for your Discussion: the corpus's most sophisticated study concludes that the right answer is *architectural*, not a single best model.

---

## 5. Evaluation methodology

### 5.1 The metrics debate is settled in principle and unsettled in practice

Every source agrees accuracy is inadequate under extreme skew. Precision measures reliability of positive predictions; recall measures coverage of actual fraud; F1 is their harmonic mean. Brandt & Lanzén (2020) add MCC, a Pearson correlation between actual and predicted values in the confusion matrix, robust to imbalance.

The consequential finding is Baisholan et al.'s (2025) measurement of what the field *actually reports* across 44 studies:

| Metric | Studies reporting | % |
|---|---|---|
| Precision & Recall | 44 | 100% (inclusion criterion) |
| F1-score | 41 | 93.2% |
| Accuracy | 39 | 88.6% |
| AUC-ROC | 21 | 47.7% |
| MCC | 9 | 20.5% |
| **AUC-PR / Average Precision** | **5** | **11.4%** |
| Specificity | 4 | 9.1% |

AUC-PR — the metric the same authors argue *should* be primary under extreme skew — appears in one study in nine, while AUC-ROC persists as the default "mainly for historical and tooling reasons." Their recommendation is direct: evaluation must move beyond over-reliance on AUC-ROC; AUC-PR should be the primary metric, complemented by threshold-specific reporting, with cost- or profit-sensitive measures becoming mandatory.

**This single table justifies a substantial portion of your methodology chapter.** Adopting AUC-PR as primary places you in the top 11% of the field on evaluation rigour, at essentially zero cost.

### 5.2 Statistical testing is rare

Most studies in the corpus report point estimates without significance testing. The exceptions are worth modelling:

- **Albalawi & Dardouri (2025):** pairwise two-tailed *t*-tests across 10 independent runs, p < 0.05, on accuracy, precision, recall, F1 and ROC-AUC.
- **Mienye & Sun (2021):** three repeats of 10-fold cross-validation.
- **Peykani et al. (2025):** Friedman–Nemenyi test for multiple-model comparison.
- **Brandt & Lanzén (2020):** bootstrapped predicted values into replicate sets to obtain standard deviations and 95% confidence intervals.
- **Ruchay et al. (2023):** 10-fold cross-validation, but no significance testing — hence the uninterpretable 0.0001 accuracy gain.

For comparing 10–12 techniques, the Friedman test with a Nemenyi post-hoc is more defensible than repeated pairwise *t*-tests, which inflate family-wise error. Brandt & Lanzén's bootstrap approach to confidence intervals is a straightforward addition worth adopting.

### 5.3 Validation design and leakage

Fernández et al. (2018) identify a subtle and widely ignored problem: **standard stratified *k*-fold cross-validation can itself induce covariate shift**, because random shuffling may leave folds with unrepresentative regional coverage. They point to DOB-SCV (assigning nearby examples to *different* folds) as a partitioning strategy shown to be a stable performance estimator under imbalance. In severely imbalanced domains a single misclassified minority example can cause a significant performance drop, so this is not a marginal concern.

They also distinguish three shift types — prior probability shift (addressable by stratification), covariate shift (a partitioning problem), and concept shift / drift (the hardest). Fraud is a concept-drift domain by definition, since adversaries adapt.

The corpus's transparency on resampling-before-split varies. Suguna et al. (2025) state explicitly that balancing was applied only to training data. Albalawi & Dardouri (2025) report a balanced training set of 226,602 per class, which is consistent with correct practice, though they do not state it directly. Ruchay et al. (2023) do not specify. **Explicitly documenting that resampling occurs inside the cross-validation fold, after the split, is a cheap and visible methodological win.**

---

## 6. Comparative findings: what the evidence actually supports

### 6.1 No resampling technique consistently wins

This is the most important synthesised finding for your dissertation, and it is well-supported:

- **Brandt & Lanzén (2020)**, comparing SMOTE and ADASYN across three datasets and three classifiers: both improve performance in most cases, but *neither consistently outperforms the other* as imbalance varies. SVM with SMOTE beat SVM with ADASYN as imbalance increased; the highest F-measure came from Random Forest with ADASYN; both *worsened* logistic regression. On the Credit Card Fraud dataset specifically, both improved sensitivity, but F-measure improved only for Random Forest with ADASYN.
- **Gnip et al. (2021):** the choice of *classifier* mattered more than the choice of oversampling method — differences between classifiers exceeded ten percentage points on some datasets, for both oversampled and non-oversampled data. Their explicit recommendation is to pay attention to classifier choice for oversampled data.
- **Cheah et al. (2023):** hyperparameter settings affected performance regardless of generation technique.
- **López et al. (2012)**, via Fernández et al. (2018): preprocessing and cost-sensitive learning are "good and equivalent approaches" to the imbalance problem.
- **Ruchay et al. (2023):** some researchers argue for sampling superiority, others for cost-sensitive and ensemble methods; conclusions are inconclusive.

**The synthesis:** technique choice interacts with classifier, dataset characteristics, imbalance degree and hyperparameters, and marginal differences between well-tuned methods are frequently smaller than the variance introduced by those interactions. This directly justifies your framing — a rigorous, controlled comparison with significance testing is more valuable than another novel variant, precisely *because* the field cannot currently say which existing method to use when.

### 6.2 The recall–precision trade-off is the real design decision

Ravaglia (2022) articulates the practitioner framing: a model at 99% precision and 15% recall is not a good predictor because it misses 85% of fraud — but the converse is equally unusable operationally. Peykani et al. (2025) demonstrate the failure empirically (sensitivity 0.909, low precision across all models). Ruchay et al. (2023) explain why the asymmetry is real but bounded: false positives cost customer friction (freeze the account, contact the owner, unfreeze), whereas false negatives are direct financial liability — so the costs are asymmetric but *not unboundedly* so.

Papanastassiou et al. (2026) formalise this as translating the business's asymmetric cost matrix directly into the optimisation objective. **Threshold optimisation is therefore not a tuning detail but the primary deployment decision**, and should be treated as such in your Chapter 5.

---

## 7. Research gaps

These are ordered by how directly they support your contribution claim. Gaps 1–3 are yours to address; 4–6 belong in Future Work.

### Gap 1 — Deployment constraints are named but almost never measured

This is the strongest gap in the corpus, and I verified it rather than asserting it. Across the 21 substantive sources:

- **Only two report any quantitative latency measurement.** Albalawi & Dardouri (2025) report per-sample inference times (LR and DT <1ms; RF and XGBoost 3–10ms; deep model ~25ms, GPU-dependent), and Darwish et al. (2025) report 50ms average latency at 495 transactions/second throughput.
- **No source in the corpus measures the latency implications of the imbalance-handling techniques themselves** — how resampled training-set size affects training cost, whether cost-sensitive variants change inference cost, whether ensemble methods like RUSBoost or EasyEnsemble meet a real-time budget.
- Papanastassiou et al. (2026) come closest by separating training cost (~2× GBT) from inference cost (comparable), but only for one method pair.
- The grey literature that *claims* production focus does not measure it. Idris (2025) describes returning results "with minimal latency" without a single figure. The AWS reference architecture specifies a REST API and cost breakdown but no latency benchmarks.

**Your contribution:** a systematic latency-versus-detection-performance characterisation across 10–12 imbalance-handling techniques on a common dataset and hardware, with an explicit deployment threshold, does not exist in this literature. This is a defensible novel contribution that requires no novel algorithm.

### Gap 2 — Evaluation practice lags evaluation theory

AUC-PR appears in 11.4% of studies despite near-universal agreement that it is the appropriate primary metric under extreme skew (Baisholan et al., 2025). Cost- and profit-sensitive measures are almost entirely absent, though the same review argues they should be mandatory. Significance testing is the exception rather than the norm, and where absent produces uninterpretable results (Ruchay et al.'s 0.0001).

**Your contribution:** adopting AUC-PR as primary, reporting threshold-specific operating points, and applying Friedman–Nemenyi across all techniques is straightforwardly above the field's median practice.

### Gap 3 — No controlled comparison holds the pipeline constant

Cross-study comparison is nearly impossible: different datasets, splits, hyperparameter budgets, seeds and metrics. Gnip et al. (2021) and Cheah et al. (2023) independently show that classifier choice and hyperparameters can dominate the resampling effect — which means most reported technique rankings are confounded. Baisholan et al. (2025) excluded 619 of 663 full-text records partly for insufficient evaluation reporting and dataset opacity, which indicates how widespread this is.

**Your contribution:** fixed seeds, identical splits, identical tuning budget per technique, single dataset, single hardware configuration. The comparison is only interpretable if the pipeline is held constant, and most published comparisons do not do this.

### Gap 4 — Explainability is discussed and not implemented

Only **two of 44** studies operationalised interpretability, both using SHAP (Baisholan et al., 2025). Adoption is constrained by three barriers they name: semantic loss from anonymised feature spaces (a direct problem for the ULB dataset's PCA-transformed V1–V28), absence of analyst-friendly explanation interfaces, and instability of explanations under drift. Liu et al. (2022) are the corpus's positive example, using feature importance plus PDP. Ravaglia (2022) states the operational stake: a bank employee reviewing an alert needs the reason, and a model that cannot explain itself cannot be used for legal purposes.

Given your 6–7 hours/week, SHAP on the final model only is a realistic scope; a full explainability comparison is not. Note the anonymised-features limitation honestly.

### Gap 5 — Concept drift is acknowledged and untested

Fraud is adversarial and non-stationary. Fernández et al. (2018) identify concept shift as the hardest form of dataset shift; Papanastassiou et al. (2026) motivate their entire framework on adaptive fraudster behaviour; Cheah et al. (2023) concede their findings may not simulate evolving fraud. Yet **no study in the corpus evaluates degradation over time.** The ULB dataset contains a Time feature that would permit temporal rather than random splitting. This is worth one paragraph in Limitations and a line in Future Work; a temporal-split sensitivity check would be a cheap bonus if time permits.

### Gap 6 — Reproducibility and MLOps are absent from the peer-reviewed corpus

Everything in this corpus about deployment infrastructure — CI/CD, monitoring, model registries, drift detection — comes from grey or vendor literature. AWS (2025) reports Radial achieving a >75% reduction in deployment cycle and 9% model performance improvement, but this is a vendor case study without independent verification. There is no peer-reviewed MLOps evidence base for fraud detection here. Your plan correctly descopes full MLOps; cite this gap to justify that decision rather than leaving it unexplained.

---

## 8. How this maps to your dissertation

| Dissertation element | Supporting citations |
|---|---|
| **Ch.1** — problem scale, motivation | Baisholan et al. (2025); Ruchay et al. (2023) |
| **Ch.1** — contribution claim | Gaps 1–3 above |
| **Ch.2** — imbalance mechanism, data difficulty factors | Fernández et al. (2018); Gnip et al. (2021) |
| **Ch.2** — resampling taxonomy | Fernández et al. (2018); Li et al. (2025); Matharaarachchi et al. (2024); Brandt & Lanzén (2020) |
| **Ch.2** — cost-sensitive taxonomy | Araf et al. (2024); Mienye & Sun (2021); Wang et al. (2020); Liu et al. (2022) |
| **Ch.3** — metric selection (AUC-PR primary) | Baisholan et al. (2025) §3.2.3 |
| **Ch.3** — CV design, leakage avoidance, DOB-SCV note | Fernández et al. (2018) §5.3; Suguna et al. (2025) |
| **Ch.3** — significance testing protocol | Peykani et al. (2025); Brandt & Lanzén (2020); Albalawi & Dardouri (2025) |
| **Ch.4** — API architecture | Idris (2025); AWS (2025) — *grey lit, architectural reference only* |
| **Ch.5** — latency baselines to compare against | Albalawi & Dardouri (2025) Table 4; Darwish et al. (2025) Table 9 |
| **Ch.5** — training vs. inference cost separation | Papanastassiou et al. (2026) |
| **Ch.6** — "no consistent winner" | Brandt & Lanzén (2020); Gnip et al. (2021); López et al. (2012) |
| **Ch.6** — publication bias caveat | Araf et al. (2024) 97.7% figure |
| **Ch.6** — hybrid deployment argument | Papanastassiou et al. (2026) |
| **Ch.7** — future work | Gaps 4–6 |

---

## 9. Corpus issues to resolve

Five things need fixing before you write Chapter 2. None is serious, but each will cost you marks if it reaches the examiner.

1. **The file named "9 - Smote Synthetic Minority Over-sampling Technique.pdf" is not the SMOTE paper.** It is Fernández, García, Herrera & Chawla (2018), *SMOTE for Learning from Imbalanced Data: Progress and Challenges, Marking the 15-year Anniversary*, JAIR 61, 863–905. It is an excellent source and I have used it heavily, but if you cite it as Chawla et al. (2002) that is a factual error. **Obtain the original**: Chawla, N.V., Bowyer, K.W., Hall, L.O. & Kegelmeyer, W.P. (2002), *SMOTE: Synthetic Minority Over-sampling Technique*, JAIR 16, 321–357.

2. **Four papers on your reading list have no PDF**: ADASYN (He et al., 2008 — #10), RUSBoost (Seiffert et al., 2010 — #20), EasyEnsemble (Liu et al., 2009 — #21), and the original SMOTE (#9). Your Chapter 5 plans to evaluate ADASYN, RUSBoost and EasyEnsemble. You cannot describe a method in Chapter 3 from a secondary source.

3. **Two reading-list entries have wrong publication details.** #14 is Brandt & Lanzén, *Uppsala* University, Department of Statistics (not Linnaeus). #16 is *Expert Systems with Applications* 208, 118158 (not Engineering Applications of AI), by Liu, Fan, Xia & Xia.

4. **Three sources are non-peer-reviewed** (Ravaglia 2022, Idris 2025, AWS repo) and one is vendor marketing (AWS blog 2025). Use them for practitioner framing and architecture only. If a claim in your Results or Discussion rests on a Medium post, an examiner will find it.

5. **Two textbooks are unread** — Fernández et al. (2018), *Learning from Imbalanced Data Sets*, and He & Ma (2013), *Imbalanced Learning: Foundations, Algorithms and Applications*. You do not need to read either cover to cover. Read He & Ma Ch. 1–2 for the formal problem statement and evaluation-metric derivations; that is roughly two hours and will make Chapter 2 read as though it is grounded in the foundations rather than in recent papers alone.

---

## References

Albalawi, T. & Dardouri, S. (2025) 'Enhancing credit card fraud detection using traditional and deep learning models with class imbalance mitigation', *Frontiers in Artificial Intelligence*, 8, 1643292. doi:10.3389/frai.2025.1643292

Araf, I., Idri, A. & Chairi, I. (2024) 'Cost-sensitive learning for imbalanced medical data: a review', *Artificial Intelligence Review*, 57, 80. doi:10.1007/s10462-023-10652-8

AWS (2025) *Modernize and migrate on-premises fraud detection machine learning workflows to Amazon SageMaker*. AWS Machine Learning Blog. [Vendor case study]

AWS Solutions Library (n.d.) *Guidance for Fraud Detection using Machine Learning on AWS*. GitHub: aws-solutions-library-samples/fraud-detection-using-machine-learning. [Reference architecture]

Baisholan, N., Dietz, J.E., Gnatyuk, S., et al. (2025) 'A Systematic Review of Machine Learning in Credit Card Fraud Detection Under Original Class Imbalance', *Computers*, 14(10), 437. doi:10.3390/computers14100437

Brandt, J. & Lanzén, E. (2020) *A Comparative Review of SMOTE and ADASYN in Imbalanced Data Classification*. Bachelor's thesis, Department of Statistics, Uppsala University.

Cheah, P.C.Y., Yang, Y. & Lee, B.G. (2023) 'Enhancing Financial Fraud Detection through Addressing Class Imbalance Using Hybrid SMOTE-GAN Techniques', *International Journal of Financial Studies*, 11(3), 110. doi:10.3390/ijfs11030110

Darwish, S.M., Salama, A.I. & Elzoghabi, A.A. (2025) 'Intelligent approach to detecting online fraudulent trading with solution for imbalanced data in fintech forensics', *Scientific Reports*, 15. doi:10.1038/s41598-025-01223-8

Fernández, A., García, S., Herrera, F. & Chawla, N.V. (2018) 'SMOTE for Learning from Imbalanced Data: Progress and Challenges, Marking the 15-year Anniversary', *Journal of Artificial Intelligence Research*, 61, 863–905.

Gnip, P., Vokorokos, L. & Drotár, P. (2021) 'Selective oversampling approach for strongly imbalanced data', *PeerJ Computer Science*, 7, e604.

He, H. & Ma, Y. (eds.) (2013) *Imbalanced Learning: Foundations, Algorithms, and Applications*. Wiley-IEEE Press.

Idris, N.L. (2025) *Building a Three-Model, Real-Time Fraud Detection System with FastAPI*. Medium, 30 August. [Grey literature]

Kennedy, R.K.L., Villanustre, F., Khoshgoftaar, T.M. & Salekshahrezaee, Z. (2024) 'Synthesizing class labels for highly imbalanced credit card fraud detection data', *Journal of Big Data*, 11, 38. doi:10.1186/s40537-024-00897-7

Li, Y., Yang, Y., Song, P., Duan, L. & Ren, R. (2025) 'An improved SMOTE algorithm for enhanced imbalanced data classification by expanding sample generation space', *Scientific Reports*, 15, 23521.

Liu, W., Fan, H., Xia, M. & Xia, M. (2022) 'A focal-aware cost-sensitive boosted tree for imbalanced credit scoring', *Expert Systems with Applications*, 208, 118158. doi:10.1016/j.eswa.2022.118158

Matharaarachchi, S., Domaratzki, M. & Muthukumarana, S. (2024) 'Enhancing SMOTE for imbalanced data with abnormal minority instances', *Machine Learning with Applications*, 18, 100597. doi:10.1016/j.mlwa.2024.100597

Mienye, I.D. & Sun, Y. (2021) 'Performance analysis of cost-sensitive learning methods with application to imbalanced medical data', *Informatics in Medicine Unlocked*, 25, 100690. doi:10.1016/j.imu.2021.100690

Papanastassiou, A., Camaiani, B., Lenzi, P. & Crupi, R. (2026) 'A Reinforcement Learning Framework for Fraud Detection in Highly Imbalanced Financial Data', *Applied Sciences*, 16(1), 252. doi:10.3390/app16010252

Peykani, P., Peymany Foroushany, M., Tanasescu, C., Sargolzaei, M. & Kamyabfar, H. (2025) 'Evaluation of Cost-Sensitive Learning Models in Forecasting Business Failure of Capital Market Firms', *Mathematics*, 13(3), 368. doi:10.3390/math13030368

Ravaglia, A. (2022) *Imbalanced classification in Fraud Detection*. Medium, Data Reply IT | DataTech, 31 May. [Grey literature]

Ruchay, A., Feldman, E., Cherbadzhi, D. & Sokolov, A. (2023) 'The Imbalanced Classification of Fraudulent Bank Transactions Using Machine Learning', *Mathematics*, 11(13), 2862. doi:10.3390/math11132862

Suguna, R., Suriya Prakash, J., Aditya Pai, H., Mahesh, T.R., Vinoth Kumar, V. & Yimer, T.E. (2025) 'Mitigating class imbalance in churn prediction with ensemble methods and SMOTE', *Scientific Reports*, 15, 16256.

Wang, C., Deng, C. & Wang, S. (2020) 'Imbalance-XGBoost: leveraging weighted and focal losses for binary label-imbalanced classification with XGBoost', *Pattern Recognition Letters*, 136, 190–197. doi:10.1016/j.patrec.2020.05.035

### Cited via secondary sources — obtain before Chapter 3

Batista, G.E.A.P.A., Prati, R.C. & Monard, M.C. (2004) 'A study of the behaviour of several methods for balancing machine learning training data', *SIGKDD Explorations*, 6(1), 20–29. [via Fernández et al., 2018]

Chawla, N.V., Bowyer, K.W., Hall, L.O. & Kegelmeyer, W.P. (2002) 'SMOTE: Synthetic Minority Over-sampling Technique', *Journal of Artificial Intelligence Research*, 16, 321–357. **[MISSING — required]**

He, H., Bai, Y., Garcia, E.A. & Li, S. (2008) 'ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning', *IEEE IJCNN*. **[MISSING — required]**

Liu, X.Y., Wu, J. & Zhou, Z.H. (2009) 'Exploratory Undersampling for Class-Imbalance Learning', *IEEE Transactions on Systems, Man, and Cybernetics – Part B*, 39(2), 539–550. **[MISSING — required]**

López, V., Fernández, A., Moreno-Torres, J.G. & Herrera, F. (2012) 'Analysis of preprocessing vs. cost-sensitive learning for imbalanced classification', *Expert Systems with Applications*. [via Fernández et al., 2018 — key source for §6.1]

Moreno-Torres, J.G., Sáez, J.A. & Herrera, F. (2012) 'Study on the impact of partition-induced dataset shift on k-fold cross-validation', *IEEE TNNLS*. [via Fernández et al., 2018 — DOB-SCV]

Seiffert, C., Khoshgoftaar, T.M., Van Hulse, J. & Napolitano, A. (2010) 'RUSBoost: A Hybrid Approach to Alleviating Class Imbalance', *IEEE Transactions on Systems, Man, and Cybernetics – Part A*, 40(1), 185–197. **[MISSING — required]**
