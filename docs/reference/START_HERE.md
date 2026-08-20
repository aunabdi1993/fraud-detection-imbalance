# 🚀 START HERE - Your Complete MSc Dissertation Package

**Created:** August 7, 2026  
**Your Submission Date:** January 7, 2027  
**Timeline:** 24 weeks (5.7 months)  
**Method:** Assistive coding with Claude Code  

---

## WHAT YOU HAVE (Everything)

### 📄 **Master Document (READ FIRST)**
**File:** `DISSERTATION_COMPLETE_GUIDE.md` (8,500 words)

**Contains:**
- Executive summary (key changes, advantages, success metrics)
- Complete 24-week timeline with WEEK-BY-WEEK breakdown
- Your 5,000-word literature review (integrated)
- Implementation guide with code examples
- Weekly action checklists
- Quick reference tables
- All resources and citations

**Read Time:** 2-3 hours (but worth every minute)  
**This replaces:** The original 11-month plan with realistic 5.7-month plan

**👉 Start here. Read it before anything else.**

---

### 📊 **Supporting Documents**

| File | Purpose | Read Time | When |
|------|---------|-----------|------|
| `IMPLEMENTATION_ROADMAP.md` | Week-by-week code tasks + writing milestones | 20 min | After main guide |
| `LIT_REVIEW_TO_IMPLEMENTATION.md` | How lit review maps to code and chapters | 15 min | During Week 1 |
| `THIS_WEEK.md` | Your immediate action plan (Aug 7-14) | 10 min | This week |
| `CONVERT_TO_PDF.md` | How to convert markdown to PDF for Claude Code | 5 min | When uploading |

**Total supporting docs:** 50 minutes to read all

---

### 💻 **Code Templates (Ready to Use)**

Located in project folder structure:

```
fraud-detection-dissertation/
├── src/
│   ├── data_loader.py (300+ lines) ✅ Ready
│   ├── evaluation.py (500+ lines) ✅ Ready
│   └── [MORE to come as you progress]
├── requirements.txt ✅ Ready
├── README.md ✅ Ready
└── [Notebooks and experiments added Weeks 1-13]
```

**Status:** 
- ✅ Data loading template: DONE
- ✅ Evaluation framework: DONE
- ✅ Requirements: DONE
- ⏳ Feature engineering: I'll provide Week 3
- ⏳ Baseline models: I'll provide Week 4
- ⏳ Imbalance methods (12 techniques): I'll provide Week 6
- ⏳ API: I'll provide Week 12

---

### 📋 **Project Structure Created**

```
fraud-detection-dissertation/
├── data/
│   ├── raw/              ← Kaggle dataset goes here
│   └── processed/        ← Train/val/test splits
├── src/
│   ├── __init__.py
│   ├── data_loader.py    (300+ lines)
│   ├── evaluation.py     (500+ lines)
│   ├── preprocessing.py  (coming Week 3)
│   ├── baseline_models.py (coming Week 4)
│   ├── imbalance_methods.py (coming Week 6)
│   ├── inference_profiler.py (coming Week 10)
│   └── api.py            (coming Week 12)
├── experiments/
│   ├── mlflow_setup.py
│   ├── run_all_experiments.py (coming Week 6)
│   └── results/          ← MLflow tracking results
├── notebooks/
│   ├── 01_eda.ipynb      (coming Week 2)
│   ├── 02_baseline.ipynb (coming Week 4)
│   └── 03_results_analysis.ipynb (coming Week 10)
├── dissertation/
│   ├── chapters/
│   │   ├── ch1_introduction.md
│   │   ├── ch2_literature_review.md
│   │   ├── ch3_methodology.md
│   │   ├── ch4_system_architecture.md
│   │   ├── ch5_results.md
│   │   ├── ch6_discussion.md
│   │   └── ch7_conclusion.md
│   ├── figures/          ← All plots and visualizations
│   ├── tables/           ← All results tables
│   └── references.bib    ← Zotero export
├── README.md             ✅ Created
├── requirements.txt      ✅ Created
├── DISSERTATION_COMPLETE_GUIDE.md ✅ Created
├── THIS_WEEK.md          ✅ Created
├── IMPLEMENTATION_ROADMAP.md ✅ Created
├── LIT_REVIEW_TO_IMPLEMENTATION.md ✅ Created
└── .gitignore           ✅ Created
```

Everything is scaffolded. You just fill in the content.

---

## YOUR TIMELINE AT A GLANCE

### **MONTH 1: Aug 7 - Sep 4 (Data & Baselines)**
- **Week 1-2:** Data loading, EDA (6 hours)
- **Week 3:** Feature engineering (5 hours)
- **Week 4-5:** Baseline models, evaluation framework (8 hours)
- **Deliverables:** Chapters 1, 3 DRAFTED | 3 baseline models trained
- **Total hours:** 19 hours YOUR time

### **MONTH 2: Sep 4 - Oct 2 (12 Techniques + Novel Contribution)**
- **Week 6-9:** 12 imbalance techniques × 3 classifiers × 5-fold CV (12 hours)
- **Week 10-11:** Latency profiling, results analysis (8 hours) ← **YOUR NOVEL CONTRIBUTION**
- **Deliverables:** Chapters 2, 5 DRAFTED | 180 trained models | Accuracy-latency trade-off analysis
- **Total hours:** 20 hours YOUR time (experiments run overnight)

### **MONTH 3: Oct 2 - Nov 6 (API + More Writing)**
- **Week 12-13:** API development and deployment (7 hours)
- **Deliverables:** Working deployed fraud detection API | Chapter 4 COMPLETE
- **Total hours:** 7 hours YOUR time (I write 90% of code)

### **MONTH 4: Nov 6 - Dec 24 (Writing Sprint)**
- **Week 14-21:** Intensive writing - Chapters 2, 1, 5, 6, 7, References, Abstract (54 hours)
- **Deliverables:** COMPLETE 9,000-10,000 word dissertation
- **Total hours:** 54 hours YOUR time (realistic: 6-7 hrs/week spread over 8 weeks)

### **FINAL: Dec 24 - Jan 7 (Submit)**
- **Week 22-24:** Supervisor feedback, final review (5 hours)
- **Deliverables:** SUBMITTED ✅
- **Total hours:** 5 hours YOUR time

**GRAND TOTAL: 105 hours your time over 24 weeks = 4.4 hrs/week average**

---

## YOUR IMMEDIATE ACTION PLAN (THIS WEEK - Aug 7-14)

### **Today (Aug 7):**
- [ ] Read this file (5 min)
- [ ] Read `THIS_WEEK.md` (10 min)
- [ ] Download Kaggle dataset (5 min)

### **This Week (Aug 7-14):**
- [ ] Read `DISSERTATION_COMPLETE_GUIDE.md` FULLY (2-3 hours)
- [ ] Email supervisor with scope confirmation (use template in guide) (30 min)
- [ ] Set up Python venv: `python -m venv venv && source venv/bin/activate`
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Create directory structure (run scripts in `IMPLEMENTATION_ROADMAP.md`)
- [ ] **Total: ~4 hours**

### **Week 2 (Aug 15-21):**
- [ ] I'll provide: `notebooks/01_eda.ipynb` template
- [ ] You review data_loader.py, run it
- [ ] You review EDA notebook
- [ ] Begin writing Chapter 1 problem statement
- [ ] **Total: ~6 hours**

### **Week 3-4 (Aug 22-Sep 4):**
- [ ] I'll provide: `src/preprocessing.py` and `src/baseline_models.py`
- [ ] You review, run, understand
- [ ] Complete Chapter 3 Methodology
- [ ] **Total: ~13 hours**

**By Sept 4: Everything set up, first 30 hours invested, ready for technique comparison phase.**

---

## HOW TO CONVERT TO PDF

**Quick option (GitHub):**
1. Push repo to GitHub
2. Open file in browser
3. Ctrl+P → Save as PDF
4. Done (60 seconds)

**Better option (Pandoc):**
```bash
pandoc DISSERTATION_COMPLETE_GUIDE.md -o DISSERTATION_COMPLETE_GUIDE.pdf --toc
```

See `CONVERT_TO_PDF.md` for 4 different methods.

---

## NOVEL CONTRIBUTIONS (Your Differentiators)

### **Gap 1: Latency Characterization** (Week 10-11)
> "First systematic measurement of inference latency across imbalance-handling techniques"
- Only 2/21 existing studies report latency at all
- None measure technique-specific latency
- Your contribution: Accuracy-latency trade-off visualization (Chapter 5 §5.4)

### **Gap 3: Controlled Comparison** (Weeks 6-9)
> "Rigorous comparison with fixed pipeline, statistical significance testing"
- Fixed random seed (reproducibility)
- Identical train/val/test splits
- Identical hyperparameter tuning budget
- Friedman-Nemenyi significance testing
- NO novel algorithm (but rigorous evaluation is more valuable)

### **Supporting: Evaluation Rigor** (Week 4)
> "AUC-PR as primary metric (top 11% of published work)"
- Only 11.4% of existing studies use AUC-PR despite it being theoretically optimal
- Your adoption = top 11% on evaluation rigor alone

---

## SUCCESS CRITERIA AT SUBMISSION

**Dissertation (9,000-10,000 words):**
- ✅ Chapter 1: Introduction (1,200 words)
- ✅ Chapter 2: Literature Review (2,000 words)
- ✅ Chapter 3: Methodology (1,500 words)
- ✅ Chapter 4: System Architecture (1,200 words)
- ✅ Chapter 5: Results & Evaluation (2,500 words + 4 figures + 3 tables)
- ✅ Chapter 6: Discussion (1,200 words)
- ✅ Chapter 7: Conclusion (1,000 words)
- ✅ References (~50-60 citations)
- ✅ Abstract (300 words)

**Code & Deployment:**
- ✅ 12 imbalance techniques implemented and compared
- ✅ 180 trained models (12 techniques × 3 classifiers × 5 folds)
- ✅ Statistical significance testing (Friedman-Nemenyi)
- ✅ Working deployed API at public URL
- ✅ Latency profiling across all techniques
- ✅ GitHub repo with documented, clean code

**Expected Grade:** Merit (60%) to Distinction (70%)
- Rigorous evaluation of existing methods (not novel algorithm, but better than most MSc work)
- Production-quality system (API deployment)
- Practical insights (latency-accuracy trade-off)
- Excellent writing (all chapters polished)

---

## CRITICAL SUCCESS FACTORS

1. **Email supervisor THIS WEEK** with scope confirmation
   - Don't wait, don't assume approval
   - Use template in `DISSERTATION_COMPLETE_GUIDE.md`

2. **Follow the timeline exactly**
   - Weeks 1-13: Code/experiments (I write most, you run)
   - Weeks 14-21: Writing (you write, I provide structure)
   - No descoping techniques or API

3. **Write incrementally**
   - 200-300 words per week starting Week 3
   - Not 100 hours crammed into December
   - Makes iteration and feedback easier

4. **Track hours weekly**
   - You have ~150 hours available
   - Allocating 105 leaves 45-hour buffer
   - If you exceed, something went wrong

5. **Use code templates**
   - Don't rewrite anything I've provided
   - Review once (10-15 min), run it, move on
   - This is how you get 4x speed-up

---

## MATERIALS CHECKLIST

### 📚 Documentation (Read in this order):
- [ ] START_HERE.md (this file - 5 min)
- [ ] THIS_WEEK.md (your action plan - 10 min)
- [ ] DISSERTATION_COMPLETE_GUIDE.md (master guide - 2-3 hours)
- [ ] IMPLEMENTATION_ROADMAP.md (week-by-week - 20 min)
- [ ] LIT_REVIEW_TO_IMPLEMENTATION.md (mapping - 15 min)

### 💻 Code & Templates:
- [ ] `src/data_loader.py` (300+ lines - ready)
- [ ] `src/evaluation.py` (500+ lines - ready)
- [ ] `requirements.txt` (dependencies - ready)
- [ ] More templates provided as you progress

### 📖 References:
- [ ] Your literature review (5,000 words - integrated in main guide)
- [ ] 50+ academic papers (citations in guide)
- [ ] Zotero bibliography setup (Week 3)

---

## QUICK LINKS

**Main Reference:**
- DISSERTATION_COMPLETE_GUIDE.md ← Bookmark this

**This Week:**
- THIS_WEEK.md ← Do this first

**Upload to Claude Code:**
- DISSERTATION_COMPLETE_GUIDE.pdf (convert using CONVERT_TO_PDF.md)
- All `.md` files as context
- All code templates as base

---

## REMEMBER

✅ **You're not starting from zero.** I've built the scaffolding. You fill in the content.

✅ **You're not rebuilding code.** I write most of it. You review and run experiments.

✅ **You're not in a crunch.** 54 hours writing over 8 weeks (realistic), not 100 hours in 4 weeks (unsustainable).

✅ **You have a literature review that justifies every choice.** No guessing about methodology.

✅ **Your novel contributions are defensible.** Latency analysis + rigorous evaluation = original work worth publishing.

---

## NEXT STEP

1. **Read THIS_WEEK.md (10 min)**
2. **Email your supervisor TODAY (30 min)**
3. **Download dataset (5 min)**
4. **Read DISSERTATION_COMPLETE_GUIDE.md fully (2-3 hours)**
5. **Start Week 1 data loading**

That's it. Everything else follows from the timeline.

---

**You've got 24 weeks. You've got a plan. You've got code templates. You've got a literature review.**

**Now go build something great.** 🚀

---

**Document:** START_HERE.md  
**Version:** 1.0  
**Date:** August 7, 2026  
**Status:** Ready to execute  

**Questions?** Everything is in DISSERTATION_COMPLETE_GUIDE.md.

Good luck!
