# 🏥 VITALSCAN — AI Multi-Disease Health Risk Engine

> A production-grade, single-file ML web app that **simultaneously predicts risk for 4 major diseases** — Heart Disease, Type 2 Diabetes, Hypertension, and Stroke — from 18 clinical biomarkers. Uses `ClassifierChain` to model label dependencies between diseases, plus 5 `MultiOutputClassifier` variants for comparison.

**Made by [Musharib](https://linktr.ee/Musharib_)**

[![GitHub](https://img.shields.io/badge/GitHub-musharibramzan950--bit-181717?style=flat&logo=github)](https://github.com/musharibramzan950-bit)
[![Linktree](https://img.shields.io/badge/Linktree-Musharib__-43E55E?style=flat&logo=linktree)](https://linktr.ee/Musharib_)

---

## 🆕 What Makes This Different From The Others

| Project | Problem Type | ML Technique |
|---|---|---|
| HOMEVAL | Single regression | Ensemble regressors |
| CREDIX | Binary classification | Stratified + CV |
| FRAUDSENSE | Imbalanced binary | SMOTE + threshold tuning |
| **VITALSCAN** | **Multi-output / Multi-label** | **ClassifierChain + MultiOutputClassifier** |

This is fundamentally different ML — predicting **4 targets at once**, with models that understand **disease co-occurrence** (e.g., someone with hypertension is more likely to have heart disease).

---

## ✨ Features

- **Predicts 4 diseases simultaneously** — Heart Disease, Type 2 Diabetes, Hypertension, Stroke Risk
- **ClassifierChain** — captures label dependency (e.g., hypertension → heart disease → stroke)
- **6 ML models** — all wrapped in MultiOutputClassifier or ClassifierChain
- **18 clinical biomarkers** — vitals, blood labs, lifestyle, metabolic markers
- **Radar/Spider chart** — visual multi-disease risk profile
- **Overall Health Index** — 0–100 composite score
- **Metabolic Syndrome Score** — 5-criteria clinical assessment
- **Per-disease risk cards** with progress bars and verdicts
- **SHAP-style attribution** — per-feature, per-disease risk contribution
- **ROC curves** for all 4 diseases on one chart
- **Feature Importance** — top 12 clinical predictors
- **Disease prevalence by age group** — line chart across 6 cohorts
- **Risk by gender** — grouped bar comparison
- **10,000 patient records** — largest dataset in the series
- **Clean clinical white/mint aesthetic** — Fraunces serif + dark forest green hero

---

## 🚀 Quick Start

```bash
python vitalscan.py
```

Opens at → `http://localhost:8080`

Zero config. Auto-installs everything.

---

## 📦 Dependencies

```bash
pip install flask scikit-learn numpy pandas
```

**Python 3.8+** required.

---

## 🧠 ML Architecture

### Multi-Output Strategy

```
Patient Features (18) 
        │
        ├── MultiOutputClassifier(RandomForest)   → [Heart, Diabetes, Hypertension, Stroke]
        ├── MultiOutputClassifier(GradientBoost)  → [Heart, Diabetes, Hypertension, Stroke]
        ├── MultiOutputClassifier(ExtraTrees)     → [Heart, Diabetes, Hypertension, Stroke]
        ├── MultiOutputClassifier(NeuralNet)      → [Heart, Diabetes, Hypertension, Stroke]
        ├── MultiOutputClassifier(LogisticReg)    → [Heart, Diabetes, Hypertension, Stroke]
        └── ClassifierChain(RandomForest)         → Hypertension → Heart → Diabetes → Stroke
```

### ClassifierChain — The Key Innovation
Unlike `MultiOutputClassifier` which trains each disease independently, `ClassifierChain` trains in order `[Hypertension → Heart Disease → Diabetes → Stroke]`, feeding each prediction as a feature into the next model — capturing real-world disease dependencies.

| Model | Technique | Notes |
|---|---|---|
| Random Forest | `MultiOutputClassifier` | 200 trees, balanced |
| Gradient Boost | `MultiOutputClassifier` | 200 trees, lr=0.08 |
| Extra Trees | `MultiOutputClassifier` | 200 trees, balanced |
| Neural Net | `MultiOutputClassifier` | MLP 128→64→32, Adam |
| Logistic Reg. | `MultiOutputClassifier` | L2, balanced |
| Classifier Chain | `ClassifierChain` | RF base, order=[2,0,1,3] |

### Evaluation Metrics (Multi-Label)
- **Macro AUC** — average AUC across all 4 disease targets
- **Macro F1** — average F1 across all diseases
- **Hamming Loss** — fraction of label predictions wrong (lower = better)
- **Per-disease AUC & F1** — individual breakdown for all 4 conditions

---

## 🩺 Input Features (18 Clinical Biomarkers)

| Feature | Normal Range | High Risk Indicator |
|---|---|---|
| Age | — | >50 |
| Gender | — | Male (higher cardiovascular risk) |
| BMI | 18.5–24.9 | >30 (obese) |
| Systolic BP | <120 mmHg | >130 |
| Diastolic BP | <80 mmHg | >85 |
| Fasting Glucose | 70–99 mg/dL | >100 (pre-diabetic) |
| HbA1c | <5.7% | >6.5% (diabetic) |
| Total Cholesterol | <200 mg/dL | >240 |
| LDL | <100 mg/dL | >160 |
| HDL | >60 mg/dL | <40 (low = bad) |
| Triglycerides | <150 mg/dL | >200 |
| Waist Circumference | <88cm (F), <102cm (M) | Metabolic risk marker |
| Sleep Hours | 7–9 hrs | <6 or >9 |
| Stress Level | — | >7/10 |
| Smoking | Never | Current |
| Alcohol Use | None/Moderate | Heavy |
| Physical Activity | Moderate–Active | Sedentary |
| Family History | — | Yes = +risk |

---

## 📊 Dashboard Panels

| Panel | Description |
|---|---|
| Patient Form | 18-input clinical profile entry |
| Overall Health Index | 0–100 composite score with verdict |
| Metabolic Syndrome Score | 5-criteria test (BMI, BP, glucose, HDL, triglycerides) |
| 4 Risk Cards | Per-disease probability % with bar + verdict |
| Risk Drivers | Feature attribution across all 4 diseases |
| Model Leaderboard | Macro AUC, F1, Hamming Loss + per-disease breakdown |
| Radar Chart | Spider chart of all 4 disease risk levels |
| ROC Curves | All 4 diseases overlaid on one chart |
| Feature Importance | Top 12 clinical predictors |
| Age Prevalence | Disease % by age group (20–85) |
| Gender Risk | Side-by-side comparison |

---

## 📸 Terminal Output

```
══════════════════════════════════════════════════════════
  🏥  VITALSCAN — AI Multi-Disease Health Risk Engine
  Made by Musharib | linktr.ee/Musharib_
══════════════════════════════════════════════════════════
  ⚙️  Generating 10,000 patient records…
     Heart Disease         prevalence: 24.3%
     Type 2 Diabetes       prevalence: 18.7%
     Hypertension          prevalence: 38.1%
     Stroke Risk           prevalence: 8.2%
  🧠  Training 6 multi-output ML models…
     (ClassifierChain + MultiOutputClassifier)

     Random Forest         AUC=0.9210  F1=0.8430  Hamming=0.1120
     Gradient Boost        AUC=0.9340  F1=0.8560  Hamming=0.1050 ◄ BEST
     Extra Trees           AUC=0.9180  F1=0.8390  Hamming=0.1140
     Neural Net            AUC=0.9010  F1=0.8210  Hamming=0.1200
     Logistic Reg.         AUC=0.8640  F1=0.7920  Hamming=0.1350
     Classifier Chain      AUC=0.9190  F1=0.8410  Hamming=0.1130

  🌐  Running at → http://localhost:8080
══════════════════════════════════════════════════════════
```

---

## 🛠️ Config

```bash
PORT=9090 python vitalscan.py
```

---

## 👤 Author

**Musharib Ramzan**

- 🌐 [Linktree](https://linktr.ee/Musharib_)
- 💻 [GitHub](https://github.com/musharibramzan950-bit)

---

## 📄 License

MIT — free to use, modify, and distribute.

---

*Built with Python · Flask · scikit-learn · Chart.js*
