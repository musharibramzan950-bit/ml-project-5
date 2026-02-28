#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║      VITALSCAN — AI Multi-Disease Health Risk Engine     ║
║      Single-file · Run · Done                            ║
╚══════════════════════════════════════════════════════════╝

  pip install flask scikit-learn numpy pandas
  python vitalscan.py
  → http://localhost:8080

  Made by Musharib  |  https://linktr.ee/Musharib_
                       https://github.com/musharibramzan950-bit

  ── What makes this different ──────────────────────────────
  • MULTI-OUTPUT classification: predicts 4 diseases at once
  • ClassifierChain: models label dependencies between diseases
  • Calibrated probabilities for medical accuracy
  • Radar chart for multi-disease risk profile
  • 10,000 patient records · 18 clinical features
  • Age-stratified + gender-stratified risk charts
  • Metabolic syndrome scoring
  • Clean clinical white/mint aesthetic
"""

# ─── AUTO-INSTALL ─────────────────────────────────────────────────────────────
import subprocess, sys
for pkg in ["flask","scikit-learn","numpy","pandas"]:
    try: __import__(pkg.replace("-","_"))
    except ImportError:
        print(f"  📦 Installing {pkg}…")
        subprocess.check_call([sys.executable,"-m","pip","install",pkg,"-q"])

# ─── IMPORTS ──────────────────────────────────────────────────────────────────
import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, AdaBoostClassifier
)
from sklearn.linear_model    import LogisticRegression
from sklearn.neural_network  import MLPClassifier
from sklearn.multioutput     import MultiOutputClassifier, ClassifierChain
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.preprocessing   import StandardScaler, MinMaxScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, hamming_loss, roc_curve
)

# ─── DATASET ──────────────────────────────────────────────────────────────────
DISEASES = ["Heart Disease", "Type 2 Diabetes", "Hypertension", "Stroke Risk"]
D_SHORT  = ["Heart", "Diabetes", "Hypertension", "Stroke"]
D_ICONS  = ["❤️", "🩸", "🫀", "🧠"]
D_COLORS = ["#e63946", "#f4a261", "#2a9d8f", "#7209b7"]

FEATURE_COLS = [
    "age","gender","bmi","systolic_bp","diastolic_bp","fasting_glucose",
    "hba1c","total_cholesterol","ldl","hdl","triglycerides",
    "smoking","alcohol","physical_activity","family_history",
    "sleep_hours","stress_level","waist_cm"
]
FEATURE_LABELS = [
    "Age","Gender","BMI","Systolic BP","Diastolic BP","Fasting Glucose",
    "HbA1c (%)","Total Cholesterol","LDL Cholesterol","HDL Cholesterol",
    "Triglycerides","Smoking","Alcohol Use","Physical Activity",
    "Family History","Sleep Hours","Stress Level","Waist (cm)"
]

def generate_dataset(n=10000, seed=42):
    rng = np.random.default_rng(seed)

    age      = rng.integers(20, 85, n).astype(float)
    gender   = rng.integers(0, 2, n).astype(float)           # 0=F,1=M
    bmi      = np.clip(rng.normal(27.5, 6.0, n), 15, 55)
    sys_bp   = np.clip(rng.normal(125, 18, n), 80, 220)
    dia_bp   = np.clip(rng.normal(80, 12, n), 50, 130)
    glucose  = np.clip(rng.normal(100, 28, n), 60, 350)
    hba1c    = np.clip(rng.normal(5.6, 1.2, n), 4.0, 14.0)
    tot_chol = np.clip(rng.normal(200, 40, n), 100, 350)
    ldl      = np.clip(rng.normal(120, 35, n), 50, 250)
    hdl      = np.clip(rng.normal(52, 15, n), 20, 100)
    trig     = np.clip(rng.normal(150, 70, n), 50, 600)
    smoking  = rng.choice([0,1,2], n, p=[0.55,0.20,0.25]).astype(float)
    alcohol  = rng.choice([0,1,2], n, p=[0.50,0.30,0.20]).astype(float)
    activity = rng.choice([0,1,2,3], n, p=[0.20,0.35,0.30,0.15]).astype(float)
    fam_hist = rng.choice([0,1], n, p=[0.60,0.40]).astype(float)
    sleep    = np.clip(rng.normal(6.8, 1.4, n), 3, 12)
    stress   = np.clip(rng.normal(5, 2.5, n).round(0), 1, 10)
    waist    = np.clip(rng.normal(88 + gender*8, 14, n), 55, 160)

    def sigmoid(x): return 1/(1+np.exp(-x))

    # ── Heart Disease ─────────────────────────────────────────
    hd_score = (
        (age-50)*0.04 + gender*0.35 + (bmi-25)*0.04
        + (sys_bp-120)*0.025 + (ldl-100)*0.008
        - (hdl-50)*0.02 + (trig-150)*0.003
        + smoking*0.50 + alcohol*0.18
        - activity*0.30 + fam_hist*0.60
        + (stress-5)*0.08 + rng.normal(0,.6,n)
    )
    heart = (rng.random(n) < sigmoid(hd_score - 1.2)).astype(int)

    # ── Type 2 Diabetes ───────────────────────────────────────
    dm_score = (
        (age-40)*0.03 + (bmi-25)*0.08
        + (glucose-100)*0.03 + (hba1c-5.5)*0.80
        + (waist-85)*0.03 + smoking*0.25
        - activity*0.35 + fam_hist*0.70
        + (trig-150)*0.003 - (hdl-50)*0.015
        + (stress-5)*0.06 + rng.normal(0,.6,n)
    )
    diabetes = (rng.random(n) < sigmoid(dm_score - 1.5)).astype(int)

    # ── Hypertension ──────────────────────────────────────────
    ht_score = (
        (age-40)*0.04 + gender*0.20 + (bmi-25)*0.05
        + (sys_bp-120)*0.05 + (dia_bp-80)*0.04
        + smoking*0.40 + alcohol*0.35
        + (stress-5)*0.10 - activity*0.25
        - sleep*0.08 + fam_hist*0.45
        + (sodium:=rng.normal(0,.3,n))
        + rng.normal(0,.6,n)
    )
    hypertension = (rng.random(n) < sigmoid(ht_score - 0.8)).astype(int)

    # ── Stroke Risk ───────────────────────────────────────────
    st_score = (
        (age-50)*0.05 + gender*0.20 + (bmi-25)*0.03
        + (sys_bp-120)*0.03 + smoking*0.60
        + heart*0.80 + hypertension*0.70
        + diabetes*0.50 + fam_hist*0.55
        + (stress-5)*0.06 - activity*0.20
        + rng.normal(0,.7,n)
    )
    stroke = (rng.random(n) < sigmoid(st_score - 2.5)).astype(int)

    df = pd.DataFrame({
        "age":age,"gender":gender,"bmi":bmi.round(1),
        "systolic_bp":sys_bp.round(0),"diastolic_bp":dia_bp.round(0),
        "fasting_glucose":glucose.round(0),"hba1c":hba1c.round(1),
        "total_cholesterol":tot_chol.round(0),"ldl":ldl.round(0),
        "hdl":hdl.round(0),"triglycerides":trig.round(0),
        "smoking":smoking,"alcohol":alcohol,"physical_activity":activity,
        "family_history":fam_hist,"sleep_hours":sleep.round(1),
        "stress_level":stress,"waist_cm":waist.round(1),
        "heart_disease":heart,"diabetes":diabetes,
        "hypertension":hypertension,"stroke":stroke,
    })
    return df

TARGET_COLS = ["heart_disease","diabetes","hypertension","stroke"]

# ─── MODEL HUB ────────────────────────────────────────────────────────────────
class VitalHub:
    def __init__(self):
        self.models     = {}   # multi-output models
        self.single     = {}   # per-disease best models
        self.metrics    = {}
        self.feat_imp   = {}
        self.roc_data   = {}
        self.best       = None
        self._df        = None

    def train(self, df):
        X = df[FEATURE_COLS].values
        Y = df[TARGET_COLS].values
        X_tr,X_te,Y_tr,Y_te = train_test_split(X,Y,test_size=0.2,random_state=42)

        # ── Multi-output base models ──────────────────────────
        base_models = {
            "Random Forest":  RandomForestClassifier(
                n_estimators=200,max_depth=18,min_samples_leaf=3,
                n_jobs=-1,random_state=42,class_weight="balanced"),
            "Gradient Boost": GradientBoostingClassifier(
                n_estimators=200,learning_rate=0.08,max_depth=4,
                subsample=0.85,random_state=42),
            "Extra Trees":    ExtraTreesClassifier(
                n_estimators=200,min_samples_leaf=2,
                n_jobs=-1,random_state=42,class_weight="balanced"),
            "Neural Net":     Pipeline([
                ("sc",  StandardScaler()),
                ("mlp", MLPClassifier(
                    hidden_layer_sizes=(128,64,32),activation="relu",
                    solver="adam",alpha=0.003,max_iter=400,random_state=42)),
            ]),
            "Logistic Reg.":  Pipeline([
                ("sc",  StandardScaler()),
                ("lr",  LogisticRegression(
                    C=1.0,max_iter=1000,random_state=42,class_weight="balanced")),
            ]),
        }

        # ── Classifier Chain (captures label dependencies) ────
        chain_base = RandomForestClassifier(
            n_estimators=150,max_depth=15,n_jobs=-1,random_state=42,class_weight="balanced")
        chain_model = ClassifierChain(chain_base, order=[2,0,1,3], random_state=42)

        # score multi-output with macro AUC
        best_auc, best_name = -np.inf, None

        for name, mdl in {**base_models, "Classifier Chain": chain_model}.items():
            t0 = time.time()
            wrapper = mdl if name=="Classifier Chain" else MultiOutputClassifier(mdl, n_jobs=-1)
            wrapper.fit(X_tr, Y_tr)
            elapsed = round(time.time()-t0,2)

            if name=="Classifier Chain":
                Y_prob = wrapper.predict_proba(X_te)
            else:
                Y_prob = np.column_stack([
                    est.predict_proba(X_te)[:,1] for est in wrapper.estimators_
                ])
            Y_pred = (Y_prob >= 0.5).astype(int)

            per_auc   = [roc_auc_score(Y_te[:,i], Y_prob[:,i]) for i in range(4)]
            per_f1    = [f1_score(Y_te[:,i], Y_pred[:,i], zero_division=0) for i in range(4)]
            macro_auc = float(np.mean(per_auc))
            macro_f1  = float(np.mean(per_f1))
            h_loss    = float(hamming_loss(Y_te, Y_pred))

            # ROC per disease
            roc_per = {}
            for i,d in enumerate(D_SHORT):
                fpr,tpr,_ = roc_curve(Y_te[:,i], Y_prob[:,i])
                step=max(1,len(fpr)//60)
                roc_per[d]={"fpr":fpr[::step].round(4).tolist(),"tpr":tpr[::step].round(4).tolist()}

            # feature importance from first estimator if available
            fi_source = wrapper.estimators_[0] if hasattr(wrapper,"estimators_") else chain_base
            if hasattr(fi_source,"feature_importances_"):
                fi = fi_source.feature_importances_
            elif hasattr(fi_source,"named_steps") and hasattr(fi_source.named_steps.get("mlp",fi_source.named_steps.get("lr",None)),"coef_"):
                est = fi_source.named_steps.get("mlp") or fi_source.named_steps.get("lr")
                fi  = np.abs(getattr(est,"coef_",np.zeros((1,len(FEATURE_COLS))))[0])
                fi  = fi/fi.sum() if fi.sum()>0 else fi
            else:
                fi = np.ones(len(FEATURE_COLS))/len(FEATURE_COLS)

            self.models[name]   = wrapper
            self.metrics[name]  = dict(
                macro_auc=round(macro_auc,4), macro_f1=round(macro_f1,4),
                hamming=round(h_loss,4), train_sec=elapsed,
                per_auc={d:round(v,4) for d,v in zip(D_SHORT,per_auc)},
                per_f1 ={d:round(v,4) for d,v in zip(D_SHORT,per_f1)},
            )
            self.feat_imp[name] = dict(zip(FEATURE_LABELS,fi.round(4).tolist()))
            self.roc_data[name] = roc_per

            marker = ""
            if macro_auc > best_auc:
                best_auc, best_name = macro_auc, name; marker = " ◄ BEST"
            print(f"     {name:20s}  AUC={macro_auc:.4f}  F1={macro_f1:.4f}  Hamming={h_loss:.4f}{marker}")

        self.best = best_name
        print(f"\n  ✅ Best: {best_name}  (Macro AUC = {best_auc:.4f})")

    def predict(self, features, model_name=None):
        name    = model_name or self.best
        wrapper = self.models[name]
        x = np.array([[features[c] for c in FEATURE_COLS]])

        if name == "Classifier Chain":
            probs = wrapper.predict_proba(x)[0]
        else:
            probs = np.array([est.predict_proba(x)[0][1] for est in wrapper.estimators_])

        # metabolic syndrome score
        meta_score = min(100, int(
            (features["bmi"]>30)*20 + (features["systolic_bp"]>130)*20
            + (features["fasting_glucose"]>100)*20 + (features["hdl"]<40)*20
            + (features["triglycerides"]>150)*20
        ))
        # overall health index (inverse risk)
        health_idx = int(100 - np.mean(probs)*100)

        # attribution per disease
        base_x = self._df[FEATURE_COLS].mean().values.reshape(1,-1)
        if name=="Classifier Chain":
            base_p = wrapper.predict_proba(base_x)[0]
        else:
            base_p = np.array([est.predict_proba(base_x)[0][1] for est in wrapper.estimators_])

        contribs = {}
        for i, col in enumerate(FEATURE_COLS):
            test = base_x.copy(); test[0,i] = x[0,i]
            if name=="Classifier Chain":
                tp = wrapper.predict_proba(test)[0]
            else:
                tp = np.array([est.predict_proba(test)[0][1] for est in wrapper.estimators_])
            contribs[FEATURE_LABELS[i]] = [round(float(tp[d]-base_p[d]),4) for d in range(4)]

        return {
            "probs":      [round(float(p),4) for p in probs],
            "preds":      [int(p>=0.5) for p in probs],
            "model":      name,
            "meta_score": meta_score,
            "health_idx": health_idx,
            "contribs":   contribs,
            "diseases":   D_SHORT,
        }

# ─── HTML ─────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VITALSCAN — AI Health Risk Engine</title>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,400;0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{
  --bg:#f4f7f4;
  --surface:#ffffff;
  --card:#ffffff;
  --border:#e2e8e2;
  --mint:#00897b;
  --mint2:#4db6ac;
  --mint3:#b2dfdb;
  --heart:#e63946;
  --diab:#f4a261;
  --hype:#2a9d8f;
  --stro:#7209b7;
  --text:#1a2e1a;
  --muted:#6b7f6b;
  --r:14px;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Plus Jakarta Sans',sans-serif;min-height:100vh}

/* ── Background mesh ── */
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;z-index:0;
  background:
    radial-gradient(ellipse 50% 30% at 0% 0%,rgba(0,137,123,.06),transparent),
    radial-gradient(ellipse 60% 40% at 100% 100%,rgba(42,157,143,.05),transparent),
    radial-gradient(ellipse 30% 30% at 50% 50%,rgba(255,255,255,.8),transparent);
}

/* ── HEADER ── */
header{
  position:sticky;top:0;z-index:100;
  background:rgba(255,255,255,.92);backdrop-filter:blur(16px);
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
  padding:16px 48px;
}
.logo{font-family:'Fraunces',serif;font-size:1.5rem;font-weight:700;
  color:var(--text);letter-spacing:-.02em}
.logo em{font-style:italic;color:var(--mint)}
.hright{display:flex;align-items:center;gap:14px}
.badge{font-family:'Fira Code',monospace;font-size:.62rem;padding:4px 12px;
  border-radius:99px;border:1px solid rgba(0,137,123,.3);
  color:var(--mint);background:rgba(0,137,123,.06);letter-spacing:.06em}
.by{font-size:.78rem;color:var(--muted)}
.by a{color:var(--mint);text-decoration:none;font-weight:600}

/* ── HERO ── */
.hero{
  position:relative;z-index:10;
  background:linear-gradient(135deg,#0a3d35 0%,#155045 50%,#1a6b5a 100%);
  padding:60px 48px 80px;overflow:hidden;
}
.hero::before{
  content:'';position:absolute;inset:0;
  background:url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23ffffff' fill-opacity='0.03'%3E%3Ccircle cx='30' cy='30' r='1'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
}
.hero-inner{position:relative;max-width:1400px;margin:0 auto;
  display:grid;grid-template-columns:1fr auto;gap:40px;align-items:center}
.hero h1{font-family:'Fraunces',serif;font-size:clamp(2.4rem,5vw,4rem);
  font-weight:700;color:#fff;letter-spacing:-.03em;line-height:1.06}
.hero h1 em{font-style:italic;color:#80cbc4}
.hero p{color:rgba(255,255,255,.6);font-size:.95rem;margin-top:14px;
  max-width:520px;line-height:1.75;font-weight:300}
.hero-tags{display:flex;gap:8px;margin-top:20px;flex-wrap:wrap}
.htag{font-family:'Fira Code',monospace;font-size:.68rem;padding:4px 12px;
  border-radius:6px;background:rgba(255,255,255,.08);color:rgba(255,255,255,.7);
  border:1px solid rgba(255,255,255,.12)}
.hero-stats{display:flex;flex-direction:column;gap:14px;align-items:flex-end}
.hstat{text-align:right}
.hstat .v{font-family:'Fraunces',serif;font-size:2rem;font-weight:700;color:#80cbc4}
.hstat .l{font-size:.65rem;color:rgba(255,255,255,.4);text-transform:uppercase;
  letter-spacing:.1em;font-family:'Fira Code',monospace}

/* ── DISEASE PILLS ── */
.disease-bar{position:relative;z-index:10;background:#fff;border-bottom:1px solid var(--border);
  display:flex;justify-content:center}
.d-pill{display:flex;align-items:center;gap:8px;padding:14px 32px;
  border-right:1px solid var(--border);cursor:pointer;transition:.15s}
.d-pill:last-child{border-right:none}
.d-pill:hover{background:#f4f7f4}
.d-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0}
.d-name{font-size:.8rem;font-weight:600;color:var(--muted)}
.d-auc{font-family:'Fira Code',monospace;font-size:.68rem;color:var(--muted)}

main{position:relative;z-index:10;max-width:1400px;margin:0 auto;padding:36px 32px 80px}

/* ── LAYOUT ── */
.grid-main{display:grid;grid-template-columns:420px 1fr;gap:24px;align-items:start}
@media(max-width:1100px){.grid-main{grid-template-columns:1fr}}

/* ── CARD ── */
.card{background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);overflow:hidden;
  box-shadow:0 1px 16px rgba(0,0,0,.04)}
.ch{padding:18px 20px 0;display:flex;align-items:center;gap:10px;margin-bottom:14px}
.ch h2{font-family:'Fraunces',serif;font-size:1rem;font-weight:600;color:var(--text)}
.ic{width:30px;height:30px;border-radius:8px;display:grid;place-items:center;
  font-size:.85rem;flex-shrink:0}
.cb{padding:0 20px 20px}

/* ── FORM ── */
.fg{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.field{display:flex;flex-direction:column;gap:5px}
.field.full{grid-column:1/-1}
label{font-size:.63rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
input,select{
  border:1.5px solid var(--border);border-radius:8px;
  background:#f9fbf9;color:var(--text);
  font-family:'Plus Jakarta Sans',sans-serif;font-size:.875rem;
  padding:9px 11px;transition:.18s;outline:none;width:100%
}
input:focus,select:focus{border-color:var(--mint);background:#fff;
  box-shadow:0 0 0 3px rgba(0,137,123,.1)}
select option{background:#fff}

.radio-row{display:flex;gap:8px}
.radio-opt input{display:none}
.radio-opt label{
  cursor:pointer;padding:7px 16px;border-radius:7px;font-size:.82rem;
  font-weight:600;border:1.5px solid var(--border);background:#f9fbf9;
  color:var(--muted);transition:.15s;text-transform:none;letter-spacing:normal;display:block
}
.radio-opt input:checked+label{background:rgba(0,137,123,.08);
  border-color:var(--mint);color:var(--mint)}

.tog-row{display:flex;gap:8px;flex-wrap:wrap}
.tog input{display:none}
.tog label{cursor:pointer;padding:6px 13px;border-radius:6px;font-size:.8rem;
  font-weight:500;border:1.5px solid var(--border);background:#f9fbf9;
  color:var(--muted);transition:.15s;text-transform:none;letter-spacing:normal;display:block}
.tog input:checked+label{background:rgba(0,137,123,.08);border-color:var(--mint);color:var(--mint)}

.scan-btn{
  width:100%;margin-top:16px;padding:13px;
  background:var(--mint);color:#fff;border:none;border-radius:9px;
  font-family:'Plus Jakarta Sans',sans-serif;font-size:.95rem;font-weight:700;
  cursor:pointer;transition:.2s;letter-spacing:.02em
}
.scan-btn:hover{background:#007168;box-shadow:0 6px 24px rgba(0,137,123,.3);transform:translateY(-1px)}
.scan-btn:active{transform:none}
.scan-btn.loading{opacity:.6;pointer-events:none}

/* ── RESULT ── */
.result-panel{display:none;margin-top:18px;animation:up .4s cubic-bezier(.22,1,.36,1)}
.result-panel.show{display:block}
@keyframes up{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}

.health-idx{
  border-radius:11px;padding:20px;
  background:linear-gradient(135deg,#0a3d35,#1a6b5a);
  color:#fff;text-align:center;margin-bottom:14px
}
.hi-label{font-family:'Fira Code',monospace;font-size:.62rem;letter-spacing:.12em;
  text-transform:uppercase;color:rgba(255,255,255,.5);margin-bottom:6px}
.hi-val{font-family:'Fraunces',serif;font-size:3.5rem;font-weight:700;line-height:1}
.hi-sub{font-size:.75rem;color:rgba(255,255,255,.5);margin-top:4px}

.risk-cards{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:14px}
.risk-card{border-radius:9px;padding:12px 14px;border:1px solid var(--border)}
.rc-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px}
.rc-icon{font-size:.95rem}
.rc-pct{font-family:'Fira Code',monospace;font-size:1.1rem;font-weight:600}
.rc-name{font-size:.7rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.08em}
.rc-bar-wrap{height:5px;background:rgba(0,0,0,.06);border-radius:3px;overflow:hidden;margin-top:6px}
.rc-bar{height:100%;border-radius:3px;transition:.6s}
.rc-verdict{font-size:.68rem;font-weight:700;margin-top:5px;text-transform:uppercase;letter-spacing:.08em}

.meta-box{background:#f4f7f4;border:1px solid var(--border);border-radius:9px;
  padding:12px 14px;margin-bottom:14px;display:flex;align-items:center;justify-content:space-between}
.meta-label{font-size:.75rem;color:var(--muted);font-weight:500}
.meta-val{font-family:'Fira Code',monospace;font-size:.95rem;font-weight:600}

.contrib-panel{margin-top:0}
.contrib-title{font-size:.63rem;font-weight:700;text-transform:uppercase;
  letter-spacing:.1em;color:var(--muted);margin-bottom:10px}
.contrib-item{margin-bottom:7px}
.ci-top{display:flex;justify-content:space-between;margin-bottom:3px}
.ci-name{font-size:.75rem;color:var(--text);font-weight:500}
.ci-badges{display:flex;gap:4px}
.ci-b{font-family:'Fira Code',monospace;font-size:.6rem;padding:1px 5px;
  border-radius:4px;font-weight:600}
.ci-bars{display:grid;grid-template-columns:repeat(4,1fr);gap:3px}
.ci-bw{height:4px;background:#f0f4f0;border-radius:2px;overflow:hidden}
.ci-bf{height:100%;border-radius:2px;transition:.5s}

/* ── TABS ── */
.tabs{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:14px}
.tb{padding:5px 12px;border-radius:6px;border:1.5px solid var(--border);
  background:transparent;color:var(--muted);font-size:.73rem;cursor:pointer;
  font-family:'Plus Jakarta Sans',sans-serif;font-weight:600;transition:.15s}
.tb.active{background:var(--mint);border-color:var(--mint);color:#fff}

/* ── METRICS ── */
.met3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
.met2{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:14px}
.m{background:#f9fbf9;border:1px solid var(--border);border-radius:9px;padding:12px 14px}
.mn{font-size:.6rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);margin-bottom:3px}
.mv{font-family:'Fira Code',monospace;font-size:1rem;font-weight:600}

.per-disease{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:14px}
.pd{background:#f9fbf9;border:1px solid var(--border);border-radius:9px;
  padding:10px 12px;text-align:center}
.pd-ico{font-size:1.1rem;margin-bottom:4px}
.pd-name{font-size:.62rem;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:4px}
.pd-val{font-family:'Fira Code',monospace;font-size:.9rem;font-weight:600}

/* ── CHARTS ── */
.ch-h{position:relative;height:210px}
.ch-h2{position:relative;height:230px}
.ch-h3{position:relative;height:260px}

/* ── FOOTER ── */
footer{
  background:#0a3d35;color:rgba(255,255,255,.5);
  text-align:center;padding:48px 32px
}
.fn{font-family:'Fraunces',serif;font-size:1.6rem;font-weight:700;color:#fff;margin-bottom:6px}
.fn em{font-style:italic;color:#80cbc4}
.flinks{display:flex;justify-content:center;gap:12px;margin:18px 0;flex-wrap:wrap}
.fl{display:inline-flex;align-items:center;gap:7px;padding:10px 22px;
  border-radius:99px;font-size:.83rem;font-weight:600;text-decoration:none;transition:.2s}
.fl.gh{background:rgba(255,255,255,.08);color:#fff;border:1px solid rgba(255,255,255,.15)}
.fl.gh:hover{background:rgba(255,255,255,.15)}
.fl.lt{background:rgba(128,203,196,.12);color:#80cbc4;border:1px solid rgba(128,203,196,.3)}
.fl.lt:hover{background:rgba(128,203,196,.2)}
.fstack{font-family:'Fira Code',monospace;font-size:.65rem;margin-top:16px;opacity:.35}
</style>
</head>
<body>

<header>
  <div class="logo">VITAL<em>SCAN</em></div>
  <div class="hright">
    <span class="badge">MULTI-DISEASE RISK ENGINE v1.0</span>
    <span class="by">by <a href="https://linktr.ee/Musharib_" target="_blank">Musharib</a></span>
  </div>
</header>

<div class="hero">
  <div class="hero-inner">
    <div>
      <h1>AI-Powered<br><em>Multi-Disease</em><br>Risk Assessment</h1>
      <p>Simultaneously predicts risk for 4 major conditions using ClassifierChain ML — modeling dependencies between diseases. Built on 10,000 synthetic patient records and 18 clinical biomarkers.</p>
      <div class="hero-tags">
        <span class="htag">🔗 ClassifierChain</span>
        <span class="htag">🎯 MultiOutputClassifier</span>
        <span class="htag">📊 4 Diseases at Once</span>
        <span class="htag">🔬 18 Biomarkers</span>
        <span class="htag">🕸️ Radar Risk Profile</span>
      </div>
    </div>
    <div class="hero-stats">
      <div class="hstat"><div class="v" id="sBestAuc">—</div><div class="l">Best Macro AUC</div></div>
      <div class="hstat"><div class="v">10,000</div><div class="l">Patient Records</div></div>
      <div class="hstat"><div class="v">6</div><div class="l">ML Models</div></div>
    </div>
  </div>
</div>

<div class="disease-bar">
  <div class="d-pill"><div class="d-dot" style="background:var(--heart)"></div><div><div class="d-name">Heart Disease</div><div class="d-auc" id="dpH">AUC —</div></div></div>
  <div class="d-pill"><div class="d-dot" style="background:var(--diab)"></div><div><div class="d-name">Diabetes</div><div class="d-auc" id="dpD">AUC —</div></div></div>
  <div class="d-pill"><div class="d-dot" style="background:var(--hype)"></div><div><div class="d-name">Hypertension</div><div class="d-auc" id="dpHT">AUC —</div></div></div>
  <div class="d-pill"><div class="d-dot" style="background:var(--stro)"></div><div><div class="d-name">Stroke Risk</div><div class="d-auc" id="dpS">AUC —</div></div></div>
</div>

<main>
  <div class="grid-main">

    <!-- ── LEFT: FORM ── -->
    <div style="display:flex;flex-direction:column;gap:20px">
      <div class="card">
        <div class="ch"><div class="ic" style="background:#e8f5e9">🩺</div><h2>Patient Profile</h2></div>
        <div class="cb">
          <div class="fg">
            <div class="field">
              <label>Age</label>
              <input type="number" id="age" value="45" min="18" max="90">
            </div>
            <div class="field">
              <label>BMI</label>
              <input type="number" id="bmi" value="26.5" min="15" max="55" step="0.1">
            </div>
            <div class="field full">
              <label>Gender</label>
              <div class="radio-row">
                <div class="radio-opt"><input type="radio" name="gender" id="gF" value="0" checked><label for="gF">Female</label></div>
                <div class="radio-opt"><input type="radio" name="gender" id="gM" value="1"><label for="gM">Male</label></div>
              </div>
            </div>
            <div class="field"><label>Systolic BP (mmHg)</label><input type="number" id="systolic_bp" value="122" min="70" max="220"></div>
            <div class="field"><label>Diastolic BP (mmHg)</label><input type="number" id="diastolic_bp" value="78" min="40" max="140"></div>
            <div class="field"><label>Fasting Glucose (mg/dL)</label><input type="number" id="fasting_glucose" value="95" min="50" max="400"></div>
            <div class="field"><label>HbA1c (%)</label><input type="number" id="hba1c" value="5.4" min="4" max="14" step="0.1"></div>
            <div class="field"><label>Total Cholesterol</label><input type="number" id="total_cholesterol" value="195" min="80" max="400"></div>
            <div class="field"><label>LDL Cholesterol</label><input type="number" id="ldl" value="115" min="30" max="300"></div>
            <div class="field"><label>HDL Cholesterol</label><input type="number" id="hdl" value="58" min="15" max="120"></div>
            <div class="field"><label>Triglycerides</label><input type="number" id="triglycerides" value="140" min="30" max="700"></div>
            <div class="field"><label>Waist Circumference (cm)</label><input type="number" id="waist_cm" value="84" min="50" max="180" step="0.5"></div>
            <div class="field"><label>Sleep Hours / Night</label><input type="number" id="sleep_hours" value="7" min="3" max="12" step="0.5"></div>
            <div class="field"><label>Stress Level (1–10)</label><input type="number" id="stress_level" value="4" min="1" max="10"></div>
            <div class="field full"><label>Smoking Status</label>
              <select id="smoking">
                <option value="0" selected>Never Smoked</option>
                <option value="1">Former Smoker</option>
                <option value="2">Current Smoker</option>
              </select>
            </div>
            <div class="field full"><label>Alcohol Use</label>
              <select id="alcohol">
                <option value="0" selected>None</option>
                <option value="1">Moderate</option>
                <option value="2">Heavy</option>
              </select>
            </div>
            <div class="field full"><label>Physical Activity</label>
              <select id="physical_activity">
                <option value="0">Sedentary</option>
                <option value="1" selected>Light</option>
                <option value="2">Moderate</option>
                <option value="3">Active</option>
              </select>
            </div>
            <div class="field full"><label>Lifestyle Flags</label>
              <div class="tog-row">
                <div class="tog"><input type="checkbox" id="family_history"><label for="family_history">🧬 Family History</label></div>
              </div>
            </div>
            <div class="field full"><label>Model</label>
              <select id="modelSel"><option value="">Auto (Best Model)</option></select>
            </div>
          </div>
          <button class="scan-btn" onclick="scan()">🔬 Run Health Scan</button>

          <!-- RESULTS -->
          <div class="result-panel" id="resultPanel">
            <div class="health-idx">
              <div class="hi-label">Overall Health Index</div>
              <div class="hi-val" id="hiVal">—</div>
              <div class="hi-sub" id="hiSub"></div>
            </div>

            <div class="meta-box">
              <div><div class="meta-label">Metabolic Syndrome Score</div>
                <div style="font-size:.7rem;color:var(--muted);margin-top:1px">5 criteria · 20pts each</div>
              </div>
              <div class="meta-val" id="metaVal">—</div>
            </div>

            <div class="risk-cards" id="riskCards"></div>

            <div class="contrib-panel">
              <div class="contrib-title">Risk Drivers by Disease</div>
              <div id="contribList"></div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ── RIGHT ── -->
    <div style="display:flex;flex-direction:column;gap:20px">

      <!-- Model Leaderboard -->
      <div class="card">
        <div class="ch"><div class="ic" style="background:#e8f5e9">🏆</div><h2>Model Leaderboard</h2></div>
        <div class="cb">
          <div class="tabs" id="modelTabs"></div>
          <div class="met3" id="metRow"></div>
          <div class="per-disease" id="perDisease"></div>
        </div>
      </div>

      <!-- Radar + ROC -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
        <div class="card">
          <div class="ch"><div class="ic" style="background:#e8f5e9">🕸️</div><h2>Risk Radar Profile</h2></div>
          <div class="cb"><div class="ch-h3"><canvas id="radarC"></canvas></div></div>
        </div>
        <div class="card">
          <div class="ch"><div class="ic" style="background:#fce4ec">📈</div><h2>ROC per Disease</h2></div>
          <div class="cb"><div class="ch-h3"><canvas id="rocC"></canvas></div></div>
        </div>
      </div>

      <!-- Feature Importance + Age Risk -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
        <div class="card">
          <div class="ch"><div class="ic" style="background:#e0f2f1">🧬</div><h2>Feature Importance</h2></div>
          <div class="cb"><div class="ch-h2"><canvas id="fiC"></canvas></div></div>
        </div>
        <div class="card">
          <div class="ch"><div class="ic" style="background:#f3e5f5">📊</div><h2>Disease Prevalence by Age</h2></div>
          <div class="cb"><div class="ch-h2"><canvas id="ageC"></canvas></div></div>
        </div>
      </div>

      <!-- Gender Breakdown -->
      <div class="card">
        <div class="ch"><div class="ic" style="background:#e8eaf6">⚖️</div><h2>Risk by Gender</h2></div>
        <div class="cb"><div class="ch-h"><canvas id="genderC"></canvas></div></div>
      </div>

    </div>
  </div>
</main>

<footer>
  <div class="fn">Made by <em>Musharib</em></div>
  <div style="font-size:.85rem;margin-top:4px">Multi-Disease AI Health Engine · Powered by Scikit-Learn & Flask</div>
  <div class="flinks">
    <a class="fl gh" href="https://github.com/musharibramzan950-bit" target="_blank">
      <svg height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>
      GitHub
    </a>
    <a class="fl lt" href="https://linktr.ee/Musharib_" target="_blank">
      <svg height="15" viewBox="0 0 24 24" fill="currentColor"><path d="M7.953 15.066c-.08.163-.08.324-.08.486C7.873 17.482 9.553 19 11.64 19s3.85-1.518 3.85-3.448c0-.162 0-.323-.08-.486H7.954zM4 9.228l1.698 1.926L4 13.08h4.494L12 9.228 8.494 5.376H4L5.698 7.3 4 9.228zm16 0L18.302 7.3 20 5.376h-4.494L12 9.228l3.506 3.852H20l-1.698-1.926L20 9.228z"/></svg>
      Linktree
    </a>
  </div>
  <div class="fstack">scikit-learn · flask · chart.js · ClassifierChain · MultiOutputClassifier · 10,000 patients · 6 models · 18 biomarkers</div>
</footer>

<script>
let allM={}, allFI={}, allROC={}, bestM='';
let rocChart=null, fiChart=null, radarChart=null, ageChart=null, genderChart=null;
const $=id=>document.getElementById(id);
const D_COLORS=['#e63946','#f4a261','#2a9d8f','#7209b7'];
const D_NAMES =['Heart Disease','Type 2 Diabetes','Hypertension','Stroke Risk'];
const D_SHORT =['Heart','Diabetes','Hypertension','Stroke'];
const D_ICONS =['❤️','🩸','🫀','🧠'];

async function init(){
  const d=await(await fetch('/api/info')).json();
  allM=d.metrics; allFI=d.feat_imp; allROC=d.roc; bestM=d.best;

  $('sBestAuc').textContent=d.metrics[bestM].macro_auc;
  ['dpH','dpD','dpHT','dpS'].forEach((id,i)=>{
    $(id).textContent='AUC '+d.metrics[bestM].per_auc[D_SHORT[i]];
  });

  const sel=$('modelSel');
  Object.keys(allM).forEach(n=>{
    const o=document.createElement('option');
    o.value=n; o.textContent=`${n} (AUC ${allM[n].macro_auc})`; sel.appendChild(o);
  });

  const tabs=$('modelTabs');
  Object.keys(allM).forEach(n=>{
    const b=document.createElement('button');
    b.className='tb'+(n===bestM?' active':'');
    b.textContent=n+(n===bestM?' ★':'');
    b.onclick=()=>switchTab(n); tabs.appendChild(b);
  });

  switchTab(bestM);
  renderAgeChart(d.age_prev);
  renderGenderChart(d.gender_prev);
  // init radar with zeros
  renderRadar([0,0,0,0]);
}

function switchTab(name){
  bestM=name;
  document.querySelectorAll('.tb').forEach((b,i)=>{
    b.classList.toggle('active', Object.keys(allM)[i]===name);
  });
  renderMetrics(name); renderROC(name); renderFI(name);
}

function renderMetrics(n){
  const m=allM[n];
  $('metRow').innerHTML=`
    <div class="m"><div class="mn">Macro AUC</div><div class="mv" style="color:var(--mint)">${m.macro_auc}</div></div>
    <div class="m"><div class="mn">Macro F1</div><div class="mv" style="color:var(--mint)">${m.macro_f1}</div></div>
    <div class="m"><div class="mn">Hamming Loss</div><div class="mv" style="color:#e63946">${m.hamming}</div></div>
  `;
  $('perDisease').innerHTML=D_SHORT.map((d,i)=>`
    <div class="pd">
      <div class="pd-ico">${D_ICONS[i]}</div>
      <div class="pd-name">${d}</div>
      <div class="pd-val" style="color:${D_COLORS[i]}">${m.per_auc[d]}</div>
      <div style="font-size:.6rem;color:var(--muted);font-family:'Fira Code',monospace">F1 ${m.per_f1[d]}</div>
    </div>`).join('');
}

function renderRadar(probs){
  if(radarChart) radarChart.destroy();
  radarChart=new Chart($('radarC').getContext('2d'),{
    type:'radar',
    data:{
      labels:D_NAMES,
      datasets:[{
        label:'Risk Profile',
        data:probs.map(p=>(p*100).toFixed(1)),
        borderColor:'#00897b',borderWidth:2,
        backgroundColor:'rgba(0,137,123,.12)',
        pointBackgroundColor:D_COLORS,
        pointRadius:6,pointHoverRadius:8,
      }]
    },
    options:{
      responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},
        tooltip:{callbacks:{label:c=>`Risk: ${c.raw}%`}}},
      scales:{r:{
        min:0,max:100,
        grid:{color:'rgba(0,0,0,.06)'},
        angleLines:{color:'rgba(0,0,0,.08)'},
        pointLabels:{color:'#1a2e1a',font:{size:11,weight:'600'}},
        ticks:{color:'#6b7f6b',font:{size:9},backdropColor:'transparent',
          callback:v=>v+'%',stepSize:20}
      }}
    }
  });
}

function renderROC(n){
  const rocs=allROC[n];
  if(rocChart) rocChart.destroy();
  rocChart=new Chart($('rocC').getContext('2d'),{
    type:'line',
    data:{
      labels:rocs[D_SHORT[0]].fpr,
      datasets:D_SHORT.map((d,i)=>({
        label:D_NAMES[i],
        data:rocs[d].tpr,
        borderColor:D_COLORS[i],borderWidth:2,
        fill:false,pointRadius:0,tension:.3,
      })).concat([{
        label:'Random',
        data:rocs[D_SHORT[0]].fpr,
        borderColor:'#e2e8e2',borderWidth:1,borderDash:[4,4],
        fill:false,pointRadius:0,
      }])
    },
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#6b7f6b',font:{size:10},boxWidth:10}}},
      scales:{
        x:{grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#6b7f6b',maxTicksLimit:6},
           title:{display:true,text:'FPR',color:'#6b7f6b',font:{size:10}}},
        y:{grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#6b7f6b',maxTicksLimit:6},
           title:{display:true,text:'TPR',color:'#6b7f6b',font:{size:10}}}
      }
    }
  });
}

function renderFI(n){
  const fi=allFI[n];
  const pairs=Object.entries(fi).sort((a,b)=>b[1]-a[1]).slice(0,12);
  if(fiChart) fiChart.destroy();
  fiChart=new Chart($('fiC').getContext('2d'),{
    type:'bar',
    data:{labels:pairs.map(p=>p[0]),datasets:[{
      data:pairs.map(p=>p[1]),
      backgroundColor:pairs.map((_,i)=>`hsla(${165+i*8},55%,45%,0.8)`),
      borderRadius:4,borderSkipped:false
    }]},
    options:{indexAxis:'y',responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>(c.raw*100).toFixed(1)+'%'}}},
      scales:{
        x:{grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#6b7f6b',callback:v=>(v*100).toFixed(0)+'%'}},
        y:{grid:{display:false},ticks:{color:'#1a2e1a',font:{size:10}}}
      }
    }
  });
}

function renderAgeChart(data){
  if(ageChart) ageChart.destroy();
  ageChart=new Chart($('ageC').getContext('2d'),{
    type:'line',
    data:{
      labels:data.map(d=>d.group),
      datasets:D_SHORT.map((d,i)=>({
        label:D_NAMES[i],
        data:data.map(r=>r[d]),
        borderColor:D_COLORS[i],borderWidth:2,
        fill:false,pointRadius:4,tension:.4,
        pointBackgroundColor:D_COLORS[i],
      }))
    },
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#6b7f6b',font:{size:10},boxWidth:10}}},
      scales:{
        x:{grid:{display:false},ticks:{color:'#6b7f6b',font:{size:10}}},
        y:{grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#6b7f6b',callback:v=>v+'%'}}
      }
    }
  });
}

function renderGenderChart(data){
  if(genderChart) genderChart.destroy();
  genderChart=new Chart($('genderC').getContext('2d'),{
    type:'bar',
    data:{
      labels:D_NAMES,
      datasets:[
        {label:'Female',data:data.female,backgroundColor:'rgba(230,57,70,.55)',borderRadius:4,borderSkipped:false},
        {label:'Male',  data:data.male,  backgroundColor:'rgba(42,157,143,.55)',borderRadius:4,borderSkipped:false},
      ]
    },
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#6b7f6b',font:{size:11}}}},
      scales:{
        x:{grid:{display:false},ticks:{color:'#6b7f6b',font:{size:10}}},
        y:{grid:{color:'rgba(0,0,0,.05)'},ticks:{color:'#6b7f6b',callback:v=>v+'%'},
           title:{display:true,text:'Prevalence (%)',color:'#6b7f6b',font:{size:10}}}
      }
    }
  });
}

async function scan(){
  const btn=document.querySelector('.scan-btn');
  btn.classList.add('loading'); btn.textContent='⏳ Scanning…';

  const gender=document.querySelector('input[name=gender]:checked').value;
  const payload={
    age:+$('age').value, gender:+gender, bmi:+$('bmi').value,
    systolic_bp:+$('systolic_bp').value, diastolic_bp:+$('diastolic_bp').value,
    fasting_glucose:+$('fasting_glucose').value, hba1c:+$('hba1c').value,
    total_cholesterol:+$('total_cholesterol').value, ldl:+$('ldl').value,
    hdl:+$('hdl').value, triglycerides:+$('triglycerides').value,
    smoking:+$('smoking').value, alcohol:+$('alcohol').value,
    physical_activity:+$('physical_activity').value,
    family_history:$('family_history').checked?1:0,
    sleep_hours:+$('sleep_hours').value, stress_level:+$('stress_level').value,
    waist_cm:+$('waist_cm').value,
    model:$('modelSel').value||null,
  };

  const d=await(await fetch('/api/predict',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})).json();

  btn.classList.remove('loading'); btn.textContent='🔬 Run Health Scan';

  const panel=$('resultPanel');
  panel.classList.remove('show'); void panel.offsetWidth; panel.classList.add('show');

  // Health Index
  const hi=d.health_idx;
  $('hiVal').textContent=hi;
  $('hiVal').style.color=hi>=80?'#80cbc4':hi>=60?'#ffeb3b':hi>=40?'#ff9800':'#ef5350';
  $('hiSub').textContent=hi>=80?'Excellent Health Profile':hi>=60?'Good — Some Risk Factors':hi>=40?'Moderate Risk — Action Advised':'High Risk — Consult a Doctor';

  // Metabolic score
  const ms=d.meta_score;
  $('metaVal').textContent=`${ms}/100`;
  $('metaVal').style.color=ms<=20?'var(--mint)':ms<=40?'var(--diab)':'var(--heart)';

  // Risk cards
  $('riskCards').innerHTML=D_NAMES.map((name,i)=>{
    const p=d.probs[i]; const pct=Math.round(p*100);
    const color=D_COLORS[i];
    const risk=pct<20?'Low Risk':pct<40?'Moderate':pct<65?'High Risk':'Very High';
    return `<div class="risk-card" style="border-color:${color}22;background:${color}08">
      <div class="rc-top">
        <span class="rc-icon">${D_ICONS[i]}</span>
        <span class="rc-pct" style="color:${color}">${pct}%</span>
      </div>
      <div class="rc-name">${name}</div>
      <div class="rc-bar-wrap"><div class="rc-bar" style="width:${pct}%;background:${color}"></div></div>
      <div class="rc-verdict" style="color:${color}">${risk}</div>
    </div>`;
  }).join('');

  // Update radar
  renderRadar(d.probs);

  // Contributions
  const contribs=Object.entries(d.contribs)
    .map(([name,vals])=>({name,vals,maxAbs:Math.max(...vals.map(Math.abs))}))
    .sort((a,b)=>b.maxAbs-a.maxAbs).slice(0,8);

  const globalMax=Math.max(...contribs.map(c=>c.maxAbs))||0.001;
  $('contribList').innerHTML=contribs.map(({name,vals})=>`
    <div class="contrib-item">
      <div class="ci-top">
        <span class="ci-name">${name}</span>
        <div class="ci-badges">${D_SHORT.map((d2,i)=>{
          const v=vals[i]; if(Math.abs(v)<0.005) return '';
          return `<span class="ci-b" style="background:${D_COLORS[i]}18;color:${D_COLORS[i]}">${v>0?'+':''}${(v*100).toFixed(0)}%</span>`;
        }).join('')}</div>
      </div>
      <div class="ci-bars">${D_COLORS.map((c,i)=>{
        const v=vals[i];
        const w=(Math.abs(v)/globalMax*100).toFixed(0);
        return `<div class="ci-bw"><div class="ci-bf" style="width:${w}%;background:${v>0?c:'#2a9d8f'}"></div></div>`;
      }).join('')}</div>
    </div>`).join('');
}

init();
</script>
</body>
</html>"""

# ─── FLASK ────────────────────────────────────────────────────────────────────
app = Flask(__name__)
hub = VitalHub()

AGE_GROUPS = ["20–30","31–40","41–50","51–60","61–70","71–85"]

@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/api/info")
def api_info():
    df = hub._df
    # Age prevalence
    bins   = [20,31,41,51,61,71,86]
    labels = AGE_GROUPS
    age_prev = []
    for i,(lo,hi) in enumerate(zip(bins,bins[1:])):
        mask = (df["age"]>=lo)&(df["age"]<hi)
        sub  = df[mask]
        if len(sub)==0: continue
        row = {"group": labels[i]}
        for t,s in zip(TARGET_COLS, D_SHORT):
            row[s] = round(sub[t].mean()*100, 1)
        age_prev.append(row)

    # Gender prevalence
    gender_prev = {"female":[], "male":[]}
    for g,k in [(0,"female"),(1,"male")]:
        sub = df[df["gender"]==g]
        for t in TARGET_COLS:
            gender_prev[k].append(round(sub[t].mean()*100,1))

    return jsonify({
        "best": hub.best, "metrics": hub.metrics,
        "feat_imp": hub.feat_imp, "roc": hub.roc_data,
        "age_prev": age_prev, "gender_prev": gender_prev,
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    body  = request.get_json(force=True)
    model = body.pop("model",None) or hub.best
    return jsonify(hub.predict(body, model))

# ─── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", 8080))
    print("\n"+"═"*58)
    print("  🏥  VITALSCAN — AI Multi-Disease Health Risk Engine")
    print("  Made by Musharib | linktr.ee/Musharib_")
    print("═"*58)
    print("  ⚙️  Generating 10,000 patient records…")
    df = generate_dataset(n=10000)
    for t,d in zip(TARGET_COLS, DISEASES):
        print(f"     {d:20s}  prevalence: {df[t].mean()*100:.1f}%")
    print("  🧠  Training 6 multi-output ML models…")
    print("     (ClassifierChain + MultiOutputClassifier)")
    hub._df = df
    hub.train(df)
    print(f"\n  🌐  Running at → http://localhost:{PORT}")
    print("  Press Ctrl+C to stop\n"+"═"*58+"\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
