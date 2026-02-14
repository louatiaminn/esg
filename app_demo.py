# app_demo.py
"""
Streamlit demo: loads final model and shows predictions for pasted text.
Make sure to only paste synthetic or anonymized text.
Run: streamlit run app_demo.py
"""

import streamlit as st
import json, numpy as np
from joblib import load
from train import transform_texts

MODEL_DIR = "outputs/baseline"
META_PATH = f"{MODEL_DIR}/meta.json"

st.set_page_config(page_title="ESG Radar Demo", layout="wide")
st.title("ESG Radar — Demo (Synthetic examples only)")

# Load meta + final model
meta = json.load(open(META_PATH))
labels = meta["labels"]
final = load(meta["final_model"])
thresholds = np.array(meta["thresholds"])

st.sidebar.markdown("**Demo rules**")
st.sidebar.write("Do NOT paste actual contest dataset rows here. Use synthetic examples only.")

text = st.text_area("Paste a short text (synthetic)", height=200)
if st.button("Analyze") and text.strip():
    texts = [text]
    probs = None
    try:
        probs = np.vstack([est.predict_proba(transform_texts(final["vec_spec"], final["vec_obj"], texts))[:,1] for est in final["clf"].estimators_]).T
    except Exception:
        probs = final["clf"].predict_proba(transform_texts(final["vec_spec"], final["vec_obj"], texts))
    probs = probs[0]
    st.subheader("Probabilities and labels")
    cols = st.columns(len(labels))
    for i, c in enumerate(cols):
        c.metric(labels[i], f"{probs[i]:.3f}", delta="1" if probs[i] >= thresholds[i] else "0")
    st.write("Thresholds:", {labels[i]: float(thresholds[i]) for i in range(len(labels))})
    # Simple ESG Risk Score
    E,S,G,nonESG = probs
    risk = ((E+S+G)/3.0) * 100.0 * (1.0 - nonESG)
    st.subheader(f"ESG Risk Intensity Score: {risk:.1f}/100")
    # Greenwash heuristic
    VAGUE = ["sustainable","green","eco-friendly","responsible","net-zero","carbon neutral"]
    vague_count = sum(text.lower().count(w) for w in VAGUE)
    measurable = any(tok in text.lower() for tok in ["%", "ton", "by 20", "by 202", "target"])
    gw_flag = (vague_count >= 2) and (not measurable)
    st.write("Greenwash heuristic:", {"vague_count":vague_count, "measurable": measurable, "flag": gw_flag})
