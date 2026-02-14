# inference.py
"""
Run inference and produce submission CSV.

Usage:
python inference.py --input_csv data/synthetic_test.csv --model_dir outputs/baseline --output_csv submission.csv --mode ensemble
--mode choices: 'final' (use final_model.joblib), 'ensemble' (average fold models)
"""

import os, json, argparse
import numpy as np
import pandas as pd
from joblib import load

def load_meta(model_dir):
    meta = json.load(open(os.path.join(model_dir, "meta.json")))
    return meta

def predict_with_model_obj(model_obj, texts):
    # model_obj keys: 'clf', 'vec_spec', 'vec_obj'
    vec_spec = model_obj["vec_spec"]; vec_obj = model_obj["vec_obj"]; clf = model_obj["clf"]
    # transform texts
    from train import transform_texts
    X = transform_texts(vec_spec, vec_obj, texts)
    if hasattr(clf, "predict_proba"):
        try:
            probs = np.vstack([est.predict_proba(X)[:,1] for est in clf.estimators_]).T
        except Exception:
            probs = clf.predict_proba(X)
    else:
        probs = clf.predict_proba(X)
    return probs

def load_fold_models(fold_paths):
    models = [load(p) for p in fold_paths]
    return models

def ensemble_predict(models, texts):
    # Average probabilities from each fold model
    probs_list = []
    for m in models:
        p = predict_with_model_obj(m, texts)
        probs_list.append(p)
    return np.mean(probs_list, axis=0)

def apply_calibrators_if_any(calib_path, probs):
    if calib_path is None: return probs
    calibrators = load(calib_path)
    probs_cal = probs.copy()
    for i, cal in enumerate(calibrators):
        if cal is None: continue
        probs_cal[:, i] = cal.predict(probs[:, i])
    return probs_cal

def main(args):
    df = pd.read_csv(args.input_csv)
    texts = df[args.text_col].astype(str).tolist()
    meta = load_meta(args.model_dir)
    thresholds = np.array(meta["thresholds"])
    # load calibrators if exist
    calib_path = meta.get("calibrators", None)

    if args.mode == "final":
        model_obj = load(meta["final_model"])
        probs = predict_with_model_obj(model_obj, texts)
    else:
        models = load_fold_models(meta["fold_models"])
        probs = ensemble_predict(models, texts)

    # apply calibration if requested / available
    if args.apply_calibration and calib_path is not None:
        probs = apply_calibrators_if_any(calib_path, probs)

    preds = (probs >= thresholds).astype(int)
    # attach to df
    labels = meta["labels"]
    for i, lab in enumerate(labels):
        df[f"{lab}_prob"] = probs[:, i]
        df[lab] = preds[:, i]
    df.to_csv(args.output_csv, index=False)
    print("Wrote predictions to", args.output_csv)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_csv", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--model_dir", default="outputs/baseline")
    p.add_argument("--text_col", default="text")
    p.add_argument("--mode", default="ensemble", choices=["final","ensemble"])
    p.add_argument("--apply_calibration", action="store_true")
    args = p.parse_args()
    main(args)
