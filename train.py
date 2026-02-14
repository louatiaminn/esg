import os, json, argparse
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import dump, load

# sklearn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from sklearn.isotonic import IsotonicRegression

# try to use MultilabelStratifiedKFold for better multilabel splits
try:
    from iterative_stratification import iterative_train_test_split
    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
    HAVE_MLSKF = True
except Exception:
    HAVE_MLSKF = False

RANDOM_STATE = 42

def read_csv(train_csv, text_col, label_cols):
    df = pd.read_csv(train_csv)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_cols].astype(int).values
    return df, texts, labels

def build_vectorizers(use_char=False, max_features=100000, min_df=2, max_df=0.95, word_ngram=(1,2), char_ngram=(3,5)):
    if use_char:
        word_vec = TfidfVectorizer(analyzer="word", ngram_range=word_ngram, max_features=max_features, min_df=min_df, max_df=max_df, sublinear_tf=True)
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram, max_features=max_features//3, min_df=min_df, max_df=max_df, sublinear_tf=True)
        return ("combined", (word_vec, char_vec))
    else:
        vec = TfidfVectorizer(analyzer="word", ngram_range=word_ngram, max_features=max_features, min_df=min_df, max_df=max_df, sublinear_tf=True)
        return ("single", vec)

def fit_vectorizer_on_texts(vec_spec, texts):
    if vec_spec[0] == "single":
        vec = vec_spec[1]
        X = vec.fit_transform(texts)
        return vec, X
    else:
        word_vec, char_vec = vec_spec[1]
        Xw = word_vec.fit_transform(texts)
        Xc = char_vec.fit_transform(texts)
        from scipy.sparse import hstack
        X = hstack([Xw, Xc], format="csr")
        return (word_vec, char_vec), X

def transform_texts(vec_spec, vec_obj, texts):
    if vec_spec[0] == "single":
        return vec_obj.transform(texts)
    else:
        word_vec, char_vec = vec_obj
        Xw = word_vec.transform(texts)
        Xc = char_vec.transform(texts)
        from scipy.sparse import hstack
        return hstack([Xw, Xc], format="csr")

def build_classifier(C=1.0, solver="saga", max_iter=2000, n_jobs=1):
    base = LogisticRegression(C=C, solver=solver, max_iter=max_iter, class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE)
    ovr = OneVsRestClassifier(base, n_jobs=n_jobs)
    return ovr

def tune_thresholds_oof(probs_oof, y_true, resolution=100):
    n_labels = y_true.shape[1]
    thresholds = []
    for i in range(n_labels):
        best_thr, best_f1 = 0.5, 0.0
        p = probs_oof[:, i]
        y = y_true[:, i]
        for thr in np.linspace(0.01, 0.99, resolution):
            preds = (p >= thr).astype(int)
            f1 = f1_score(y, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
        thresholds.append(float(best_thr))
    return thresholds

def macro_f1_from_preds(preds, y_true):
    f1s = []
    for i in range(y_true.shape[1]):
        f1s.append(f1_score(y_true[:,i], preds[:,i], zero_division=0))
    return float(np.mean(f1s)), f1s

def fit_isotonic_calibrators(oof_probs, y_true):
    # Fit one isotonic regressor per label using OOF probs.
    calibrators = []
    for i in range(y_true.shape[1]):
        p = oof_probs[:, i]
        y = y_true[:, i]
        # Isotonic needs at least 2 distinct values; guard against degenerate cases
        if len(np.unique(p)) <= 2:
            calibrators.append(None)
            continue
        iso = IsotonicRegression(out_of_bounds='clip')
        try:
            iso.fit(p, y)
            calibrators.append(iso)
        except Exception:
            calibrators.append(None)
    return calibrators

def apply_calibrators(calibrators, probs):
    calibrated = probs.copy()
    for i, cal in enumerate(calibrators):
        if cal is None: continue
        calibrated[:, i] = cal.predict(probs[:, i])
    return calibrated

def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    df, texts, labels = read_csv(args.train_csv, args.text_col, args.labels)
    n = len(texts)
    n_labels = labels.shape[1]
    print(f"Loaded {n} samples, {n_labels} labels")

    # Prepare CV splitter
    if HAVE_MLSKF:
        print("Using MultilabelStratifiedKFold for CV")
        mskf = MultilabelStratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
        splits = list(mskf.split(np.zeros(n), labels))
    else:
        print("Multilabel stratified KFold not available; using simple KFold")
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=RANDOM_STATE)
        splits = list(kf.split(texts))

    # storage for OOF preds
    oof_probs = np.zeros((n, n_labels), dtype=float)
    fold_paths = []
    val_scores = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        print(f"\n=== Fold {fold}/{args.n_splits} ===")
        train_texts = [texts[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]
        y_tr = labels[train_idx]
        y_val = labels[val_idx]

        # build vectorizer and fit on train_texts ONLY (no-leak)
        vec_spec = build_vectorizers(use_char=args.use_char_ngram, max_features=args.max_features,
                                     min_df=args.min_df, max_df=args.max_df,
                                     word_ngram=(1, args.word_ngram_max),
                                     char_ngram=(args.char_ngram_min, args.char_ngram_max))
        vec_obj, X_tr = fit_vectorizer_on_texts(vec_spec, train_texts)
        X_val = transform_texts(vec_spec, vec_obj, val_texts)

        # train classifier on X_tr
        clf = build_classifier(C=args.C, solver=args.solver, max_iter=args.max_iter, n_jobs=args.n_jobs)
        print("Fitting classifier (this may take a while)...")
        clf.fit(X_tr, y_tr)

        # predict probabilities on val set
        if hasattr(clf, "predict_proba"):
            try:
                probs_val = np.vstack([est.predict_proba(X_val)[:,1] for est in clf.estimators_]).T
            except Exception:
                # fallback to predict_proba on the wrapper
                probs_val = clf.predict_proba(X_val)
        else:
            probs_val = clf.predict_proba(X_val)

        # store OOF probs
        oof_probs[val_idx, :] = probs_val

        # compute per-fold thresholds and f1 for reporting
        thresholds = tune_thresholds_oof(probs_val, y_val, resolution=args.thr_resolution)
        preds = (probs_val >= np.array(thresholds)).astype(int)
        macro, per_label = macro_f1_from_preds(preds, y_val)
        print(f"Fold {fold} macro-F1: {macro:.4f} per-label: {per_label}")
        val_scores.append(macro)

        # save model and vectorizer for this fold
        fold_path = os.path.join(args.output_dir, f"model_fold{fold}.joblib")
        dump({"clf": clf, "vec_spec": vec_spec, "vec_obj": vec_obj}, fold_path)
        fold_paths.append(fold_path)
        print("Saved fold model to", fold_path)

    # After CV: tune thresholds on pooled OOF predictions
    print("\n=== OOF Threshold tuning ===")
    thresholds = tune_thresholds_oof(oof_probs, labels, resolution=args.thr_resolution)
    preds_oof = (oof_probs >= np.array(thresholds)).astype(int)
    oof_macro, oof_per_label = macro_f1_from_preds(preds_oof, labels)
    print("OOF macro-F1:", oof_macro, "per-label:", oof_per_label)
    print("Thresholds:", thresholds)

    # optional calibration on OOF probs
    calibrators = None
    if args.calibrate:
        print("Fitting isotonic calibrators on OOF predictions (no leak)")
        calibrators = fit_isotonic_calibrators(oof_probs, labels)
        # show a quick check
        calibrated_oof = apply_calibrators(calibrators, oof_probs)
        cal_preds = (calibrated_oof >= np.array(thresholds)).astype(int)
        cal_macro, cal_per_label = macro_f1_from_preds(cal_preds, labels)
        print("After calibration (using same thresholds) macro-F1:", cal_macro)

    # Train final model on full training set (vectorizer fit on full set)
    print("\nTraining final model on full data...")
    full_vec_spec = build_vectorizers(use_char=args.use_char_ngram, max_features=args.max_features,
                                      min_df=args.min_df, max_df=args.max_df,
                                      word_ngram=(1, args.word_ngram_max),
                                      char_ngram=(args.char_ngram_min, args.char_ngram_max))
    full_vec_obj, X_full = fit_vectorizer_on_texts(full_vec_spec, texts)
    final_clf = build_classifier(C=args.C, solver=args.solver, max_iter=args.max_iter, n_jobs=args.n_jobs)
    final_clf.fit(X_full, labels)

    final_path = os.path.join(args.output_dir, "final_model.joblib")
    dump({"clf": final_clf, "vec_spec": full_vec_spec, "vec_obj": full_vec_obj}, final_path)
    print("Saved final model to", final_path)

    # Save calibrators (if any), thresholds, and meta info
    meta = {
        "labels": args.labels,
        "thresholds": thresholds,
        "fold_models": fold_paths,
        "final_model": final_path,
        "n_splits": args.n_splits,
        "oof_macro_f1": float(oof_macro)
    }
    if calibrators is not None:
        calib_path = os.path.join(args.output_dir, "calibrators.joblib")
        dump(calibrators, calib_path)
        meta["calibrators"] = calib_path

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Training complete. Mean CV macro-F1 (per fold):", float(np.mean(val_scores)))
    print("Meta written to", os.path.join(args.output_dir, "meta.json"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--text_col", default="text")
    p.add_argument("--labels", nargs="+", default=["E","S","G","nonESG"])
    p.add_argument("--output_dir", default="outputs/baseline")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--max_features", type=int, default=100000)
    p.add_argument("--use_char_ngram", action="store_true")
    p.add_argument("--char_ngram_min", type=int, default=3)
    p.add_argument("--char_ngram_max", type=int, default=5)
    p.add_argument("--word_ngram_max", type=int, default=2)
    p.add_argument("--min_df", type=int, default=2)
    p.add_argument("--max_df", type=float, default=0.95)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--solver", default="saga")
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--n_jobs", type=int, default=1)
    p.add_argument("--thr_resolution", type=int, default=100)
    p.add_argument("--calibrate", action="store_true", help="Fit isotonic calibrators on OOF probs")
    args = p.parse_args()
    train(args)
