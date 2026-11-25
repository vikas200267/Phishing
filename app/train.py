from __future__ import annotations
import argparse, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .features import MODEL_FEATURE_NAMES, dataframe_to_features, normalize_url
from .domain import DOMAIN_GROUPING_VERSION, registrable_domain
from .policy import valid_http_url
from .url_model import URLModelBundle, _meta_features
from .registry import update_manifest

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / 'models'; DATA_DIR = ROOT / 'data'
MODEL_DIR.mkdir(exist_ok=True); DATA_DIR.mkdir(exist_ok=True)
SCHEMA_VERSION = 6


def sha256_file(path: str | None) -> str | None:
    if not path:
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def url_series(df: pd.DataFrame) -> pd.Series:
    out = []
    for _, r in df.iterrows():
        raw = r.get('url')
        if pd.isna(raw) or not str(raw).strip() or str(raw).strip().lower() in {'nan', 'none'}:
            raw = r.get('_normalized_url')
        if pd.notna(raw) and str(raw).strip() and str(raw).strip().lower() not in {'nan','none'}:
            out.append(normalize_url(str(raw)))
        else:
            out.append(normalize_url('', r.get('protocol'), r.get('domain_name'), r.get('address')))
    return pd.Series(out, index=df.index, dtype='string')


def load_data(phishing: str, legitimate: str, extra: str | None = None) -> pd.DataFrame:
    frames = []
    for path, label in [(phishing, 1), (legitimate, 0)]:
        df = pd.read_csv(path)
        df = df.drop(columns=['label'], errors='ignore')
        df['label'] = label
        frames.append(df)
    if extra:
        ex = pd.read_csv(extra)
        if 'label' not in ex.columns:
            raise ValueError('--extra CSV must contain a label column with 0/1 values')
        # Production training must not silently ingest hand-curated benign seed
        # rows. If a source column marks provenance, only explicitly permitted
        # real-feed sources are accepted here.
        if 'source' in ex.columns:
            src = ex['source'].astype(str).str.lower().str.strip()
            if (src == 'benign_seed').any():
                raise ValueError('manual benign_seed rows are prohibited in production training extras')
        ex['label'] = pd.to_numeric(ex['label'], errors='coerce')
        ex = ex[ex['label'].isin([0, 1])].copy()
        frames.append(ex)
    df = pd.concat(frames, ignore_index=True)
    df['_normalized_url'] = url_series(df)
    df = df[df['_normalized_url'].str.len() >= 8].copy()
    valid = df['_normalized_url'].map(valid_http_url)
    if (~valid).any():
        print(f'Excluded {int((~valid).sum())} URLs rejected by the serving URL policy.', flush=True)
    df = df[valid].copy()
    df = df.drop_duplicates(subset=['_normalized_url', 'label'])
    conflicts = df.groupby('_normalized_url')['label'].nunique()
    bad = set(conflicts[conflicts > 1].index)
    if bad:
        df = df[~df['_normalized_url'].isin(bad)]
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def domain_groups(df: pd.DataFrame) -> pd.Series:
    return df['_normalized_url'].map(registrable_domain)


def domain_set_from_csv(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p)
    if not any(c in d.columns for c in ('url','_normalized_url','domain_name')):
        raise ValueError(f'--exclude-domains-csv needs url/_normalized_url/domain_name: {p}')
    urls = url_series(d)
    domains = set(domain_groups(pd.DataFrame({'_normalized_url': urls})).astype(str)) - {''}
    if not domains:
        raise ValueError(f'--exclude-domains-csv contains no usable domains: {p}')
    return domains


def validate_temporal_sources(args) -> str:
    """Require hashed, source-attested event times for every training input."""
    metadata_path = getattr(args, 'temporal_source_metadata', None)
    if not metadata_path:
        raise ValueError('--require-temporal needs --temporal-source-metadata for every input file')
    meta = json.loads(Path(metadata_path).read_text(encoding='utf-8'))
    semantics = str(meta.get('timestamp_semantics') or '').lower()
    if meta.get('timestamp_provenance') not in {'source_attested', 'source_provided'} or not any(
        term in semantics for term in ('first seen', 'first-seen', 'first_seen', 'first observed', 'collection date', 'collected_at', 'submission time', 'verification time')
    ):
        raise ValueError('Temporal metadata must describe source-attested event times, not retrieval times')
    column = str(meta.get('timestamp_column') or '').strip()
    if not column:
        raise ValueError('Temporal metadata must name the event-time column')
    recorded = {str(Path(item['path']).resolve()): item.get('sha256') for item in meta.get('source_files', [])}
    for name in ('phishing', 'legitimate', 'extra'):
        path = getattr(args, name, None)
        if not path:
            continue
        expected = recorded.get(str(Path(path).resolve()))
        if not expected or sha256_file(path) != expected:
            raise ValueError(f'Temporal metadata does not verify the {name} input file')
    return column


def choose_split(df: pd.DataFrame, require_temporal: bool = False, timestamp_column: str | None = None):
    candidates = [timestamp_column] if timestamp_column else [c for c in ['event_time','first_seen','timestamp','created_at','observed_at','date','datetime','submission_time','verification_time','collected_at'] if c in df.columns]
    for c in candidates:
        if c not in df.columns:
            continue
        ts = pd.to_datetime(df[c], errors='coerce', utc=True)
        required_count = len(df) if require_temporal else max(100, int(.7 * len(df)))
        if ts.notna().sum() >= required_count and ts.dropna().nunique() > 1:
            valid = ts.dropna().sort_values()
            cut = int(len(valid) * .8)
            if 0 < cut < len(valid):
                # Split at a timestamp boundary, not an arbitrary row boundary;
                # otherwise rows observed at the same time can leak across sets.
                boundary = valid.iloc[cut]
                train_idx = ts[ts < boundary].index.to_numpy()
                test_idx = ts[ts >= boundary].index.to_numpy()
                train_urls = set(df.loc[train_idx, '_normalized_url'].astype(str))
                # Exact URL repeats in a later snapshot are memorization tests, not
                # temporal generalization. Drop those duplicates from the test fold.
                test_idx = np.asarray([
                    i for i in test_idx
                    if str(df.at[i, '_normalized_url']) not in train_urls
                ], dtype=int)
                if require_temporal:
                    train_domains = set(domain_groups(df.loc[train_idx]).astype(str))
                    test_idx = np.asarray([i for i in test_idx if registrable_domain(df.at[i, '_normalized_url']) not in train_domains], dtype=int)
                if len(train_idx) >= 100 and len(test_idx) >= 50:
                    return train_idx, test_idx, 'temporal', c
    if require_temporal:
        raise ValueError('Temporal validation was required but no sufficiently complete authentic timestamp column produced viable train/test sets; refusing to fabricate dates.')
    gss = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
    tr, te = next(gss.split(df, df['label'], groups=domain_groups(df)))
    return tr, te, 'unseen_domain', None


def make_vectorizers(args):
    return (
        TfidfVectorizer(analyzer='char', ngram_range=(2, 6), min_df=1, max_features=args.char_features, sublinear_tf=True, dtype=np.float32),
        TfidfVectorizer(analyzer='word', token_pattern=r'(?u)[a-zA-Z0-9]+', ngram_range=(1, 2), min_df=1, max_features=args.word_features, sublinear_tf=True, dtype=np.float32),
        TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=1, max_features=args.host_features, sublinear_tf=True, dtype=np.float32),
    )


def urls_to_host(urls: pd.Series | list[str]) -> list[str]:
    return [(urlparse(normalize_url(str(u))).hostname or '').lower() for u in urls]


def make_text_train_test(fit_urls, transform_urls, args):
    char_vec, word_vec, host_vec = make_vectorizers(args)
    svc_vec = TfidfVectorizer(analyzer='char', ngram_range=(2, 7), min_df=1, max_features=args.svc_features, sublinear_tf=True, dtype=np.float32)
    charwb_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 6), min_df=1, max_features=args.charwb_features, sublinear_tf=True, dtype=np.float32)
    fit_norm = [normalize_url(str(x)) for x in fit_urls]
    tr_norm = [normalize_url(str(x)) for x in transform_urls]
    char = char_vec.fit_transform(fit_norm)
    word = word_vec.fit_transform(fit_norm)
    host = host_vec.fit_transform(urls_to_host(fit_norm))
    svc = svc_vec.fit_transform(fit_norm)
    charwb = charwb_vec.fit_transform(fit_norm)
    text_fit = hstack([char, word], format='csr')
    char2 = char_vec.transform(tr_norm)
    word2 = word_vec.transform(tr_norm)
    host2 = host_vec.transform(urls_to_host(tr_norm))
    svc2 = svc_vec.transform(tr_norm)
    charwb2 = charwb_vec.transform(tr_norm)
    text_transform = hstack([char2, word2], format='csr')
    return text_fit, text_transform, host, host2, svc, svc2, charwb, charwb2, char_vec, word_vec, host_vec, svc_vec, charwb_vec


def evaluate(prob, y, threshold):
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'precision': float(precision_score(y, pred, zero_division=0)),
        'recall': float(recall_score(y, pred, zero_division=0)),
        'f1': float(f1_score(y, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else None,
        'pr_auc': float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else None,
        'brier_score': float(brier_score_loss(y, prob)),
        'false_positive_rate': float(fp / max(fp + tn, 1)),
        'false_negative_rate': float(fn / max(fn + tp, 1)),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
        'threshold': float(threshold),
    }


def source_weights(frame: pd.DataFrame, specifications: list[str] | None) -> np.ndarray:
    """Return explicit source weights without using evaluation labels or scores."""
    weights = np.ones(len(frame), dtype=np.float64)
    if not specifications:
        return weights
    if 'source' not in frame.columns:
        raise ValueError('Source weights require a source column in the training data')
    sources = frame['source'].fillna('').astype(str)
    seen = set()
    for specification in specifications:
        source, sep, value = specification.rpartition('=')
        if not sep or not source or source in seen:
            raise ValueError(f'Invalid or duplicate source weight: {specification!r}')
        try:
            factor = float(value)
        except ValueError as exc:
            raise ValueError(f'Invalid source weight: {specification!r}') from exc
        if not np.isfinite(factor) or not 0 < factor <= 100:
            raise ValueError('Source weights must be finite, greater than zero, and at most 100')
        mask = sources.eq(source).to_numpy()
        if not mask.any():
            raise ValueError(f'Weighted source not found in training data: {source}')
        weights[mask] = factor
        seen.add(source)
    return weights


def choose_operating_threshold(prob, y, min_precision=0.985, min_recall=0.95, lower=0.10, upper=0.75, max_fpr=None):
    # Robust threshold selection: avoid degenerate near-1.0 thresholds caused by
    # a small calibration fold. Prefer a balanced operating point that keeps both
    # precision and recall high, with a bounded search range.
    candidates = np.unique(np.clip(np.r_[np.linspace(lower, upper, 261), prob], lower, upper))
    # Compute all confusion matrices from sorted scores in O(n log n). Calling
    # sklearn's full metrics (including AUC) once per candidate is quadratic.
    p = np.asarray(prob, dtype=float).reshape(-1)
    labels = np.asarray(y, dtype=int).reshape(-1)
    if len(p) == 0 or len(p) != len(labels):
        raise ValueError('Threshold selection requires equally sized nonempty arrays')
    order = np.argsort(p, kind='stable')
    sorted_p, sorted_y = p[order], labels[order]
    positive_suffix = np.r_[np.cumsum((sorted_y == 1)[::-1])[::-1], 0]
    negative_suffix = np.r_[np.cumsum((sorted_y == 0)[::-1])[::-1], 0]
    indices = np.searchsorted(sorted_p, candidates, side='left')
    tp, fp = positive_suffix[indices], negative_suffix[indices]
    positives, negatives = positive_suffix[0], negative_suffix[0]
    precision = np.divide(tp, tp + fp, out=np.zeros(len(tp), dtype=float), where=(tp + fp) != 0)
    recall = tp / max(positives, 1)
    f1_denom = 2 * tp + fp + (positives - tp)
    f1 = np.divide(2 * tp, f1_denom, out=np.zeros(len(tp), dtype=float), where=f1_denom != 0)
    accuracy = (tp + negatives - fp) / len(labels)
    balanced = np.minimum(precision, recall)
    stability = -np.abs(candidates - .5)
    fpr = fp / max(negatives, 1)
    feasible_mask = (precision >= min_precision) & (recall >= min_recall)
    if max_fpr is not None:
        if not 0 <= max_fpr <= 1:
            raise ValueError('max_fpr must be between zero and one')
        feasible_mask &= fpr <= max_fpr
    feasible = np.flatnonzero(feasible_mask)
    if len(feasible):
        # argmax on an ordered tuple preserves the original first-candidate tie rule.
        best = max(feasible, key=lambda i: (balanced[i], f1[i], accuracy[i], stability[i]))
        mode = 'bounded_precision_recall'
    else:
        penalty = np.maximum(0, min_precision - precision) + np.maximum(0, min_recall - recall)
        if max_fpr is not None:
            penalty += np.maximum(0, fpr - max_fpr)
        score = f1 - .75 * penalty
        best = max(range(len(candidates)), key=lambda i: (score[i], balanced[i], accuracy[i], stability[i]))
        mode = 'bounded_balanced_fallback'
    threshold = float(candidates[best])
    return threshold, evaluate(p, labels, threshold), mode




def choose_suspicious_threshold(prob, y, phishing_threshold, target_recall=0.99):
    """Find the highest lower threshold that keeps overall phishing recall.

    Scores in [suspicious_threshold, phishing_threshold) become ``suspicious``
    instead of being incorrectly forced into ``benign``. Threshold is selected
    from the calibration fold only.
    """
    p = np.asarray(prob, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    positives = int((y == 1).sum())
    if len(p) != len(y) or positives == 0:
        return 0.0, {'target_recall': float(target_recall), 'achieved_recall': None, 'review_rate': None, 'reason': 'insufficient_calibration_data'}
    candidates = np.unique(np.r_[0.0, p[p < float(phishing_threshold)], float(phishing_threshold)])
    feasible = []
    for t in candidates:
        pred_positive_or_review = p >= t
        recall = float(((y == 1) & pred_positive_or_review).sum() / positives)
        if recall >= target_recall and t < phishing_threshold:
            benign_review_rate = float(((y == 0) & (p >= t) & (p < phishing_threshold)).sum() / max((y == 0).sum(), 1))
            feasible.append((float(t), recall, benign_review_rate))
    if not feasible:
        threshold = 0.0
        achieved = float((p >= threshold)[y == 1].mean())
        review_rate = float((((p >= threshold) & (p < phishing_threshold)) & (y == 0)).sum() / max((y == 0).sum(), 1))
        return threshold, {'target_recall': float(target_recall), 'achieved_recall': achieved, 'benign_review_rate': review_rate, 'reason': 'fallback_zero_threshold'}
    threshold, recall, review_rate = max(feasible, key=lambda x: (x[0], -x[2]))
    return threshold, {'target_recall': float(target_recall), 'achieved_recall': recall, 'benign_review_rate': review_rate, 'reason': 'calibration_recall_constraint'}


def choose_fast_thresholds(prob, y, *, min_negative_purity=0.995,
                           max_missed_positive_rate=0.005,
                           min_high_precision=0.995, min_support=12):
    """Choose high-purity shortcut regions using calibration-fold data only.

    The low region can bypass the ensemble only if its benign purity is high
    AND the fraction of all phishing examples it would miss is bounded. The
    high region can bypass only if its precision and sample support are enough.
    If the calibration corpus cannot prove these properties, shortcuts are
    disabled (0.0, 1.0) and all requests use the full calibrated ensemble.
    """
    p = np.asarray(prob, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=int).reshape(-1)
    if len(p) != len(y) or len(p) == 0 or len(np.unique(y)) < 2:
        return 0.0, 1.0, {
            'enabled': False, 'reason': 'insufficient_two-class_calibration_data',
            'calibration_samples': int(len(p)),
        }
    p = np.clip(p, 0.0, 1.0)
    positives = max(int((y == 1).sum()), 1)
    candidates = np.unique(np.r_[0.0, p, 1.0])

    low = 0.0
    low_support = 0
    low_purity = None
    low_missed_rate = None
    for t in candidates:
        mask = p <= t
        support = int(mask.sum())
        if support < min_support or support == 0:
            continue
        benign_purity = float((y[mask] == 0).sum() / support)
        missed_rate = float(((y == 1) & mask).sum() / positives)
        if benign_purity >= min_negative_purity and missed_rate <= max_missed_positive_rate:
            if t > low:
                low = float(t)
                low_support = support
                low_purity = benign_purity
                low_missed_rate = missed_rate

    high = 1.0
    high_support = 0
    high_precision = None
    for t in sorted(candidates):
        mask = p >= t
        support = int(mask.sum())
        if support < min_support or support == 0:
            continue
        precision = float((y[mask] == 1).sum() / support)
        if precision >= min_high_precision:
            high = float(t)
            high_support = support
            high_precision = precision
            break

    enabled = bool(0.0 < low < high < 1.0)
    if not enabled:
        low, high = 0.0, 1.0
    return low, high, {
        'enabled': enabled,
        'reason': 'calibration_constraints_met' if enabled else 'no_non_degenerate_shortcut_region_met_constraints',
        'calibration_samples': int(len(p)),
        'min_negative_purity': float(min_negative_purity),
        'max_missed_positive_rate': float(max_missed_positive_rate),
        'min_high_precision': float(min_high_precision),
        'min_support': int(min_support),
        'low_threshold': float(low),
        'low_support': int(low_support if enabled else 0),
        'low_benign_purity': low_purity if enabled else None,
        'low_missed_positive_rate': low_missed_rate if enabled else None,
        'high_threshold': float(high),
        'high_support': int(high_support if enabled else 0),
        'high_precision': high_precision if enabled else None,
    }


def _logit_probability(values):
    p = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p)).reshape(-1, 1)

def base_models(args):
    xgb_params = {
        'n_estimators': args.trees, 'max_depth': args.depth, 'learning_rate': args.lr,
        'subsample': .92, 'colsample_bytree': .9, 'min_child_weight': 2,
        'reg_alpha': .2, 'reg_lambda': 2.5, 'gamma': .04,
        'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist',
        'n_jobs': args.threads, 'random_state': 42,
    }
    lgb_params = {
        'n_estimators': args.lgb_trees, 'num_leaves': args.lgb_leaves, 'learning_rate': args.lgb_lr,
        'subsample': .92, 'colsample_bytree': .9, 'min_child_samples': 12,
        'reg_alpha': .15, 'reg_lambda': 2.2, 'verbosity': -1, 'n_jobs': args.threads, 'random_state': 42,
    }
    extra_params = {
        'n_estimators': args.extra_trees, 'max_depth': args.extra_depth, 'min_samples_leaf': 2,
        'max_features': .8, 'class_weight': 'balanced', 'n_jobs': args.threads, 'random_state': 42,
    }
    return (
        XGBClassifier(**xgb_params),
        LGBMClassifier(**lgb_params),
        ExtraTreesClassifier(**extra_params),
        LogisticRegression(C=args.text_C, max_iter=3000, class_weight='balanced', solver='liblinear'),
        LogisticRegression(C=args.host_C, max_iter=2500, class_weight='balanced', solver='liblinear'),
        LinearSVC(C=args.svc_C, class_weight='balanced', max_iter=6000),
        LogisticRegression(C=args.charwb_C, max_iter=2500, class_weight='balanced', solver='liblinear'),
    )


def fit_components(df, args, sample_weight=None, feature_names=None):
    feature_names = feature_names or MODEL_FEATURE_NAMES
    structured = dataframe_to_features(df)[feature_names].astype(np.float32)
    text_url, _, host, _, svc, _, charwb, _, char_vec, word_vec, host_vec, svc_vec, charwb_vec = make_text_train_test(
        df['_normalized_url'], df['_normalized_url'], args
    )
    models = base_models(args)
    xgb, lgbm, extra, text_model, host_model, svc_model, charwb_model = models
    fit_kw = {'sample_weight': sample_weight} if sample_weight is not None else {}
    xgb.fit(structured, df['label'].astype(int).to_numpy(), **fit_kw)
    lgbm.fit(structured, df['label'].astype(int).to_numpy(), **fit_kw)
    extra.fit(structured, df['label'].astype(int).to_numpy(), **fit_kw)
    text_model.fit(text_url, df['label'].astype(int).to_numpy(), **fit_kw)
    host_model.fit(host, df['label'].astype(int).to_numpy(), **fit_kw)
    svc_model.fit(svc, df['label'].astype(int).to_numpy(), **fit_kw)
    charwb_model.fit(charwb, df['label'].astype(int).to_numpy(), **fit_kw)
    return models, (char_vec, word_vec, host_vec, svc_vec, charwb_vec)


def component_predict(models, vectorizers, df, feature_names=None):
    feature_names = feature_names or MODEL_FEATURE_NAMES
    xgb, lgbm, extra, text_model, host_model, svc_model, charwb_model = models
    char_vec, word_vec, host_vec, svc_vec, charwb_vec = vectorizers
    structured = dataframe_to_features(df)[feature_names].astype(np.float32)
    norm = [normalize_url(str(x)) for x in df['_normalized_url']]
    char = char_vec.transform(norm); word = word_vec.transform(norm); host = host_vec.transform(urls_to_host(norm)); svc = svc_vec.transform(norm); charwb = charwb_vec.transform(norm)
    text = hstack([char, word], format='csr')
    from scipy.special import expit
    return np.column_stack([
        xgb.predict_proba(structured)[:, 1],
        lgbm.predict_proba(structured)[:, 1],
        extra.predict_proba(structured)[:, 1],
        text_model.predict_proba(text)[:, 1],
        host_model.predict_proba(host)[:, 1],
        expit(np.clip(svc_model.decision_function(svc), -20, 20)),
        charwb_model.predict_proba(charwb)[:,1],
    ])


def fit_fold_models(train_df, val_df, args, feature_names, sample_weight=None):
    models, vecs = fit_components(train_df, args, sample_weight=sample_weight, feature_names=feature_names)
    probs = component_predict(models, vecs, val_df, feature_names)
    return probs


def train(args):
    if getattr(args, 'promote', False):
        raise ValueError('Direct promotion during training is disabled; run scripts/promote_if_gates.py after independent evaluation.')
    if not 0 <= getattr(args, 'max_calibration_fpr', .01) <= 1:
        raise ValueError('max_calibration_fpr must be between zero and one')
    temporal_column = validate_temporal_sources(args) if getattr(args, 'require_temporal', False) else None
    df = load_data(args.phishing, args.legitimate, args.extra)
    feature_names = MODEL_FEATURE_NAMES
    if args.feature_schema_file:
        feature_names = json.loads(Path(args.feature_schema_file).read_text())
    frozen = DATA_DIR / 'frozen_regression.csv'
    exclusion_files = [frozen] if args.exclude_frozen else []
    if args.exclude_csv:
        exclusion_files.append(Path(args.exclude_csv))
    # Never silently skip a requested release-gate/frozen exclusion. A missing
    # file must fail training rather than accidentally allowing leakage.
    for exclusion in exclusion_files:
        if not exclusion.exists():
            raise FileNotFoundError(f'required exclusion file is missing: {exclusion}')
    if args.exclude_domains_csv and not Path(args.exclude_domains_csv).exists():
        raise FileNotFoundError(f'required domain-exclusion file is missing: {args.exclude_domains_csv}')
    source_row_count = int(len(df))
    excluded_urls = set()
    excluded_domains = set(domain_set_from_csv(args.exclude_domains_csv))
    exclusion_hashes = {}
    for exclusion in exclusion_files:
        ex = pd.read_csv(exclusion)
        if not any(col in ex.columns for col in ('url', 'domain_name', '_normalized_url')):
            raise ValueError(f'required exclusion file has no URL field: {exclusion}')
        exclusion_urls = url_series(ex).dropna().astype(str)
        exclusion_urls = exclusion_urls[exclusion_urls.map(valid_http_url)]
        if exclusion_urls.empty:
            raise ValueError(f'required exclusion file has no valid URLs: {exclusion}')
        excluded_urls.update(exclusion_urls.tolist())
        exclusion_hashes[str(exclusion)] = sha256_file(str(exclusion))
    if args.exclude_domains_csv:
        exclusion_hashes[str(args.exclude_domains_csv)] = sha256_file(args.exclude_domains_csv)
    before_exact = len(df)
    if excluded_urls:
        df = df[~df['_normalized_url'].isin(excluded_urls)].reset_index(drop=True)
    exact_excluded_rows = before_exact - len(df)
    if exact_excluded_rows:
        print(f'Excluded {exact_excluded_rows} exact evaluation URLs from candidate training data.')
    before_domain = len(df)
    if excluded_domains:
        candidate_domains = domain_groups(df)
        df = df[~candidate_domains.isin(excluded_domains)].reset_index(drop=True)
    domain_excluded_rows = before_domain - len(df)
    if domain_excluded_rows:
        print(f'Excluded {domain_excluded_rows} rows across {len(excluded_domains)} reserved evaluation domains.')
    prepared = df[['_normalized_url', 'label']].sort_values(['_normalized_url', 'label']).to_csv(index=False).encode('utf-8')
    prepared_dataset_sha256 = hashlib.sha256(prepared).hexdigest()
    if len(df) < 200 or df['label'].nunique() < 2:
        raise ValueError(f'Insufficient training data: {len(df)} rows / {df.label.nunique()} classes')

    outer_tr, outer_te, split_type, split_column = choose_split(df, require_temporal=args.require_temporal, timestamp_column=temporal_column)
    train_df = df.iloc[outer_tr].copy().reset_index(drop=True)
    test_df = df.iloc[outer_te].copy().reset_index(drop=True)
    if train_df['label'].nunique() < 2 or test_df['label'].nunique() < 2:
        raise ValueError('Outer split produced a single-class partition; add more diverse data.')

    weights = source_weights(train_df, args.source_weight)
    groups = domain_groups(train_df)
    n_splits = min(args.folds, max(3, groups.nunique() // 4))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=17)
    component_count = 7
    oof = np.zeros((len(train_df), component_count), dtype=np.float32)
    fold_id = np.full(len(train_df), -1, dtype=np.int32)
    for f, (tr_idx, va_idx) in enumerate(sgkf.split(train_df, train_df['label'], groups=groups)):
        fold_id[va_idx] = f
        tr = train_df.iloc[tr_idx].copy(); va = train_df.iloc[va_idx].copy()
        oof[va_idx] = fit_fold_models(tr, va, args, feature_names, sample_weight=weights[tr_idx])
        print(f'OOF fold {f+1}/{n_splits}: train={len(tr)} val={len(va)}')

    # Use the final OOF fold as an untouched calibration/threshold fold.
    cal_fold = n_splits - 1
    meta_idx = np.where(fold_id != cal_fold)[0]
    cal_idx = np.where(fold_id == cal_fold)[0]
    if len(cal_idx) < 50:
        cal_idx = meta_idx[-max(50, len(meta_idx)//6):]
        meta_idx = meta_idx[:-len(cal_idx)]
    meta_X = _meta_features(oof[meta_idx])
    cal_X = _meta_features(oof[cal_idx])
    y_train = train_df['label'].astype(int).to_numpy()

    meta_model = LogisticRegression(C=args.meta_C, max_iter=2000, class_weight='balanced')
    meta_model.fit(meta_X, y_train[meta_idx], sample_weight=weights[meta_idx])
    cal_raw = meta_model.predict_proba(cal_X)[:, 1]
    calibrator = LogisticRegression(C=10.0, max_iter=1500)
    calibrator.fit(cal_raw.reshape(-1, 1), y_train[cal_idx], sample_weight=weights[cal_idx])
    cal_prob = calibrator.predict_proba(cal_raw.reshape(-1, 1))[:, 1]
    threshold, cal_metrics, threshold_mode = choose_operating_threshold(cal_prob, y_train[cal_idx], min_precision=args.min_precision, min_recall=args.min_recall, lower=args.threshold_min, upper=args.threshold_max, max_fpr=args.max_calibration_fpr)
    calibration_gate_passed = bool(cal_metrics['precision'] >= args.min_precision and cal_metrics['recall'] >= args.min_recall and cal_metrics['false_positive_rate'] <= args.max_calibration_fpr)

    # Separately calibrate the structured fast-stage model from OOF predictions.
    # Never use the outer test/frozen/live feed to pick its thresholds.
    fast_calibrator = LogisticRegression(C=10.0, max_iter=1500)
    fast_calibrator.fit(_logit_probability(oof[cal_idx, 0]), y_train[cal_idx], sample_weight=weights[cal_idx])
    fast_cal_prob = fast_calibrator.predict_proba(_logit_probability(oof[cal_idx, 0]))[:, 1]
    fast_low, fast_high, fast_calibration_metadata = choose_fast_thresholds(
        fast_cal_prob, y_train[cal_idx],
        min_negative_purity=args.fast_min_negative_purity,
        max_missed_positive_rate=args.fast_max_missed_positive_rate,
        min_high_precision=args.fast_min_high_precision,
        min_support=args.fast_min_support,
    )
    suspicious_threshold, suspicious_calibration_metadata = choose_suspicious_threshold(
        cal_prob, y_train[cal_idx], threshold, target_recall=args.suspicious_target_recall
    )

    # Hard-negative weighting is derived only from OOF predictions, never from the outer test set.
    oof_mean = oof.mean(axis=1)
    full_oof_hard = ((oof_mean >= .5).astype(int) != y_train).astype(float)
    hard_false_positive = ((y_train == 0) & (oof_mean >= .35)).astype(float)
    train_struct = dataframe_to_features(train_df)
    lexical_hard = ((y_train == 0) & (
        (train_struct['suspicious_token_count'].to_numpy() >= 1) |
        (train_struct['brand_in_hostname'].to_numpy() >= 1) |
        (train_struct['path_depth'].to_numpy() >= 3) |
        (train_struct['query_param_count'].to_numpy() >= 3)
    )).astype(float)
    sample_weight = np.ones(len(train_df), dtype=np.float32)
    sample_weight += args.hard_negative_weight * full_oof_hard
    sample_weight += args.hard_negative_fp_weight * hard_false_positive
    sample_weight += args.lexical_hard_negative_weight * lexical_hard
    sample_weight = np.clip(sample_weight, 1.0, 6.0) * weights

    final_models, final_vecs = fit_components(train_df, args, sample_weight=sample_weight, feature_names=feature_names)
    test_components = component_predict(final_models, final_vecs, test_df, feature_names)
    test_meta = _meta_features(test_components)
    test_raw = meta_model.predict_proba(test_meta)[:, 1]
    test_prob = calibrator.predict_proba(test_raw.reshape(-1, 1))[:, 1]
    test_fast_prob = fast_calibrator.predict_proba(_logit_probability(test_components[:, 0]))[:, 1]
    test_cascade_prob = test_prob.copy()
    test_fast_mask = (test_fast_prob <= fast_low) | (test_fast_prob >= fast_high)
    # Degenerate thresholds 0/1 result in no shortcuts, intentionally.
    test_cascade_prob[test_fast_mask] = test_fast_prob[test_fast_mask]
    y_test = test_df['label'].astype(int).to_numpy()
    metrics = evaluate(test_prob, y_test, threshold)
    cascade_metrics = evaluate(test_cascade_prob, y_test, threshold)
    metrics['fast_path_evaluation'] = {
        'thresholds': {'low': float(fast_low), 'high': float(fast_high)},
        'calibration': fast_calibration_metadata,
        'outer_test': cascade_metrics,
        'outer_test_shortcut_rate': float(test_fast_mask.mean()) if len(test_fast_mask) else 0.0,
        'outer_test_shortcut_count': int(test_fast_mask.sum()),
        'outer_test_total': int(len(test_fast_mask)),
        'promoted_by_training_script': False,
    }
    train_domain_count = groups.nunique()
    test_domain_count = domain_groups(test_df).nunique()
    overlap = len(set(groups.unique()).intersection(set(domain_groups(test_df).unique())))
    exact_overlap = len(set(train_df['_normalized_url'].astype(str)).intersection(set(test_df['_normalized_url'].astype(str))))
    metrics.update({
        'schema_version': SCHEMA_VERSION, 'split_type': split_type, 'split_column': split_column,
        'samples': int(len(train_df) + len(test_df)), 'prepared_rows_after_exclusions': int(len(df)),
        'rows_without_timestamp_excluded_from_temporal_split': int(len(df) - len(train_df) - len(test_df)) if split_type == 'temporal' else 0,
        'exact_overlap_outer_split': int(exact_overlap),
        'train_samples': int(len(train_df)), 'test_samples': int(len(test_df)),
        'positive_rate': float(df.label.mean()), 'train_domains': int(train_domain_count), 'test_domains': int(test_domain_count),
        'domain_overlap_outer_split': int(overlap), 'oof_folds': int(n_splits), 'calibration_samples': int(len(cal_idx)),
        'threshold_mode': threshold_mode, 'component_models': ['xgboost_structured','lightgbm_structured','extratrees_structured','logistic_url_tfidf','logistic_host_tfidf','linear_svc_url_char','logistic_url_charwb'],
        'hard_negative_weight': float(args.hard_negative_weight), 'hard_negative_fp_weight': float(args.hard_negative_fp_weight), 'lexical_hard_negative_weight': float(args.lexical_hard_negative_weight), 'hard_negative_count': int(full_oof_hard.sum()), 'hard_false_positive_count': int(hard_false_positive.sum()), 'lexical_hard_negative_count': int(lexical_hard.sum()),
        'calibration_metrics': cal_metrics,
        'suspicious_threshold': float(suspicious_threshold),
        'suspicious_calibration': suspicious_calibration_metadata,
    })
    fast_recall_degradation = max(0.0, float(metrics['recall']) - float(cascade_metrics['recall']))
    fast_precision_degradation = max(0.0, float(metrics['precision']) - float(cascade_metrics['precision']))
    domain_split_is_valid = int(metrics.get('domain_overlap_outer_split', 1)) == 0
    exact_split_is_valid = int(metrics.get('exact_overlap_outer_split', 1)) == 0
    fast_path_validated = bool(
        fast_calibration_metadata.get('enabled')
        and fast_recall_degradation <= args.fast_max_recall_degradation
        and fast_precision_degradation <= args.fast_max_precision_degradation
        and domain_split_is_valid
        and exact_split_is_valid
    )
    fast_calibration_metadata['outer_test_gate'] = {
        'passed': fast_path_validated,
        'recall_degradation': fast_recall_degradation,
        'precision_degradation': fast_precision_degradation,
        'max_recall_degradation': args.fast_max_recall_degradation,
        'max_precision_degradation': args.fast_max_precision_degradation,
        'note': 'Outer holdout is used for release gating only; thresholds are chosen from calibration-fold data.'
    }
    metrics['fast_path_evaluation']['calibration'] = fast_calibration_metadata
    metrics['fast_path_evaluation']['enabled_for_serving'] = fast_path_validated

    version = datetime.now(timezone.utc).strftime('v%Y%m%d_%H%M%S')
    out = MODEL_DIR / version; out.mkdir(parents=True, exist_ok=False)
    bundle = URLModelBundle(
        final_models[0], final_models[1], final_models[2], final_models[3], final_models[4], final_models[5],
        final_vecs[0], final_vecs[1], final_vecs[2], final_vecs[3], meta_model, calibrator, threshold,
        feature_names, version, url_charwb_model=final_models[6], url_charwb_vectorizer=final_vecs[4],
        fast_calibrator=fast_calibrator, fast_low_threshold=fast_low,
        fast_high_threshold=fast_high, fast_calibration_metadata=fast_calibration_metadata,
        suspicious_threshold=suspicious_threshold, fast_path_validated=fast_path_validated
    )
    tmp_model = out / 'model.joblib.tmp'
    joblib.dump(bundle, tmp_model, compress=3)
    os.replace(tmp_model, out / 'model.joblib')
    (out / 'features.json').write_text(json.dumps(feature_names, indent=2))
    # Persist calibration scores without raw URLs so threshold selection can be
    # audited without reusing release holdouts for tuning.
    calibration_sources = train_df['source'].fillna('primary_input').astype(str).iloc[cal_idx].to_numpy() if 'source' in train_df else ['primary_input'] * len(cal_idx)
    pd.DataFrame({'label': y_train[cal_idx], 'probability': cal_prob, 'source': calibration_sources}).to_csv(out / 'calibration_scores.csv', index=False)
    pd.DataFrame({
        'url': test_df['_normalized_url'], 'label': y_test, 'probability': test_prob,
        'prediction': (test_prob >= threshold).astype(int)
    }).to_csv(out / 'validation_predictions.csv', index=False)
    # Persist the exact domains used by this model's training data. Live-feed
    # evaluation uses this to construct genuinely unseen-domain samples.
    training_domains = sorted(set(domain_groups(train_df).astype(str)))
    (out / 'training_domains.txt').write_text('\n'.join(training_domains) + ('\n' if training_domains else ''), encoding='utf-8')
    prepared_domains = sorted(set(domain_groups(df).astype(str)))
    (out / 'prepared_domains.txt').write_text('\n'.join(prepared_domains) + ('\n' if prepared_domains else ''), encoding='utf-8')
    metadata = {
        'version': version, 'schema_version': SCHEMA_VERSION, 'created_at': datetime.now(timezone.utc).isoformat(),
        'model_components': ['xgboost_structured','lightgbm_structured','extratrees_structured','logistic_url_tfidf','logistic_host_tfidf','linear_svc_url_char','logistic_url_charwb'],
        'model': 'XGBoost+LightGBM+ExtraTrees structured + URL char/word TFIDF + host char TFIDF + LinearSVC URL char + logistic meta-stack + Platt calibration',
        'feature_count': len(feature_names), 'feature_schema_file': f'models/{version}/features.json', 'char_max_features': args.char_features, 'word_max_features': args.word_features, 'charwb_max_features': args.charwb_features,
        'host_max_features': args.host_features, 'svc_max_features': args.svc_features, 'outer_split': split_type, 'split_column': split_column,
        'threshold': threshold, 'threshold_mode': threshold_mode, 'min_precision_constraint': args.min_precision,
        'suspicious_threshold': float(suspicious_threshold),
        'suspicious_calibration': suspicious_calibration_metadata,
        'fast_path': {
            'calibrated': True, 'calibration_method': 'logistic_on_xgboost_logit_oof',
            'low_threshold': float(fast_low), 'high_threshold': float(fast_high),
            'enabled': fast_path_validated,
            'calibration': fast_calibration_metadata,
            'outer_test_metrics': cascade_metrics,
        },
        'data_sources': {
            'phishing': {'path': args.phishing, 'sha256': sha256_file(args.phishing)},
            'legitimate': {'path': args.legitimate, 'sha256': sha256_file(args.legitimate)},
            'extra': ({'path': args.extra, 'sha256': sha256_file(args.extra)} if args.extra else None),
        },
        'exclusion_sources': {
            'files': [{'path': str(path), 'sha256': exclusion_hashes.get(str(path))} for path in exclusion_files],
            'domain_file': ({'path': args.exclude_domains_csv, 'sha256': exclusion_hashes.get(str(args.exclude_domains_csv))} if args.exclude_domains_csv else None),
            'frozen_regression_enabled': bool(args.exclude_frozen),
            'exact_urls_excluded': int(exact_excluded_rows),
            'domain_rows_excluded': int(domain_excluded_rows),
        },
        'prepared_dataset': {
            'sha256': prepared_dataset_sha256,
            'rows_before_exclusions': int(source_row_count),
            'rows_after_exclusions': int(len(df)),
            'label_counts_after_exclusions': {str(int(k)): int(v) for k, v in df['label'].value_counts().sort_index().items()},
            'unique_domain_groups_after_exclusions': int(domain_groups(df).nunique()),
        },
        'training_parameters': vars(args),
        'temporal_source_metadata_sha256': sha256_file(getattr(args, 'temporal_source_metadata', None)),
        'release_gate': args.exclude_domains_csv,
        'training_domain_count': len(training_domains),
        'domain_grouping_version': DOMAIN_GROUPING_VERSION,
        'training_domains_sha256': sha256_file(str(out / 'training_domains.txt')),
        'prepared_domains_sha256': sha256_file(str(out / 'prepared_domains.txt')),
        'validation_predictions_sha256': sha256_file(str(out / 'validation_predictions.csv')),
        'calibration_scores_sha256': sha256_file(str(out / 'calibration_scores.csv')),
        'synthetic_training_data': False,
        'realworld_data_policy': 'traceable_sources_only; no synthetic production rows; no fabricated timestamps',
        'frozen_regression_role': 'excluded_only_when_--exclude-frozen_is_set; historical reference otherwise',
        'promotion_eligible': False,
        'release_status': 'candidate_not_promoted',
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    # Evaluate the frozen regression set with the candidate, without adding it to training.
    frozen_metrics = None
    if frozen.exists():
        fr = pd.read_csv(frozen)
        if 'label' in fr.columns:
            fr['_normalized_url'] = url_series(fr)
            fr = fr[fr['_normalized_url'].str.len() >= 8].copy()
            fr_probs = bundle.predict_proba(fr['_normalized_url'].tolist())
            frozen_metrics = evaluate(fr_probs, fr['label'].astype(int).to_numpy(), threshold)
            (out / 'frozen_regression_metrics.json').write_text(json.dumps(frozen_metrics, indent=2))

    frozen_gate_passed = bool(
        frozen_metrics is not None
        and frozen_metrics['accuracy'] >= args.min_frozen_accuracy
        and frozen_metrics['precision'] >= args.min_frozen_precision
        and frozen_metrics['recall'] >= args.min_frozen_recall
    )
    outer_domain_gate = int(metrics.get('domain_overlap_outer_split', 1)) == 0
    outer_gate_passed = bool(
        metrics['precision'] >= args.min_precision
        and metrics['recall'] >= args.min_recall
        and outer_domain_gate
        and int(metrics.get('exact_overlap_outer_split', 1)) == 0
    )
    promotion_eligible = bool(calibration_gate_passed and outer_gate_passed and frozen_gate_passed)
    metadata['promotion_eligible'] = promotion_eligible
    metadata['release_status'] = 'promotion_eligible' if promotion_eligible else 'candidate_not_promoted'
    metadata['promotion_gate'] = {
        'outer_test_passed': outer_gate_passed,
        'calibration_constraints_passed': calibration_gate_passed,
        'frozen_regression_passed': frozen_gate_passed,
        'frozen_constraints': {
            'accuracy': args.min_frozen_accuracy,
            'precision': args.min_frozen_precision,
            'recall': args.min_frozen_recall,
        },
        'fast_path_enabled': fast_path_validated,
        'fast_path_may_be_disabled_without_rejecting_full_model': True,
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))

    # COMPLETE is written only after every release artifact exists and is readable.
    # Integrity is recorded only after COMPLETE, so interrupted training cannot pollute the manifest.
    (out / 'COMPLETE').write_text('ok\n')
    update_manifest(version)
    promoted = False
    result = {'version': version, 'promoted': promoted, 'promotion_eligible': promotion_eligible, 'metrics': metrics, 'frozen_metrics': frozen_metrics, 'model_dir': str(out)}
    print(json.dumps(result, indent=2))
    return version, metrics


def build_arg_parser():
    ap = argparse.ArgumentParser(description='PhishGuard v5 leakage-resistant URL training pipeline')
    ap.add_argument('--phishing', required=True); ap.add_argument('--legitimate', required=True); ap.add_argument('--extra')
    ap.add_argument('--trees', type=int, default=320); ap.add_argument('--depth', type=int, default=5); ap.add_argument('--lr', type=float, default=.03)
    ap.add_argument('--lgb-trees', type=int, default=300); ap.add_argument('--lgb-leaves', type=int, default=20); ap.add_argument('--lgb-lr', type=float, default=.028)
    ap.add_argument('--extra-trees', type=int, default=320); ap.add_argument('--extra-depth', type=int, default=18)
    ap.add_argument('--text-C', type=float, default=5.0); ap.add_argument('--host-C', type=float, default=4.0); ap.add_argument('--svc-C', type=float, default=1.0); ap.add_argument('--charwb-C', type=float, default=3.0); ap.add_argument('--meta-C', type=float, default=1.0)
    ap.add_argument('--char-features', type=int, default=30000); ap.add_argument('--word-features', type=int, default=14000); ap.add_argument('--host-features', type=int, default=18000); ap.add_argument('--svc-features', type=int, default=24000); ap.add_argument('--charwb-features', type=int, default=12000); ap.add_argument('--feature-schema-file')
    ap.add_argument('--folds', type=int, default=4); ap.add_argument('--hard-negative-weight', type=float, default=2.0); ap.add_argument('--hard-negative-fp-weight', type=float, default=1.2); ap.add_argument('--lexical-hard-negative-weight', type=float, default=.8); ap.add_argument('--min-precision', type=float, default=.985); ap.add_argument('--min-recall', type=float, default=.95); ap.add_argument('--threshold-min', type=float, default=.10); ap.add_argument('--threshold-max', type=float, default=.75)
    ap.add_argument('--source-weight', action='append', default=[], metavar='SOURCE=FACTOR', help='Repeat to weight a named training source in OOF, stack, calibration, and final fitting.')
    ap.add_argument('--max-calibration-fpr', type=float, default=.01, help='Maximum calibration false-positive rate for candidate eligibility.')
    ap.add_argument('--fast-min-negative-purity', type=float, default=.995); ap.add_argument('--fast-max-missed-positive-rate', type=float, default=.005); ap.add_argument('--fast-min-high-precision', type=float, default=.995); ap.add_argument('--fast-min-support', type=int, default=12); ap.add_argument('--fast-max-recall-degradation', type=float, default=.005); ap.add_argument('--fast-max-precision-degradation', type=float, default=.005); ap.add_argument('--suspicious-target-recall', type=float, default=.99); ap.add_argument('--min-frozen-accuracy', type=float, default=.98); ap.add_argument('--min-frozen-precision', type=float, default=.99); ap.add_argument('--min-frozen-recall', type=float, default=.98)
    ap.add_argument('--threads', type=int, default=max(2, min(8, os.cpu_count() or 4))); ap.add_argument('--require-temporal', action='store_true'); ap.add_argument('--temporal-source-metadata'); ap.add_argument('--exclude-csv'); ap.add_argument('--exclude-domains-csv'); ap.add_argument('--exclude-frozen', action='store_true'); ap.add_argument('--promote', action='store_true')
    return ap

def main_cli():
    train(build_arg_parser().parse_args())

if __name__ == '__main__':
    main_cli()
